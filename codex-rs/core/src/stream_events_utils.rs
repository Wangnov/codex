use std::pin::Pin;
use std::sync::Arc;

use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::items::TurnItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use tokio_util::sync::CancellationToken;

use crate::codex::Session;
use crate::codex::TurnContext;
use crate::codex_delegate::run_codex_thread_one_shot;
use crate::error::CodexErr;
use crate::error::Result;
use crate::function_tool::FunctionCallError;
use crate::parse_turn_item;
use crate::reasoning_translation::TranslationSettings;
use crate::reasoning_translation::matches_target_language;
use crate::reasoning_translation::translation_settings_for;
use crate::reasoning_translation::translation_target_language;
use crate::tools::parallel::ToolCallRuntime;
use crate::tools::router::ToolRouter;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use futures::Future;
use tracing::debug;
use tracing::instrument;

/// Handle a completed output item from the model stream, recording it and
/// queuing any tool execution futures. This records items immediately so
/// history and rollout stay in sync even if the turn is later cancelled.
pub(crate) type InFlightFuture<'f> =
    Pin<Box<dyn Future<Output = Result<ResponseInputItem>> + Send + 'f>>;

#[derive(Default)]
pub(crate) struct OutputItemResult {
    pub last_agent_message: Option<String>,
    pub needs_follow_up: bool,
    pub tool_future: Option<InFlightFuture<'static>>,
}

pub(crate) struct HandleOutputCtx {
    pub sess: Arc<Session>,
    pub turn_context: Arc<TurnContext>,
    pub tool_runtime: ToolCallRuntime,
    pub cancellation_token: CancellationToken,
}

#[instrument(level = "trace", skip_all)]
pub(crate) async fn handle_output_item_done(
    ctx: &mut HandleOutputCtx,
    item: ResponseItem,
    previously_active_item: Option<TurnItem>,
) -> Result<OutputItemResult> {
    let mut output = OutputItemResult::default();

    match ToolRouter::build_tool_call(ctx.sess.as_ref(), item.clone()).await {
        // The model emitted a tool call; log it, persist the item immediately, and queue the tool execution.
        Ok(Some(call)) => {
            let payload_preview = call.payload.log_payload().into_owned();
            tracing::info!("ToolCall: {} {}", call.tool_name, payload_preview);

            ctx.sess
                .record_conversation_items(&ctx.turn_context, std::slice::from_ref(&item))
                .await;

            let cancellation_token = ctx.cancellation_token.child_token();
            let tool_future: InFlightFuture<'static> = Box::pin(
                ctx.tool_runtime
                    .clone()
                    .handle_tool_call(call, cancellation_token),
            );

            output.needs_follow_up = true;
            output.tool_future = Some(tool_future);
        }
        // No tool call: convert messages/reasoning into turn items and mark them as complete.
        Ok(None) => {
            if let Some(mut turn_item) = handle_non_tool_response_item(&item).await {
                if let TurnItem::Reasoning(mut reasoning_item) = turn_item {
                    let summary_text = std::mem::take(&mut reasoning_item.summary_text);
                    if let Some(translated_summary_text) =
                        maybe_translate_reasoning_summary(ctx, &summary_text).await
                    {
                        reasoning_item.summary_text = translated_summary_text;
                    } else {
                        reasoning_item.summary_text = summary_text;
                    }
                    turn_item = TurnItem::Reasoning(reasoning_item);
                }
                if previously_active_item.is_none() {
                    ctx.sess
                        .emit_turn_item_started(&ctx.turn_context, &turn_item)
                        .await;
                }

                ctx.sess
                    .emit_turn_item_completed(&ctx.turn_context, turn_item)
                    .await;
            }

            ctx.sess
                .record_conversation_items(&ctx.turn_context, std::slice::from_ref(&item))
                .await;
            let last_agent_message = last_assistant_message_from_item(&item);

            output.last_agent_message = last_agent_message;
        }
        // Guardrail: the model issued a LocalShellCall without an id; surface the error back into history.
        Err(FunctionCallError::MissingLocalShellCallId) => {
            let msg = "LocalShellCall without call_id or id";
            ctx.turn_context
                .client
                .get_otel_manager()
                .log_tool_failed("local_shell", msg);
            tracing::error!(msg);

            let response = ResponseInputItem::FunctionCallOutput {
                call_id: String::new(),
                output: FunctionCallOutputPayload {
                    content: msg.to_string(),
                    ..Default::default()
                },
            };
            ctx.sess
                .record_conversation_items(&ctx.turn_context, std::slice::from_ref(&item))
                .await;
            if let Some(response_item) = response_input_to_response_item(&response) {
                ctx.sess
                    .record_conversation_items(
                        &ctx.turn_context,
                        std::slice::from_ref(&response_item),
                    )
                    .await;
            }

            output.needs_follow_up = true;
        }
        // The tool request should be answered directly (or was denied); push that response into the transcript.
        Err(FunctionCallError::RespondToModel(message)) => {
            let response = ResponseInputItem::FunctionCallOutput {
                call_id: String::new(),
                output: FunctionCallOutputPayload {
                    content: message,
                    ..Default::default()
                },
            };
            ctx.sess
                .record_conversation_items(&ctx.turn_context, std::slice::from_ref(&item))
                .await;
            if let Some(response_item) = response_input_to_response_item(&response) {
                ctx.sess
                    .record_conversation_items(
                        &ctx.turn_context,
                        std::slice::from_ref(&response_item),
                    )
                    .await;
            }

            output.needs_follow_up = true;
        }
        // A fatal error occurred; surface it back into history.
        Err(FunctionCallError::Fatal(message)) => {
            return Err(CodexErr::Fatal(message));
        }
    }

    Ok(output)
}

pub(crate) async fn handle_non_tool_response_item(item: &ResponseItem) -> Option<TurnItem> {
    debug!(?item, "Output item");

    match item {
        ResponseItem::Message { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::WebSearchCall { .. } => parse_turn_item(item),
        ResponseItem::FunctionCallOutput { .. } | ResponseItem::CustomToolCallOutput { .. } => {
            debug!("unexpected tool output from stream");
            None
        }
        _ => None,
    }
}

async fn maybe_translate_reasoning_summary(
    ctx: &HandleOutputCtx,
    summary_text: &[String],
) -> Option<Vec<String>> {
    if summary_text.is_empty() {
        return None;
    }
    let joined = summary_text.join("\n\n");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        return None;
    }
    let config = ctx.turn_context.client.config();
    if config.hide_agent_reasoning {
        return None;
    }
    let current_model = ctx.turn_context.client.get_model();
    let translation_settings = translation_settings_for(config.as_ref(), &current_model)?;
    let target_language = translation_target_language(config.as_ref());
    if matches_target_language(trimmed, target_language) == Some(true) {
        return None;
    }

    let translated =
        translate_text_with_model(ctx, &translation_settings, target_language, trimmed).await?;
    let translated = translated.trim();
    if translated.is_empty() {
        return None;
    }
    Some(vec![translated.to_string()])
}

async fn translate_text_with_model(
    ctx: &HandleOutputCtx,
    settings: &TranslationSettings,
    target_language: &str,
    text: &str,
) -> Option<String> {
    let config = ctx.turn_context.client.config();
    let mut sub_agent_config = config.as_ref().clone();
    sub_agent_config.model = Some(settings.model.clone());
    sub_agent_config.model_provider_id = settings.provider_id.clone();
    sub_agent_config.model_provider = settings.provider.clone();
    sub_agent_config.review_model = None;
    sub_agent_config.base_instructions = Some(translation_instructions(target_language));
    sub_agent_config.developer_instructions = None;
    sub_agent_config.user_instructions = None;
    sub_agent_config.model_reasoning_summary = ReasoningSummary::None;
    sub_agent_config.model_reasoning_effort = None;
    sub_agent_config.hide_agent_reasoning = true;
    sub_agent_config.show_raw_agent_reasoning = false;
    sub_agent_config.web_search_mode = Some(WebSearchMode::Disabled);

    let input = vec![UserInput::Text {
        text: text.to_string(),
        text_elements: Vec::new(),
    }];
    let io = run_codex_thread_one_shot(
        sub_agent_config,
        ctx.sess.services.auth_manager.clone(),
        ctx.sess.services.models_manager.clone(),
        input,
        ctx.sess.clone(),
        ctx.turn_context.clone(),
        ctx.cancellation_token.child_token(),
        None,
    )
    .await
    .ok()?;

    let mut last_message = None;
    while let Ok(event) = io.next_event().await {
        match event.msg {
            EventMsg::AgentMessage(message) => {
                last_message = Some(message.message);
            }
            EventMsg::TurnComplete(_) => break,
            EventMsg::TurnAborted(_) | EventMsg::StreamError(_) => return None,
            _ => {}
        }
    }
    last_message
}

fn translation_instructions(target_language: &str) -> String {
    format!(
        "Translate the user's text into {target_language}. Preserve markdown and code formatting. Reply with only the translated text."
    )
}

pub(crate) fn last_assistant_message_from_item(item: &ResponseItem) -> Option<String> {
    if let ResponseItem::Message { role, content, .. } = item
        && role == "assistant"
    {
        return content.iter().rev().find_map(|ci| match ci {
            codex_protocol::models::ContentItem::OutputText { text } => Some(text.clone()),
            _ => None,
        });
    }
    None
}

pub(crate) fn response_input_to_response_item(input: &ResponseInputItem) -> Option<ResponseItem> {
    match input {
        ResponseInputItem::FunctionCallOutput { call_id, output } => {
            Some(ResponseItem::FunctionCallOutput {
                call_id: call_id.clone(),
                output: output.clone(),
            })
        }
        ResponseInputItem::CustomToolCallOutput { call_id, output } => {
            Some(ResponseItem::CustomToolCallOutput {
                call_id: call_id.clone(),
                output: output.clone(),
            })
        }
        ResponseInputItem::McpToolCallOutput { call_id, result } => {
            let output = match result {
                Ok(call_tool_result) => FunctionCallOutputPayload::from(call_tool_result),
                Err(err) => FunctionCallOutputPayload {
                    content: err.clone(),
                    success: Some(false),
                    ..Default::default()
                },
            };
            Some(ResponseItem::FunctionCallOutput {
                call_id: call_id.clone(),
                output,
            })
        }
        _ => None,
    }
}
