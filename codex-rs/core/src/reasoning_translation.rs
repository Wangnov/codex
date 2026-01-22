use crate::ModelProviderInfo;
use crate::config::Config;
use whatlang::Lang;
use whatlang::detect;

pub(crate) struct TranslationSettings {
    pub model: String,
    pub provider_id: String,
    pub provider: ModelProviderInfo,
}

pub(crate) fn translation_settings_for(
    config: &Config,
    current_model: &str,
) -> Option<TranslationSettings> {
    let model = config.reasoning_translation_model.as_deref()?;
    let trimmed = model.trim();
    if trimmed.is_empty() || trimmed == current_model {
        return None;
    }
    let provider_id = config
        .reasoning_translation_model_provider
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(config.model_provider_id.as_str());
    let provider = config.model_providers.get(provider_id)?.clone();
    Some(TranslationSettings {
        model: trimmed.to_string(),
        provider_id: provider_id.to_string(),
        provider,
    })
}

pub(crate) fn should_translate_reasoning_summary(config: &Config, current_model: &str) -> bool {
    if config.hide_agent_reasoning {
        return false;
    }
    translation_settings_for(config, current_model).is_some()
}

pub(crate) fn translation_target_language(config: &Config) -> &str {
    config.reasoning_translation_target_language.as_str()
}

pub(crate) fn matches_target_language(text: &str, target_language: &str) -> Option<bool> {
    let info = detect(text)?;
    if !info.is_reliable() {
        return None;
    }
    let target_lang = normalize_target_language(target_language)?;
    Some(info.lang() == target_lang)
}

fn normalize_target_language(target_language: &str) -> Option<Lang> {
    let trimmed = target_language.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = trimmed.split(['-', '_']).next().map(str::to_lowercase)?;
    if base.len() == 3 {
        return Lang::from_code(base);
    }
    if base.len() != 2 {
        return None;
    }
    match base.as_str() {
        "en" => Some(Lang::Eng),
        "zh" => Some(Lang::Cmn),
        "es" => Some(Lang::Spa),
        "pt" => Some(Lang::Por),
        "fr" => Some(Lang::Fra),
        "de" => Some(Lang::Deu),
        "ru" => Some(Lang::Rus),
        "ja" => Some(Lang::Jpn),
        "ko" => Some(Lang::Kor),
        "it" => Some(Lang::Ita),
        "nl" => Some(Lang::Nld),
        "sv" => Some(Lang::Swe),
        "no" => Some(Lang::Nob),
        "da" => Some(Lang::Dan),
        "fi" => Some(Lang::Fin),
        "pl" => Some(Lang::Pol),
        "uk" => Some(Lang::Ukr),
        "ar" => Some(Lang::Ara),
        "hi" => Some(Lang::Hin),
        "tr" => Some(Lang::Tur),
        "el" => Some(Lang::Ell),
        "he" => Some(Lang::Heb),
        "id" => Some(Lang::Ind),
        "vi" => Some(Lang::Vie), // codespell:ignore Vie
        "th" => Some(Lang::Tha), // codespell:ignore Tha
        "fa" => Some(Lang::Pes),
        "ur" => Some(Lang::Urd),
        "bn" => Some(Lang::Ben),
        "ta" => Some(Lang::Tam),
        "te" => Some(Lang::Tel), // codespell:ignore te
        "mr" => Some(Lang::Mar),
        "gu" => Some(Lang::Guj),
        "kn" => Some(Lang::Kan),
        "ml" => Some(Lang::Mal),
        "or" => Some(Lang::Ori),
        "pa" => Some(Lang::Pan),
        "uz" => Some(Lang::Uzb),
        "az" => Some(Lang::Aze),
        "hu" => Some(Lang::Hun),
        "cs" => Some(Lang::Ces),
        "ro" => Some(Lang::Ron),
        "bg" => Some(Lang::Bul),
        "be" => Some(Lang::Bel),
        "hr" => Some(Lang::Hrv),
        "sr" => Some(Lang::Srp),
        "mk" => Some(Lang::Mkd),
        "lt" => Some(Lang::Lit),
        "lv" => Some(Lang::Lav),
        "et" => Some(Lang::Est),
        "ka" => Some(Lang::Kat),
        "am" => Some(Lang::Amh),
        "jv" => Some(Lang::Jav),
        "my" => Some(Lang::Mya),
        "ne" => Some(Lang::Nep),
        "si" => Some(Lang::Sin),
        "km" => Some(Lang::Khm),
        "tk" => Some(Lang::Tuk),
        _ => None,
    }
}
