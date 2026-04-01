use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;

use unicode_normalization::UnicodeNormalization;

pub fn normalize_nfc(input: &str) -> String {
    input.nfc().collect()
}

pub fn get_claude_config_home_dir() -> PathBuf {
    if let Ok(dir) = env::var("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(normalize_nfc(&dir));
    }

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(normalize_nfc(&format!("{home}/.claude")))
}

pub fn is_env_truthy(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };

    let normalized = value.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
}

pub fn parse_env_vars(raw_env_args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut parsed = BTreeMap::new();

    for env_str in raw_env_args {
        let mut parts = env_str.split('=');
        let key = parts.next().unwrap_or_default();
        let value_parts: Vec<&str> = parts.collect();
        if key.is_empty() || value_parts.is_empty() {
            return Err(format!(
                "Invalid environment variable format: {env_str}, environment variables should be added as: -e KEY1=value1 -e KEY2=value2"
            ));
        }

        parsed.insert(key.to_string(), value_parts.join("="));
    }

    Ok(parsed)
}
