use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolUse {
    pub tool: String,
    pub input: String,
}

pub fn maybe_run_echo(prompt: &str) -> Option<(ToolUse, String)> {
    let rest = prompt.strip_prefix("/echo ")?;
    Some((
        ToolUse {
            tool: "echo".to_string(),
            input: rest.to_string(),
        },
        rest.to_string(),
    ))
}

pub fn maybe_extract_bash_command(prompt: &str) -> Option<&str> {
    prompt.strip_prefix("/bash ")
}

pub fn maybe_extract_edit_command(prompt: &str) -> Option<&str> {
    prompt.strip_prefix("/edit ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineEditCommand {
    pub file_path: String,
    pub old_string: String,
    pub new_string: String,
    pub replace_all: bool,
}

pub fn parse_inline_edit_command(input: &str) -> Result<InlineEditCommand> {
    let tokens = split_command_words(input)?;
    let mut file_path = None;
    let mut old_string = None;
    let mut new_string = None;
    let mut replace_all = false;
    let mut idx = 0;

    while idx < tokens.len() {
        let token = &tokens[idx];
        match token.as_str() {
            "--file" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --file");
                };
                file_path = Some(value.clone());
                idx += 2;
            }
            "--old" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --old");
                };
                old_string = Some(value.clone());
                idx += 2;
            }
            "--new" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --new");
                };
                new_string = Some(value.clone());
                idx += 2;
            }
            "--replace-all" => {
                replace_all = true;
                idx += 1;
            }
            _ => {
                if let Some(value) = token.strip_prefix("--file=") {
                    file_path = Some(value.to_string());
                    idx += 1;
                } else if let Some(value) = token.strip_prefix("--old=") {
                    old_string = Some(value.to_string());
                    idx += 1;
                } else if let Some(value) = token.strip_prefix("--new=") {
                    new_string = Some(value.to_string());
                    idx += 1;
                } else {
                    bail!("unknown edit flag: {token}");
                }
            }
        }
    }

    Ok(InlineEditCommand {
        file_path: file_path.ok_or_else(|| anyhow::anyhow!("edit requires --file"))?,
        old_string: old_string.ok_or_else(|| anyhow::anyhow!("edit requires --old"))?,
        new_string: new_string.ok_or_else(|| anyhow::anyhow!("edit requires --new"))?,
        replace_all,
    })
}

pub fn split_command_words(input: &str) -> Result<Vec<String>> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut idx = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut token_started = false;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            idx += 1;
            continue;
        }

        if in_double {
            if ch == '"' {
                in_double = false;
                idx += 1;
                continue;
            }
            if ch == '\\' {
                idx += 1;
                let Some(next) = chars.get(idx) else {
                    bail!("unterminated escape sequence");
                };
                current.push(*next);
                idx += 1;
                token_started = true;
                continue;
            }
            current.push(ch);
            idx += 1;
            token_started = true;
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                token_started = true;
                idx += 1;
            }
            '"' => {
                in_double = true;
                token_started = true;
                idx += 1;
            }
            '\\' => {
                idx += 1;
                let Some(next) = chars.get(idx) else {
                    bail!("unterminated escape sequence");
                };
                current.push(*next);
                idx += 1;
                token_started = true;
            }
            ch if ch.is_whitespace() => {
                if token_started {
                    tokens.push(std::mem::take(&mut current));
                    token_started = false;
                }
                idx += 1;
            }
            _ => {
                current.push(ch);
                token_started = true;
                idx += 1;
            }
        }
    }

    if in_single || in_double {
        bail!("unterminated quoted string");
    }

    if token_started {
        tokens.push(current);
    }

    Ok(tokens)
}
