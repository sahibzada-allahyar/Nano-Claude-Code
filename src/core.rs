use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use crate::edit::perform_edit;
use crate::shell::execute as execute_shell;
use crate::tools::{
    ToolUse, maybe_extract_bash_command, maybe_extract_edit_command, maybe_run_echo,
    parse_inline_edit_command,
};

pub const DEFAULT_MODEL: &str = "mock-claude";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunOutcome {
    pub assistant_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUse>,
}

pub fn run(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<RunOutcome> {
    if let Some((tool_use, assistant_text)) = maybe_run_echo(prompt) {
        return Ok(RunOutcome {
            assistant_text,
            tool_use: Some(tool_use),
        });
    }

    if let Some(command) = maybe_extract_bash_command(prompt) {
        let report = execute_shell(command, cwd, shell, env_vars)?;
        let assistant_text = if !report.stdout.is_empty() {
            report.stdout
        } else if !report.stderr.is_empty() {
            report.stderr
        } else {
            format!("command exited with {}", report.exit_code)
        };
        return Ok(RunOutcome {
            assistant_text,
            tool_use: Some(ToolUse {
                tool: "bash".to_string(),
                input: command.to_string(),
            }),
        });
    }

    if let Some(command) = maybe_extract_edit_command(prompt) {
        let args = parse_inline_edit_command(command)?;
        let file_path = resolve_against_cwd(cwd, &args.file_path);
        let _report = perform_edit(
            &file_path,
            &args.old_string,
            &args.new_string,
            args.replace_all,
        )?;
        return Ok(RunOutcome {
            assistant_text: format!("Edited {}", args.file_path),
            tool_use: Some(ToolUse {
                tool: "edit".to_string(),
                input: command.to_string(),
            }),
        });
    }

    let assistant_text = if prompt == "/tools" {
        "bash\nedit\necho".to_string()
    } else {
        format!("mock:{prompt}")
    };

    Ok(RunOutcome {
        assistant_text,
        tool_use: None,
    })
}

fn resolve_against_cwd(cwd: &Path, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}
