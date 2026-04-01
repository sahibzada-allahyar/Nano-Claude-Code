use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ExecReport {
    pub cwd: String,
    pub shell: String,
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn default_shell() -> String {
    std::env::var("SHELL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "/bin/sh".to_string())
}

pub fn execute(
    command: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ExecReport> {
    let output = Command::new(shell)
        .arg("-lc")
        .arg(command)
        .current_dir(cwd)
        .envs(env_vars)
        .output()
        .with_context(|| format!("executing shell command via {shell}"))?;

    Ok(ExecReport {
        cwd: cwd.to_string_lossy().to_string(),
        shell: shell.to_string(),
        command: command.to_string(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}
