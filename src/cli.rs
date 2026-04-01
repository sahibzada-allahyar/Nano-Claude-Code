use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::io::{Read, stdin};
use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::agent::{
    AgentRecord, AgentTypeInfo, built_in_agents, close_agent, list_agent_records, send_agent,
    spawn_agent, wait_agent,
};
use crate::config::parse_env_vars;
use crate::core::{DEFAULT_MODEL, run as run_core};
use crate::edit::{EditReport, perform_edit};
use crate::session::{
    ChatReport, ListSessionsOptions, RunReport, SessionHandle, append_turn, build_chat_report,
    get_session_info, get_session_transcript_scan, list_sessions, load_session_handle, now_ms,
    write_session,
};
use crate::shell::{ExecReport, default_shell, execute as execute_shell};

#[derive(Debug)]
pub struct CliError {
    message: String,
    exit_code: i32,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 2,
        }
    }

    fn internal(error: anyhow::Error) -> Self {
        Self {
            message: error.to_string(),
            exit_code: 1,
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CliError {}

#[derive(Debug)]
struct RunArgs {
    cwd: PathBuf,
    shell: String,
    model: String,
    prompt: String,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct ChatArgs {
    cwd: Option<PathBuf>,
    shell: String,
    model: String,
    resume_session_id: Option<String>,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct PsArgs {
    dir: Option<PathBuf>,
    limit: Option<usize>,
    offset: usize,
    json: bool,
}

#[derive(Debug)]
struct InfoArgs {
    dir: Option<PathBuf>,
    session_id: String,
    json: bool,
}

#[derive(Debug)]
struct TranscriptArgs {
    dir: Option<PathBuf>,
    session_id: String,
    json: bool,
}

#[derive(Debug)]
struct ExecArgs {
    cwd: PathBuf,
    shell: String,
    command: String,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct EditArgs {
    file_path: PathBuf,
    old_string: String,
    new_string: String,
    replace_all: bool,
    json: bool,
}

#[derive(Debug)]
struct AgentSpawnArgs {
    cwd: PathBuf,
    agent_type: String,
    prompt: String,
    json: bool,
}

#[derive(Debug)]
struct AgentWaitArgs {
    agent_id: String,
    json: bool,
}

#[derive(Debug)]
struct AgentSendArgs {
    agent_id: String,
    prompt: String,
    json: bool,
}

#[derive(Debug)]
struct AgentCloseArgs {
    agent_id: String,
    json: bool,
}

pub fn run(argv: Vec<String>) -> Result<(), CliError> {
    let args = argv.get(1..).unwrap_or(&[]);

    if args.len() == 1 && matches!(args[0].as_str(), "--version" | "-v" | "-V") {
        println!("0.1.0 (nano-claude-code)");
        return Ok(());
    }

    match args.first().map(String::as_str) {
        Some("run") => {
            let cmd = parse_run_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_run(cmd).map_err(CliError::internal)?;
            print_run(report, json)
        }
        Some("chat") => {
            let cmd = parse_chat_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_chat(cmd)?;
            print_chat(report, json)
        }
        Some("ps") => {
            let cmd = parse_ps_args(&args[1..])?;
            let sessions = list_sessions(ListSessionsOptions {
                dir: cmd.dir,
                limit: cmd.limit,
                offset: cmd.offset,
            })
            .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&sessions)
                        .map_err(|err| CliError::internal(err.into()))?
                );
            } else {
                for session in sessions {
                    println!("{}\t{}", session.session_id, session.summary);
                }
            }
            Ok(())
        }
        Some("info") => {
            let cmd = parse_info_args(&args[1..])?;
            let info = get_session_info(&cmd.session_id, cmd.dir.as_deref())
                .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&info).map_err(|err| CliError::internal(err.into()))?
                );
            } else if let Some(info) = info {
                println!("{}", info.summary);
            }
            Ok(())
        }
        Some("transcript") => {
            let cmd = parse_transcript_args(&args[1..])?;
            let scan = get_session_transcript_scan(&cmd.session_id, cmd.dir.as_deref())
                .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&scan).map_err(|err| CliError::internal(err.into()))?
                );
            } else if let Some(scan) = scan {
                print!("{}", scan.transcript);
            }
            Ok(())
        }
        Some("exec") => {
            let cmd = parse_exec_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_exec(cmd).map_err(CliError::internal)?;
            print_exec(report, json)
        }
        Some("edit") => {
            let cmd = parse_edit_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_edit(cmd).map_err(CliError::internal)?;
            print_edit(report, json)
        }
        Some("agent") => run_agent_command(&args[1..]),
        Some(other) => Err(CliError::usage(format!("unknown command: {other}"))),
        None => Err(CliError::usage("expected a command")),
    }
}

pub fn eager_parse_cli_flag(flag_name: &str, argv: &[String]) -> Option<String> {
    for (idx, arg) in argv.iter().enumerate() {
        if let Some(rest) = arg.strip_prefix(&format!("{flag_name}=")) {
            return Some(rest.to_string());
        }

        if arg == flag_name {
            return argv.get(idx + 1).cloned();
        }
    }

    None
}

pub fn extract_args_after_double_dash(
    command_or_value: &str,
    args: &[String],
) -> (String, Vec<String>) {
    if command_or_value == "--" && !args.is_empty() {
        return (args[0].clone(), args[1..].to_vec());
    }

    (command_or_value.to_string(), args.to_vec())
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("run requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("run requires a prompt"));
    }

    Ok(RunArgs {
        cwd,
        shell,
        model,
        prompt,
        env_vars,
        json,
    })
}

fn parse_chat_args(args: &[String]) -> Result<ChatArgs, CliError> {
    let cwd = parse_string_flag("--cwd", args)?.map(PathBuf::from);
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let resume_session_id = parse_string_flag("--resume", args)?;
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_chat_positionals(args)?;
    if !positionals.is_empty() {
        return Err(CliError::usage(
            "chat does not accept prompt arguments; pipe prompts on stdin",
        ));
    }

    Ok(ChatArgs {
        cwd,
        shell,
        model,
        resume_session_id,
        env_vars,
        json,
    })
}

fn parse_ps_args(args: &[String]) -> Result<PsArgs, CliError> {
    let all = args.iter().any(|arg| arg == "--all");
    let dir = if all {
        None
    } else {
        Some(
            eager_parse_cli_flag("--cwd", args)
                .map(PathBuf::from)
                .or_else(|| std::env::current_dir().ok())
                .ok_or_else(|| CliError::usage("could not resolve current directory"))?,
        )
    };
    let limit = parse_usize_flag("--limit", args)?;
    let offset = parse_usize_flag("--offset", args)?.unwrap_or(0);
    let json = args.iter().any(|arg| arg == "--json");
    Ok(PsArgs {
        dir,
        limit,
        offset,
        json,
    })
}

fn parse_info_args(args: &[String]) -> Result<InfoArgs, CliError> {
    let dir = eager_parse_cli_flag("--cwd", args).map(PathBuf::from);
    let positionals = collect_positionals(args)?;
    let Some(session_id) = positionals.first() else {
        return Err(CliError::usage("info requires a session id"));
    };
    let json = args.iter().any(|arg| arg == "--json");
    Ok(InfoArgs {
        dir,
        session_id: session_id.clone(),
        json,
    })
}

fn parse_transcript_args(args: &[String]) -> Result<TranscriptArgs, CliError> {
    let dir = eager_parse_cli_flag("--cwd", args).map(PathBuf::from);
    let positionals = collect_positionals(args)?;
    let Some(session_id) = positionals.first() else {
        return Err(CliError::usage("transcript requires a session id"));
    };
    let json = args.iter().any(|arg| arg == "--json");
    Ok(TranscriptArgs {
        dir,
        session_id: session_id.clone(),
        json,
    })
}

fn parse_exec_args(args: &[String]) -> Result<ExecArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_exec_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("exec requires a command"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut command_parts = vec![command];
    command_parts.extend(rest);
    let command = command_parts.join(" ").trim().to_string();
    if command.is_empty() {
        return Err(CliError::usage("exec requires a command"));
    }

    Ok(ExecArgs {
        cwd,
        shell,
        command,
        env_vars,
        json,
    })
}

fn parse_edit_args(args: &[String]) -> Result<EditArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let raw_file_path = eager_parse_cli_flag("--file", args)
        .ok_or_else(|| CliError::usage("edit requires --file"))?;
    let old_string = eager_parse_cli_flag("--old", args)
        .ok_or_else(|| CliError::usage("edit requires --old"))?;
    let new_string = eager_parse_cli_flag("--new", args)
        .ok_or_else(|| CliError::usage("edit requires --new"))?;
    let replace_all = args.iter().any(|arg| arg == "--replace-all");
    let json = args.iter().any(|arg| arg == "--json");

    Ok(EditArgs {
        file_path: resolve_against_cwd(&cwd, &raw_file_path),
        old_string,
        new_string,
        replace_all,
        json,
    })
}

fn parse_agent_spawn_args(args: &[String]) -> Result<AgentSpawnArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let agent_type = eager_parse_cli_flag("--type", args)
        .ok_or_else(|| CliError::usage("agent spawn requires --type"))?;
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_agent_spawn_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("agent spawn requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("agent spawn requires a prompt"));
    }

    Ok(AgentSpawnArgs {
        cwd,
        agent_type,
        prompt,
        json,
    })
}

fn parse_agent_wait_args(args: &[String]) -> Result<AgentWaitArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent wait requires an agent id"));
    };

    Ok(AgentWaitArgs {
        agent_id: agent_id.clone(),
        json,
    })
}

fn parse_agent_send_args(args: &[String]) -> Result<AgentSendArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent send requires an agent id"));
    };
    let Some(first) = positionals.get(1) else {
        return Err(CliError::usage("agent send requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[2..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("agent send requires a prompt"));
    }

    Ok(AgentSendArgs {
        agent_id: agent_id.clone(),
        prompt,
        json,
    })
}

fn parse_agent_close_args(args: &[String]) -> Result<AgentCloseArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent close requires an agent id"));
    };

    Ok(AgentCloseArgs {
        agent_id: agent_id.clone(),
        json,
    })
}

fn collect_env_values(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut values = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "-e" | "--env" => {
                let Some(value) = args.get(idx + 1) else {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                };
                values.push(value.clone());
                idx += 2;
            }
            arg if arg.starts_with("--env=") => {
                values.push(arg["--env=".len()..].to_string());
                idx += 1;
            }
            _ => idx += 1,
        }
    }

    Ok(values)
}

fn collect_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_chat_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "--resume" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--resume=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_exec_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_agent_spawn_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--type" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=") || arg.starts_with("--type=") => idx += 1,
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }
    Ok(positionals)
}

fn parse_usize_flag(flag_name: &str, args: &[String]) -> Result<Option<usize>, CliError> {
    let Some(raw) = eager_parse_cli_flag(flag_name, args) else {
        return Ok(None);
    };

    raw.parse::<usize>()
        .map(Some)
        .map_err(|_| CliError::usage(format!("invalid value for {flag_name}: {raw}")))
}

fn parse_string_flag(flag_name: &str, args: &[String]) -> Result<Option<String>, CliError> {
    for (idx, arg) in args.iter().enumerate() {
        if let Some(rest) = arg.strip_prefix(&format!("{flag_name}=")) {
            return Ok(Some(rest.to_string()));
        }

        if arg == flag_name {
            let Some(value) = args.get(idx + 1) else {
                return Err(CliError::usage(format!("missing value for {flag_name}")));
            };
            return Ok(Some(value.clone()));
        }
    }

    Ok(None)
}

fn execute_run(args: RunArgs) -> Result<RunReport> {
    let outcome = run_core(&args.prompt, &args.cwd, &args.shell, &args.env_vars)?;
    let persisted = write_session(&args.cwd, &args.model, &args.prompt, &outcome)
        .with_context(|| format!("persisting run for {}", args.cwd.display()))?;
    Ok(persisted.report)
}

fn print_run(report: RunReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("{}", report.assistant_text);
    }
    Ok(())
}

fn execute_chat(args: ChatArgs) -> Result<ChatReport, CliError> {
    let prompts = read_chat_prompts()?;
    if prompts.is_empty() {
        return Err(CliError::usage(
            "chat requires at least one prompt on stdin",
        ));
    }

    let mut assistant_texts = Vec::new();

    let handle = if let Some(session_id) = args.resume_session_id.as_deref() {
        let handle = load_session_handle(session_id, args.cwd.as_deref())
            .map_err(CliError::internal)?
            .ok_or_else(|| {
                CliError::internal(anyhow::anyhow!("session not found: {session_id}"))
            })?;
        let cwd = PathBuf::from(&handle.cwd);
        let base_ms = now_ms();
        for (idx, prompt) in prompts.iter().enumerate() {
            let outcome =
                run_core(prompt, &cwd, &args.shell, &args.env_vars).map_err(CliError::internal)?;
            append_turn(&handle, prompt, &outcome, base_ms + idx as i64)
                .map_err(CliError::internal)?;
            assistant_texts.push(outcome.assistant_text);
        }
        handle
    } else {
        let cwd = args
            .cwd
            .or_else(|| std::env::current_dir().ok())
            .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
        let first_prompt = prompts.first().expect("first prompt");
        let first_outcome = run_core(first_prompt, &cwd, &args.shell, &args.env_vars)
            .map_err(CliError::internal)?;
        let persisted = write_session(&cwd, &args.model, first_prompt, &first_outcome)
            .map_err(CliError::internal)?;
        assistant_texts.push(first_outcome.assistant_text);
        let handle = SessionHandle {
            session_id: persisted.report.session_id,
            session_path: PathBuf::from(persisted.report.session_path),
            cwd: persisted.report.cwd,
            model: persisted.report.model,
        };
        let base_ms = now_ms();
        for (idx, prompt) in prompts.iter().enumerate().skip(1) {
            let outcome =
                run_core(prompt, &cwd, &args.shell, &args.env_vars).map_err(CliError::internal)?;
            append_turn(&handle, prompt, &outcome, base_ms + idx as i64)
                .map_err(CliError::internal)?;
            assistant_texts.push(outcome.assistant_text);
        }
        handle
    };

    Ok(build_chat_report(&handle, assistant_texts))
}

fn print_chat(report: ChatReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        for assistant_text in report.assistant_texts {
            if assistant_text.ends_with('\n') {
                print!("{assistant_text}");
            } else {
                println!("{assistant_text}");
            }
        }
    }
    Ok(())
}

fn execute_exec(args: ExecArgs) -> Result<ExecReport> {
    execute_shell(&args.command, &args.cwd, &args.shell, &args.env_vars)
        .with_context(|| format!("executing command in {}", args.cwd.display()))
}

fn print_exec(report: ExecReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        print!("{}", report.stdout);
        eprint!("{}", report.stderr);
    }
    Ok(())
}

fn execute_edit(args: EditArgs) -> Result<EditReport> {
    perform_edit(
        &args.file_path,
        &args.old_string,
        &args.new_string,
        args.replace_all,
    )
}

fn print_edit(report: EditReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("Edited {}", report.file_path);
    }
    Ok(())
}

fn resolve_against_cwd(cwd: &Path, raw_path: &str) -> PathBuf {
    let file_path = PathBuf::from(raw_path);
    if file_path.is_absolute() {
        file_path
    } else {
        cwd.join(file_path)
    }
}

fn read_chat_prompts() -> Result<Vec<String>, CliError> {
    let mut input = String::new();
    stdin()
        .read_to_string(&mut input)
        .map_err(|err| CliError::internal(err.into()))?;
    Ok(input
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .filter(|line| !line.trim().is_empty())
        .collect())
}

fn run_agent_command(args: &[String]) -> Result<(), CliError> {
    match args.first().map(String::as_str) {
        Some("types") => {
            let json = args[1..].iter().any(|arg| arg == "--json");
            print_agent_types(built_in_agents(), json)
        }
        Some("spawn") => {
            let cmd = parse_agent_spawn_args(&args[1..])?;
            let json = cmd.json;
            let record =
                spawn_agent(&cmd.agent_type, &cmd.cwd, &cmd.prompt).map_err(CliError::internal)?;
            print_agent_record(record, json)
        }
        Some("wait") => {
            let cmd = parse_agent_wait_args(&args[1..])?;
            let record = wait_agent(&cmd.agent_id).map_err(CliError::internal)?;
            print_optional_agent_record(record, cmd.json)
        }
        Some("send") => {
            let cmd = parse_agent_send_args(&args[1..])?;
            let record = send_agent(&cmd.agent_id, &cmd.prompt).map_err(CliError::internal)?;
            print_optional_agent_record(record, cmd.json)
        }
        Some("close") => {
            let cmd = parse_agent_close_args(&args[1..])?;
            let record = close_agent(&cmd.agent_id).map_err(CliError::internal)?;
            print_optional_agent_record(record, cmd.json)
        }
        Some("ps") => {
            let json = args[1..].iter().any(|arg| arg == "--json");
            let records = list_agent_records().map_err(CliError::internal)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&records)
                        .map_err(|err| CliError::internal(err.into()))?
                );
            } else {
                for record in records {
                    println!(
                        "{}\t{}\t{}",
                        record.agent_id, record.agent_type, record.status
                    );
                }
            }
            Ok(())
        }
        Some(other) => Err(CliError::usage(format!("unknown agent command: {other}"))),
        None => Err(CliError::usage("agent requires a subcommand")),
    }
}

fn print_agent_types(types: Vec<AgentTypeInfo>, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&types).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        for agent in types {
            println!("{}\t{}", agent.agent_type, agent.when_to_use);
        }
    }
    Ok(())
}

fn print_agent_record(record: AgentRecord, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("{}", record.agent_id);
    }
    Ok(())
}

fn print_optional_agent_record(record: Option<AgentRecord>, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|err| CliError::internal(err.into()))?
        );
    } else if let Some(record) = record {
        println!("{}", record.response);
    }
    Ok(())
}
