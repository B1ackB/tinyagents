//! Rendering a task state as the checkpoint text, and reading the carried
//! facts back out of a previous checkpoint.
//!
//! The deterministic facts are written inside XML-ish tags so the next
//! compaction can carry them forward exactly, without asking a model to
//! remember them: `<original-task>`, `<modified-files>`, `<read-files>` and
//! `<task-state>` (the model fields as JSON, so they round-trip too).

use super::types::{CommandRecord, TaskLedger, TaskState};

/// First line of every task-state checkpoint body.
pub const TASK_STATE_HEADER: &str = "# Task state (compacted)";

/// Renders the checkpoint body: readable sections for the agent, plus tagged
/// blocks the next compaction parses back.
#[must_use]
pub fn render_task_state(state: &TaskState, ledger: &TaskLedger) -> String {
    let list = |items: &[String]| -> String {
        if items.is_empty() {
            "- none".to_string()
        } else {
            items
                .iter()
                .map(|i| format!("- {i}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let or_none = |s: &str| {
        if s.trim().is_empty() {
            "none".to_string()
        } else {
            s.to_string()
        }
    };
    let commands = if ledger.commands.is_empty() {
        "- none".to_string()
    } else {
        ledger
            .commands
            .iter()
            .map(render_command)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let state_json = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
    let task = ledger.original_task.as_deref().unwrap_or("");
    format!(
        "{TASK_STATE_HEADER}\n\n\
         <original-task>\n{task}\n</original-task>\n\n\
         ## Goal\n{goal}\n\n\
         ## Requirements (verbatim)\n{requirements}\n\n\
         ## Constraints\n{constraints}\n\n\
         ## Decisions\n{decisions}\n\n\
         ## Errors and fixes\n{errors}\n\n\
         ## Done\n{done}\n\n\
         ## Open\n{open}\n\n\
         ## Current hypothesis\n{hypothesis}\n\n\
         ## Test command\n{test}\n\n\
         ## Next step\n{next}\n\n\
         ## Recent commands\n{commands}\n\n\
         <modified-files>\n{modified}\n</modified-files>\n\
         <read-files>\n{read}\n</read-files>\n\
         <task-state>\n{state_json}\n</task-state>",
        goal = or_none(&state.goal),
        requirements = list(&state.requirements),
        constraints = list(&state.constraints),
        decisions = list(&state.decisions),
        errors = list(&state.errors_and_fixes),
        done = list(&state.todos_done),
        open = list(&state.todos_open),
        hypothesis = or_none(&state.current_hypothesis),
        test = or_none(&state.test_command),
        next = or_none(&state.next_step),
        modified = ledger.files_modified.join("\n"),
        read = ledger.read_only_files().join("\n"),
    )
}

fn render_command(c: &CommandRecord) -> String {
    match (&c.failed, &c.error) {
        (false, _) => format!("- `{}` → ok", c.command),
        (true, Some(error)) => format!("- `{}` → FAILED: {error}", c.command),
        (true, None) => format!("- `{}` → FAILED", c.command),
    }
}

/// The content of the first `<tag>…</tag>` block in `text`, trimmed.
fn tagged<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    Some(text[start..end].trim())
}

/// Reads the facts a previous checkpoint carries: its ledger (task and file
/// lists; commands are not carried, the recent ones are re-read) and its
/// model-written state. A previous summary that is not a task-state
/// checkpoint (a free-form summary from another summarizer) yields an empty
/// ledger and `None`; the caller then hands its text to the model instead.
#[must_use]
pub fn parse_carried(previous: &str) -> (TaskLedger, Option<TaskState>) {
    let lines = |tag: &str| -> Vec<String> {
        tagged(previous, tag)
            .map(|block| {
                block
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    };
    let ledger = TaskLedger {
        original_task: tagged(previous, "original-task")
            .filter(|t| !t.is_empty())
            .map(String::from),
        files_modified: lines("modified-files"),
        files_read: lines("read-files"),
        commands: Vec::new(),
    };
    let state = tagged(previous, "task-state").and_then(|json| serde_json::from_str(json).ok());
    (ledger, state)
}

/// The JSON object in a model reply, tolerating prose or code fences around
/// it. `None` when no object parses.
#[must_use]
pub fn parse_state_reply(reply: &str) -> Option<TaskState> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&reply[start..=end]).ok()
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
