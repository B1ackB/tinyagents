//! Rendering a task state as the checkpoint text, and reading the carried
//! facts back out of a previous checkpoint.
//!
//! The deterministic facts are written inside XML-ish tags so the next
//! compaction can carry them forward exactly, without asking a model to
//! remember them: `<original-task>`, `<modified-files>`, `<read-files>` and
//! `<recent-commands-json>`.
//! The model-written fields are read back from their `## ` sections, so the
//! state is written once (a JSON copy beside the sections doubled every
//! checkpoint).

use super::types::{CommandRecord, TaskLedger, TaskState};

/// First line of every task-state checkpoint body.
pub const TASK_STATE_HEADER: &str = "# Task state (compacted)";

/// Renders the checkpoint body: readable sections for the agent, plus tagged
/// blocks the next compaction parses back.
#[must_use]
pub fn render_task_state(state: &TaskState, ledger: &TaskLedger) -> String {
    // One line per item and per scalar: a newline inside a value would read
    // back as a new item or section.
    let list = |items: &[String]| -> String {
        if items.is_empty() {
            NONE_ITEM.to_string()
        } else {
            items
                .iter()
                .map(|i| format!("- {}", one_line(i)))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let or_none = |s: &str| {
        if s.trim().is_empty() {
            NONE.to_string()
        } else {
            one_line(s)
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
    // Escape delimiters so a task cannot close its block and forge ledger tags.
    let task = escape_tagged(ledger.original_task.as_deref().unwrap_or(""));
    let command_data = escape_tagged(&serde_json::to_string(&ledger.commands).unwrap_or_default());
    // The visible sections render missing and explicitly cleared values the
    // same way. Carry presence separately so split-summary merge can tell
    // them apart; older checkpoints without this tag treat all as present.
    let live_presence = format!(
        "{}{}{}",
        u8::from(state.current_hypothesis.is_some()),
        u8::from(state.test_command.is_some()),
        u8::from(state.next_step.is_some())
    );
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
         <recent-commands-json>\n{command_data}\n</recent-commands-json>\n\
         <live-task-presence>{live_presence}</live-task-presence>",
        goal = or_none(&state.goal),
        requirements = list(&state.requirements),
        constraints = list(&state.constraints),
        decisions = list(&state.decisions),
        errors = list(&state.errors_and_fixes),
        done = list(&state.todos_done),
        open = list(&state.todos_open),
        hypothesis = or_none(state.current_hypothesis.as_deref().unwrap_or("")),
        test = or_none(state.test_command.as_deref().unwrap_or("")),
        next = or_none(state.next_step.as_deref().unwrap_or("")),
        modified = ledger
            .files_modified
            .iter()
            .map(|p| escape_tagged(p))
            .collect::<Vec<_>>()
            .join("\n"),
        read = ledger
            .read_only_files()
            .iter()
            .map(|p| escape_tagged(p))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn escape_tagged(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unescape_tagged(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

const NONE: &str = "none";
const NONE_ITEM: &str = "- none";

/// Section headings of the model-written fields, in render order.
const GOAL: &str = "## Goal";
const REQUIREMENTS: &str = "## Requirements (verbatim)";
const CONSTRAINTS: &str = "## Constraints";
const DECISIONS: &str = "## Decisions";
const ERRORS: &str = "## Errors and fixes";
const DONE: &str = "## Done";
const OPEN: &str = "## Open";
const HYPOTHESIS: &str = "## Current hypothesis";
const TEST: &str = "## Test command";
const NEXT: &str = "## Next step";
const COMMANDS: &str = "## Recent commands";

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The body of the `heading` section: the lines up to the next `## `
/// heading or tagged block.
fn section<'a>(text: &'a str, heading: &str) -> Option<&'a str> {
    // Only search the generated sections, never Markdown in the original task.
    let sections = text
        .split_once("</original-task>")
        .map_or(text, |(_, sections)| sections);
    let at = sections.find(&format!("\n{heading}\n"))? + heading.len() + 2;
    let rest = &sections[at..];
    let end = rest
        .find("\n## ")
        .into_iter()
        .chain(rest.find("\n<"))
        .min()
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn section_items(text: &str, heading: &str) -> Vec<String> {
    section(text, heading)
        .map(|body| {
            body.lines()
                .filter_map(|l| l.trim().strip_prefix("- "))
                .map(str::trim)
                .filter(|l| !l.is_empty() && *l != NONE)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn section_text(text: &str, heading: &str) -> String {
    section(text, heading)
        .filter(|body| *body != NONE)
        .map(String::from)
        .unwrap_or_default()
}

fn render_command(c: &CommandRecord) -> String {
    let command = escape_tagged(&c.command);
    match (&c.failed, &c.error) {
        (false, _) => format!("- `{command}` → ok"),
        (true, Some(error)) => format!("- `{command}` → FAILED: {}", escape_tagged(error)),
        (true, None) => format!("- `{command}` → FAILED"),
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

/// Read the generated ledger block after model-written state sections, which
/// may themselves contain literal examples of ledger tags.
fn last_tagged<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.rfind(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    Some(text[start..end].trim())
}

/// Reads the commands back out of the `## Recent commands` section.
fn section_commands(text: &str) -> Vec<CommandRecord> {
    section(text, COMMANDS)
        .map(|body| {
            body.lines()
                .filter_map(|l| l.trim().strip_prefix("- `"))
                .filter_map(|l| {
                    let (command, outcome) = l.rsplit_once("` → ")?;
                    let failed = outcome.starts_with("FAILED");
                    let error = outcome
                        .strip_prefix("FAILED: ")
                        .map(unescape_tagged)
                        .filter(|e| !e.is_empty());
                    Some(CommandRecord {
                        command: unescape_tagged(command),
                        failed,
                        error,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Reads the facts a previous checkpoint carries: its ledger (task and file
/// lists and recent commands) and its
/// model-written state. A previous summary that is not a task-state
/// checkpoint (a free-form summary from another summarizer) yields an empty
/// ledger and `None`; the caller then hands its text to the model instead.
#[must_use]
pub fn parse_carried(previous: &str) -> (TaskLedger, Option<TaskState>) {
    let body = previous
        .split_once("</original-task>")
        .map_or(previous, |(_, body)| body);
    let lines = |tag: &str| -> Vec<String> {
        last_tagged(body, tag)
            .map(|block| {
                block
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(unescape_tagged)
                    .collect()
            })
            .unwrap_or_default()
    };
    let ledger = TaskLedger {
        original_task: tagged(previous, "original-task")
            .filter(|t| !t.is_empty())
            .map(unescape_tagged),
        files_modified: lines("modified-files"),
        files_read: lines("read-files"),
        commands: last_tagged(body, "recent-commands-json")
            .and_then(|block| serde_json::from_str(&unescape_tagged(block)).ok())
            .unwrap_or_else(|| section_commands(previous)),
    };
    // This generated tag is the final line. Read only that trailer: model
    // fields can contain a literal copy of the tag earlier in the sections.
    let live_presence = previous
        .trim_end()
        .rsplit_once("\n<live-task-presence>")
        .and_then(|(_, tail)| tail.strip_suffix("</live-task-presence>"))
        .filter(|value| value.len() == 3)
        .map(str::as_bytes);
    let present = |index: usize| live_presence.is_none_or(|bits| bits[index] != b'0');
    let state = previous
        .trim_start()
        .starts_with(TASK_STATE_HEADER)
        .then(|| TaskState {
            goal: section_text(previous, GOAL),
            requirements: section_items(previous, REQUIREMENTS),
            constraints: section_items(previous, CONSTRAINTS),
            decisions: section_items(previous, DECISIONS),
            errors_and_fixes: section_items(previous, ERRORS),
            todos_done: section_items(previous, DONE),
            todos_open: section_items(previous, OPEN),
            current_hypothesis: present(0).then(|| section_text(previous, HYPOTHESIS)),
            test_command: present(1).then(|| section_text(previous, TEST)),
            next_step: present(2).then(|| section_text(previous, NEXT)),
        });
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
