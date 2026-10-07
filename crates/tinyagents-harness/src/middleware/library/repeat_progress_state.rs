//! Shared state and helpers behind [`super::RepeatProgressMiddleware`] and its
//! eviction observer.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::ModelRequest;
use tinytools::{ToolContent, ToolResult as TaToolResult};

use crate::no_progress::{RepeatMonitor, RepeatProgressConfig};

/// Warnings kept waiting for a later result; beyond this the newest are dropped.
const MAX_DEFERRED_NOTES: usize = 4;

/// Extract the assistant's visible text (concatenated [`ContentBlock::Text`]
/// blocks) from a model response message, for the repeat-output signature.
pub(super) fn assistant_visible_text(message: &tinyinference_llm::message::AssistantMessage) -> String {
    let mut out = String::new();
    for block in &message.content {
        if let ContentBlock::Text(t) = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
    }
    out
}

/// Per-batch state the repeat-CALL guard needs but can only fully evaluate once
/// every tool result in the assistant's batch has come back: the canonical
/// `(tool, args)` signature captured at `after_model`, plus the running
/// success/remaining accounting folded in at each `after_tool`.
#[derive(Default)]
pub(super) struct PendingCallBatch {
    /// Canonical `(tool, args)` signature of the batch, from `after_model`.
    pub(super) call_sig: String,
    /// Tool results still outstanding for this batch.
    pub(super) remaining: usize,
    /// `true` while every result so far in the batch has succeeded.
    pub(super) all_ok: bool,
    /// `true` when every call in the batch is a polling/wait exemption.
    pub(super) exempt: bool,
    /// `call_id` → per-call `(tool, argument fingerprint)` for the recurrence
    /// ledger. Polling/wait calls are left out.
    pub(super) call_sigs: HashMap<String, VecDeque<(String, String)>>,
    /// `true` once a result in this batch has already halted the run, so the
    /// batch does not pause it a second time.
    pub(super) halted: bool,
}

/// Tracker state shared between [`RepeatProgressMiddleware`] and its
/// [`RepeatEvictionObserver`].
pub(super) struct RepeatState {
    pub(super) monitors: Mutex<HashMap<u64, RepeatMonitor>>,
    pub(super) config: RepeatProgressConfig,
    /// The body a cleared tool result carries.
    pub(super) cleared_placeholder: String,
    /// `call_id`s of the results fed to the recurrence ledger since its last reset.
    pub(super) recorded: Mutex<HashMap<u64, HashSet<String>>>,
    /// Recorded results still verbatim in the current request before any
    /// reduction step ran; compared against the final request by the observer.
    pub(super) visible_before_reduction: Mutex<HashMap<u64, HashSet<String>>>,
    /// `call_id` → how the guard answered the call without running it.
    pub(super) refused: Mutex<HashMap<u64, HashMap<String, &'static str>>>,
    /// Warnings that did not fit on the result they arose with (one note per
    /// result), waiting for the next successful one.
    pub(super) deferred: Mutex<HashMap<u64, VecDeque<String>>>,
}

impl RepeatState {
    pub(super) fn new(placeholder: impl Into<String>, config: RepeatProgressConfig) -> Self {
        Self {
            monitors: Mutex::default(),
            config,
            cleared_placeholder: placeholder.into(),
            recorded: Mutex::default(),
            visible_before_reduction: Mutex::default(),
            refused: Mutex::default(),
            deferred: Mutex::default(),
        }
    }
}

impl RepeatState {
    /// Runs `f` on the run's monitor, creating it on first use. `None` only if
    /// the lock is poisoned, in which case the guard stays out of the way.
    pub(super) fn with_monitor<R>(&self, run_id: u64, f: impl FnOnce(&RepeatMonitor) -> R) -> Option<R> {
        let mut monitors = self.monitors.lock().ok()?;
        let monitor = monitors
            .entry(run_id)
            .or_insert_with(|| RepeatMonitor::new(&self.config));
        Some(f(monitor))
    }
}

impl RepeatState {
    /// Remembers that the call `call_id` was answered without running.
    pub(super) fn mark_refused(&self, run_id: u64, call_id: &str, marker: &'static str) {
        if let Ok(mut refused) = self.refused.lock() {
            refused
                .entry(run_id)
                .or_default()
                .insert(call_id.to_string(), marker);
        }
    }

    /// Takes (and forgets) how `call_id` was answered, if the guard refused it.
    pub(super) fn take_refused(&self, run_id: u64, call_id: &str) -> Option<&'static str> {
        self.refused.lock().ok()?.get_mut(&run_id)?.remove(call_id)
    }

    /// Picks the single warning to put on this result from `fresh` (most
    /// specific first) and the ones queued earlier, and queues the rest for
    /// the next result. Identical notes collapse.
    pub(super) fn take_one_note(&self, run_id: u64, fresh: Vec<String>) -> Option<String> {
        let Ok(mut deferred) = self.deferred.lock() else {
            return fresh.into_iter().next();
        };
        let queue = deferred.entry(run_id).or_default();
        for note in fresh {
            if !queue.contains(&note) {
                queue.push_back(note);
            }
        }
        queue.truncate(MAX_DEFERRED_NOTES);
        queue.pop_front()
    }

    /// Drops everything held for a finished run.
    pub(super) fn forget_run(&self, run_id: u64) {
        if let Ok(mut monitors) = self.monitors.lock() {
            monitors.remove(&run_id);
        }
        if let Ok(mut recorded) = self.recorded.lock() {
            recorded.remove(&run_id);
        }
        if let Ok(mut visible) = self.visible_before_reduction.lock() {
            visible.remove(&run_id);
        }
        if let Ok(mut refused) = self.refused.lock() {
            refused.remove(&run_id);
        }
        if let Ok(mut deferred) = self.deferred.lock() {
            deferred.remove(&run_id);
        }
    }
}

/// Metadata key the guard sets on a result it produced without running the
/// tool. The value is [`REPEAT_GUARD_BLOCKED`] or [`REPEAT_GUARD_HALTED`].
/// Metadata is host-only (never shown to the model), so a host can use this to
/// keep those results out of a failure ladder: they are the guard's answer, not
/// the tool failing.
pub const REPEAT_GUARD_METADATA_KEY: &str = "tinyagents.repeat_guard";
/// [`REPEAT_GUARD_METADATA_KEY`] value: the call was blocked and not executed.
pub const REPEAT_GUARD_BLOCKED: &str = "blocked";
/// [`REPEAT_GUARD_METADATA_KEY`] value: the call was refused and the run halted.
pub const REPEAT_GUARD_HALTED: &str = "halted";

/// The repeat-guard marker on `result`, if the guard answered it without
/// running the tool: [`REPEAT_GUARD_BLOCKED`] or [`REPEAT_GUARD_HALTED`].
pub fn repeat_guard_marker(result: &TaToolResult) -> Option<&str> {
    result
        .metadata
        .as_ref()?
        .get(REPEAT_GUARD_METADATA_KEY)?
        .as_str()
}

/// Stamps `marker` into the result's metadata, keeping any existing object.
pub(super) fn tag_result(result: &mut TaToolResult, marker: &str) {
    match result.metadata.as_mut().and_then(|m| m.as_object_mut()) {
        Some(object) => {
            object.insert(REPEAT_GUARD_METADATA_KEY.to_string(), marker.into());
        }
        None => {
            result.metadata = Some(serde_json::json!({ REPEAT_GUARD_METADATA_KEY: marker }));
        }
    }
}

/// Appends a warning to the result the model is about to read, in the plain
/// blocks and in the markdown rendering.
pub(super) fn append_notes(result: &mut TaToolResult, note: &str) {
    let mut chars = note.chars();
    let Some(first) = chars.next() else {
        return;
    };
    let note = format!("[repeat notice] {}{}", first.to_uppercase(), chars.as_str());
    result.content.push(ToolContent::Text {
        text: format!("\n\n{note}"),
    });
    if let Some(markdown) = result.markdown_formatted.as_mut() {
        markdown.push_str("\n\n");
        markdown.push_str(&note);
    }
}

/// The `ids` whose tool result is still in `request` with its body intact.
pub(super) fn visible_tool_results(
    request: &ModelRequest,
    ids: &HashSet<String>,
    placeholder: &str,
) -> HashSet<String> {
    request
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool)
                if ids.contains(&tool.tool_call_id) && message.text() != placeholder =>
            {
                Some(tool.tool_call_id.clone())
            }
            _ => None,
        })
        .collect()
}

