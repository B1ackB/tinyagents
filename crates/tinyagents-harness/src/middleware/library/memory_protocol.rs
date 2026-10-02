//! Memory-protocol enforcement state machine.
//!
//! Agents are instructed to follow a **read-index → dedupe → write →
//! update-index** cycle when they mutate durable memory:
//!
//! 1. Read the memory index (a recall / tree query, or equivalently the index
//!    file) to check for near-duplicates *before* creating an entry.
//! 2. Write the entry (store, forget, or a document ingest).
//! 3. Update the index file afterward so it stays in sync with the underlying
//!    store.
//!
//! The protocol was previously described to the model but never enforced, so it
//! was followed inconsistently — agents wrote entries without a dedupe read
//! (creating duplicates) and skipped the index update (so the index drifted).
//!
//! This module is the pure, side-effect-free state machine that observes the
//! sequence of memory tool calls in a session and reports two violations:
//!
//! - **missing index read** — a write not preceded by an index read this cycle.
//! - **index drift** — a write that was never followed by an index update
//!   (detected at the next write, and at run end via
//!   [`MemoryProtocolTracker::pending_index_update`]).
//!
//! Which tool names play which role is host vocabulary, passed in as a
//! [`MemoryProtocolSpec`]. [`super::MemoryProtocolMiddleware`] drives the
//! tracker from its `after_tool` / `after_agent` hooks and surfaces the guidance
//! back to the model as a corrective note appended to the tool result.

use std::sync::Arc;

/// Marker prefixed to every corrective note so downstream code (and tests) can
/// recognise memory-protocol guidance in a tool result.
pub const MEMORY_PROTOCOL_MARKER: &str = "[memory-protocol]";

/// A polymorphic tool that reads in every mode except one, which writes.
#[derive(Debug, Clone)]
pub struct ModeTool {
    /// Tool name (e.g. a consolidated memory-tree tool).
    pub name: String,
    /// Argument naming the mode.
    pub mode_arg: String,
    /// The one mode that is a durable write; every other mode is an index read.
    pub write_mode: String,
}

/// The host's tool vocabulary for the memory protocol.
///
/// Tools not named here are [`MemoryOp::Other`] and never affect protocol state.
#[derive(Debug, Clone)]
pub struct MemoryProtocolSpec {
    /// Tool that writes the index back into sync (the closing step).
    pub index_update_tool: String,
    /// Argument on that tool naming which file it edits.
    pub index_file_arg: String,
    /// The file value that counts as the memory index. The same tool can edit
    /// other files, which does not reconcile the index and so is a no-op.
    pub index_file: String,
    /// Tools that are durable mutations (create / delete / ingest).
    pub write_tools: Vec<String>,
    /// Dedupe reads: recall / search over stored memory.
    pub read_tools: Vec<String>,
    /// An optional read-by-default tool whose one write mode is a mutation.
    pub mode_tool: Option<ModeTool>,
    /// Recall tool named in the missing-index-read guidance.
    pub recall_tool: String,
}

impl MemoryProtocolSpec {
    /// Classify a tool call into a [`MemoryOp`], keyed by name and — for the two
    /// polymorphic tools — the arguments. Arguments are captured at
    /// `before_tool` time and correlated to the result by call id (the tool
    /// result itself carries no arguments).
    pub fn classify(&self, tool_name: &str, arguments: &serde_json::Value) -> MemoryOp {
        let arg_str = |key: &str| arguments.get(key).and_then(|v| v.as_str());
        if tool_name == self.index_update_tool {
            // The index-sync step, but only for the index file. The same tool can
            // edit other files, which does not reconcile the memory index.
            return match arg_str(&self.index_file_arg) {
                Some(file) if file == self.index_file => MemoryOp::IndexUpdate,
                _ => MemoryOp::Other,
            };
        }
        if self.write_tools.iter().any(|t| t == tool_name) {
            return MemoryOp::Write;
        }
        if let Some(mode_tool) = &self.mode_tool
            && tool_name == mode_tool.name
        {
            return match arg_str(&mode_tool.mode_arg) {
                Some(mode) if mode == mode_tool.write_mode => MemoryOp::Write,
                _ => MemoryOp::IndexRead,
            };
        }
        if self.read_tools.iter().any(|t| t == tool_name) {
            return MemoryOp::IndexRead;
        }
        MemoryOp::Other
    }
}

/// Classification of a tool call for the memory protocol. Everything the model
/// can call is one of these; non-memory tools are [`MemoryOp::Other`] and never
/// affect protocol state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOp {
    /// Reading the memory index to check for duplicates (satisfies the "read
    /// index / dedupe" step of the cycle).
    IndexRead,
    /// A durable memory mutation that should be preceded by a dedupe read and
    /// followed by an index update.
    Write,
    /// Writing the index back into sync (the closing step).
    IndexUpdate,
    /// Any tool that is not part of the memory protocol.
    Other,
}

/// What a single observed memory op means for the protocol. Returned by
/// [`MemoryProtocolTracker::observe`] so the caller can decide whether — and
/// what — to surface back to the model.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryProtocolObservation {
    /// The observed op was a durable memory write.
    pub was_write: bool,
    /// A write happened without a dedupe/index read earlier in this cycle.
    pub missing_index_read: bool,
    /// A write happened while a previous write was still awaiting
    /// `update_memory_md` — the index is drifting from the store.
    pub index_drift: bool,
}

impl MemoryProtocolObservation {
    /// Whether this observation warrants a corrective note back to the model.
    /// Every write gets one (the forward "call `update_memory_md`" reminder is
    /// itself the enforcement of the closing step); reads and other ops don't.
    pub fn needs_guidance(&self) -> bool {
        self.was_write
    }

    /// Render the corrective note appended to the tool result, or `None` when no
    /// guidance is warranted. The wording escalates with the violations detected.
    pub fn guidance(&self, spec: &MemoryProtocolSpec, tool_name: &str) -> Option<String> {
        self.guidance_with_index_update(spec, tool_name, true)
    }

    /// Keep dedupe guidance when the agent cannot update the index, without
    /// telling it to call a tool absent from its turn.
    pub fn guidance_with_index_update(
        &self,
        spec: &MemoryProtocolSpec,
        tool_name: &str,
        can_update_index: bool,
    ) -> Option<String> {
        if !self.needs_guidance() {
            return None;
        }
        let mut parts: Vec<String> = Vec::new();
        if self.missing_index_read {
            parts.push(format!(
                "`{tool_name}` wrote to memory without first reading the memory index to check for \
                 duplicates. Before creating entries, recall existing memory (e.g. `{}`) \
                 so you don't store a near-duplicate.",
                spec.recall_tool
            ));
        }
        if self.index_drift && can_update_index {
            parts.push(format!(
                "A previous memory write was never followed by `{tool}`, so the {file} \
                 index is drifting from stored memory. Reconcile it now.",
                tool = spec.index_update_tool,
                file = spec.index_file
            ));
        }
        if can_update_index {
            parts.push(format!(
                "After mutating memory, call `{tool}` to keep the {file} index in sync.",
                tool = spec.index_update_tool,
                file = spec.index_file
            ));
        }
        (!parts.is_empty()).then(|| format!("{MEMORY_PROTOCOL_MARKER} {}", parts.join(" ")))
    }
}

/// Per-session tracker of the memory-protocol cycle. One instance lives for the
/// duration of a turn/run (held by the middleware) and observes the ordered
/// sequence of *successful* memory tool calls.
///
/// The cycle is `read → write → update-index`, and it repeats: an
/// an index update closes a cycle and arms the next one (so the following
/// write again expects a fresh dedupe read).
#[derive(Debug)]
pub struct MemoryProtocolTracker {
    spec: Arc<MemoryProtocolSpec>,
    /// A dedupe/index read has occurred since the last cycle reset.
    saw_index_read: bool,
    /// A write has occurred that has not yet been followed by an index update.
    pending_index_update: bool,
}

impl MemoryProtocolTracker {
    /// Fresh tracker with no observed ops.
    pub fn new(spec: Arc<MemoryProtocolSpec>) -> Self {
        Self {
            spec,
            saw_index_read: false,
            pending_index_update: false,
        }
    }

    /// Observe one **successful** memory op and advance the state machine,
    /// returning what it means for the protocol. Callers must only pass ops that
    /// actually succeeded — a failed `memory_store` neither creates an entry nor
    /// obliges an index update.
    pub fn observe(&mut self, op: MemoryOp) -> MemoryProtocolObservation {
        match op {
            MemoryOp::IndexRead => {
                self.saw_index_read = true;
                MemoryProtocolObservation::default()
            }
            MemoryOp::Write => {
                let obs = MemoryProtocolObservation {
                    was_write: true,
                    missing_index_read: !self.saw_index_read,
                    index_drift: self.pending_index_update,
                };
                self.pending_index_update = true;
                obs
            }
            MemoryOp::IndexUpdate => {
                // The index is back in sync; arm the next cycle so its write
                // expects a fresh dedupe read.
                self.pending_index_update = false;
                self.saw_index_read = false;
                MemoryProtocolObservation::default()
            }
            MemoryOp::Other => MemoryProtocolObservation::default(),
        }
    }

    /// Classify a tool call (name + arguments) and [`observe`](Self::observe) it
    /// in one step.
    pub fn observe_tool(
        &mut self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> MemoryProtocolObservation {
        self.observe(self.spec.classify(tool_name, arguments))
    }

    /// Whether a memory write is still awaiting an index update. Checked at
    /// run end to detect a write that was never followed by an index update.
    pub fn pending_index_update(&self) -> bool {
        self.pending_index_update
    }
}

#[cfg(test)]
#[path = "memory_protocol_tests.rs"]
mod tests;
