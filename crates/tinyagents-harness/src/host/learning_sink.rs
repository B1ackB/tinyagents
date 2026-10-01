//! Post-turn learning capability — the seam a host uses to hang reflection,
//! distillation, or self-improvement pipelines off a completed turn.
//!
//! A host supplies a [`LearningSink`] when it wants to *observe* finished turns
//! and derive something durable from them (a lesson, a summary, a training
//! signal). The runtime itself learns nothing: it hands over an inert
//! [`TurnSummary`] and forgets about it. Everything interesting — what counts
//! as a lesson, where it is stored, whether it is redacted first — is host
//! policy and stays host-side.
//!
//! # When to supply one, and when to pass `None`
//!
//! This capability is **optional**. A host with no learning pipeline passes
//! `None` and the runtime skips the call entirely. That is deliberately not the
//! same as installing a sink that always errors: absence means "there is
//! nothing to notify", failure means "notification was attempted and did not
//! work", and only the host can tell those apart. Do not install
//! [`NoopLearningSink`] to *express* absence — it exists for tests, for
//! composition defaults, and for hosts that want the call path exercised
//! without side effects.
//!
//! # The call happens after the turn is committed
//!
//! [`LearningSink::on_turn_complete`] runs once the turn's result has already
//! been produced and committed. It is an epilogue, not a gate. An `Err` from it
//! therefore **must not** roll the turn back, retract its output, or fail the
//! caller — the runtime logs the error and continues. Two consequences worth
//! internalising before implementing one:
//!
//! - Returning `Err` is a *diagnostic*, not a veto. If a sink needs to prevent
//!   an action, it is the wrong trait; that is a gate's job, and gates are
//!   consulted before the fact.
//! - The runtime may invoke the sink on a path where nobody is waiting for the
//!   result, so a slow implementation costs latency without buying safety.
//!   Expensive reflection belongs behind the host's own queue; enqueue here and
//!   return.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::ThreadId;
use tinyinference_llm::usage::Usage;

// ── TurnSummary ───────────────────────────────────────────────────────────────

/// An inert record of one completed turn, handed to a [`LearningSink`].
///
/// **Dependency rule:** `serde` + `std` only. A host must be able to construct,
/// serialize, and queue this without pulling in the runtime — it is routinely
/// written to a durable queue and processed out-of-process, long after the turn
/// that produced it has ended.
///
/// It is intentionally a *flattened* view rather than a handle into live turn
/// state: the turn is already over by the time a sink sees this, so anything
/// the sink might want must be copied in here up front.
// No `Default`: `ThreadId` has none, and a summary attributed to a blank
// thread/agent is worse than no summary — see `new`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSummary {
    /// The thread the turn belongs to. Carried so a sink can attribute a lesson
    /// to a conversation without re-deriving it from the text.
    pub thread_id: ThreadId,

    /// The agent that ran the turn.
    ///
    /// A plain `String`, not a newtype: the crate has no `AgentId` type, and
    /// inventing one here would make this module the accidental owner of an
    /// identifier every other module refers to by name. Hosts key their agents
    /// by string ids already, so nothing is lost.
    pub agent_id: String,

    /// The turn's inbound text as the agent received it.
    ///
    /// Plain text, not [`tinyinference_llm::message::Message`]: a sink wants the
    /// human-readable turn, and pinning the structured message shape here would
    /// make every transcript-format change a breaking change for stored
    /// summaries.
    pub input: String,

    /// The turn's outbound text as the agent produced it.
    pub output: String,

    /// Names of the tools invoked during the turn, in first-invocation order.
    ///
    /// Names only — never arguments or results. Tool payloads are the most
    /// likely place for credentials and user data to appear, and this record is
    /// designed to be persisted, so it carries the *shape* of what happened,
    /// not its contents. Maintained normalized by [`TurnSummary::record_tool`].
    #[serde(default)]
    pub tools_invoked: Vec<String>,

    /// Token usage accumulated by the turn, for sinks that weight or budget
    /// what they learn from.
    #[serde(default)]
    pub usage: Usage,
}

impl TurnSummary {
    /// A summary for `thread_id` / `agent_id` with empty text, no tools, and
    /// zeroed usage. Fill the rest in with the `with_*` setters.
    pub fn new(thread_id: impl Into<ThreadId>, agent_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            agent_id: agent_id.into(),
            input: String::new(),
            output: String::new(),
            tools_invoked: Vec::new(),
            usage: Usage::default(),
        }
    }

    /// Sets the inbound and outbound turn text.
    pub fn with_text(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.input = input.into();
        self.output = output.into();
        self
    }

    /// Sets the accumulated token usage.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    /// Records that the tool named `name` was invoked.
    ///
    /// Normalizes as it goes: the name is trimmed, blank names are dropped, and
    /// a repeat invocation of an already-recorded tool is ignored so the vector
    /// stays a first-invocation-ordered *set*. A tool called in a five-step
    /// loop should not out-weigh four distinct tools when a sink counts them,
    /// and callers building this incrementally cannot be relied on to
    /// deduplicate themselves.
    pub fn record_tool(&mut self, name: impl AsRef<str>) {
        let trimmed = name.as_ref().trim();
        if trimmed.is_empty() || self.tools_invoked.iter().any(|t| t == trimmed) {
            return;
        }
        self.tools_invoked.push(trimmed.to_string());
    }

    /// Chaining form of [`record_tool`](Self::record_tool).
    pub fn with_tool(mut self, name: impl AsRef<str>) -> Self {
        self.record_tool(name);
        self
    }

    /// Whether the turn invoked any tool. Cheaper than inspecting the vector at
    /// call sites that only branch on "did this turn act, or merely talk".
    pub fn used_tools(&self) -> bool {
        !self.tools_invoked.is_empty()
    }
}

// ── LearningSink ──────────────────────────────────────────────────────────────

/// Host capability notified once per completed turn.
///
/// Optional: a host without a learning pipeline supplies `None` rather than an
/// erroring implementation, so the runtime can distinguish "not configured"
/// from "configured and broken".
#[async_trait]
pub trait LearningSink: Send + Sync {
    /// Called after the turn has been committed.
    ///
    /// `summary` is borrowed, not owned, because the runtime may fan the same
    /// summary out to more than one observer; an implementation that needs to
    /// keep it must clone.
    ///
    /// **Errors are advisory.** The turn is already final by the time this
    /// runs, so the runtime logs a returned `Err` and continues — it will not
    /// retry, retract the turn, or surface the failure to the caller. Return
    /// `Err` to make a problem visible in host logs, never to reject the turn.
    /// Implementations should also avoid blocking: hand slow work to the host's
    /// own queue and return promptly.
    async fn on_turn_complete(&self, summary: &TurnSummary) -> Result<()>;
}

// ── NoopLearningSink ──────────────────────────────────────────────────────────

/// A [`LearningSink`] that accepts every summary and does nothing with it.
///
/// Useful as a composition default and in tests that want the call path
/// exercised without side effects. It is **not** the way to express "this host
/// has no learning pipeline" — pass `None` for that, so absence stays
/// distinguishable from a sink that ran and succeeded vacuously.
///
/// It never fails, which is the honest behaviour for a sink that never does
/// anything: an always-erroring default would push spurious failures into host
/// logs on every single turn.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopLearningSink;

impl NoopLearningSink {
    /// Creates a [`NoopLearningSink`].
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LearningSink for NoopLearningSink {
    async fn on_turn_complete(&self, _summary: &TurnSummary) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "learning_sink_tests.rs"]
mod tests;
