//! Host seam for deciding what a tool result *means*.
//!
//! The runtime knows whether a tool call returned; it does not know whether
//! what came back should be treated as a success, retried, or given up on.
//! That judgement is product knowledge, and it is tool-specific: a `404` from a
//! CRM lookup usually means "no such record" — a perfectly good empty result —
//! while a `404` from a document fetch means the run cannot proceed. The same
//! transport-level shape lands in two different classes depending on which
//! integration produced it, so no rule the crate could hard-code would be right
//! for both. Hence a trait: the host, which owns the integration catalogue,
//! answers.
//!
//! This capability is **optional** (RFC §3.8). A host that has no per-tool
//! knowledge passes `None` and the runtime skips classification entirely,
//! rather than being handed a stub that always answers
//! [`OutcomeClass::Success`]. Absence and "classified as fine" are different
//! facts, and conflating them is exactly the failure mode RFC §2.3 forbids.
//! [`ErrorFieldClassifier`] exists for hosts that want the obvious baseline
//! *deliberately*, not as a silent default.
//!
//! **Reuses [`ToolResult`] on purpose.** There is no parallel
//! "classifiable result" type here. A second result struct would have to be
//! populated by the runtime from the real one, and every field added to
//! [`ToolResult`] would then either need mirroring or would be invisible to
//! classifiers — a drift surface with no upside. Classifiers read the real
//! result, including its structured content blocks. Hosts that need richer
//! error codes keep their taxonomy beside the classifier rather than adding
//! provider metadata to the shared tool result.
//!
//! **Dependency rule:** `serde` + `std` only, matching the inert-value-type
//! carve-out in `crate::config`'s type definitions. A host can implement this
//! trait without pulling an engine.

use serde::{Deserialize, Serialize};

use tinytools::ToolResult;

// ── OutcomeClass ──────────────────────────────────────────────────────────────

/// What the runtime should do with a tool result.
///
/// Three variants, not a boolean, because "failed" and "worth trying again"
/// are independent facts and the loop reacts differently to each: a retryable
/// failure may be re-dispatched without telling the model anything new, while a
/// permanent failure must be surfaced into the transcript so the model can pick
/// a different approach instead of burning iterations on the same call.
///
/// Deliberately closed and coarse. A richer taxonomy (rate-limited,
/// unauthorized, not-found, …) would be transport vocabulary leaking into a
/// decision the host has already made — the host maps its own detail *into*
/// these three and keeps the detail on its side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeClass {
    /// The call did what was asked. Includes legitimately empty results — a
    /// search that matched nothing succeeded.
    #[default]
    Success,
    /// The call failed for a reason that may not recur: a timeout, a throttle,
    /// a transient upstream error. The runtime marks the transcript result as
    /// retryable so the model may choose a new attempt within normal turn
    /// limits; it never silently re-dispatches an action.
    ///
    /// A classifier must only return this for calls it believes are safe to
    /// repeat. The runtime cannot know whether a tool had side effects, so this
    /// variant is an assertion by the host that repeating is acceptable.
    RetryableFailure,
    /// The call failed in a way that repeating will not fix: bad arguments,
    /// missing permissions, an entity that does not exist. The runtime should
    /// surface it to the model rather than retry.
    PermanentFailure,
}

impl OutcomeClass {
    /// Whether the result should be treated as a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Whether this result is surfaced as retryable to the model.
    ///
    /// Only [`RetryableFailure`](Self::RetryableFailure) is retryable —
    /// [`Success`](Self::Success) has nothing to retry, and a
    /// [`PermanentFailure`](Self::PermanentFailure) must be surfaced as a
    /// terminal tool failure. Any model-selected retry remains bounded by the
    /// run's ordinary tool and iteration limits.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::RetryableFailure)
    }

    /// Whether this class represents a failure of either kind.
    ///
    /// Useful for reporting paths that care that something went wrong but not
    /// what the loop should do about it.
    pub fn is_failure(&self) -> bool {
        !self.is_success()
    }
}

// ── ToolOutcomeClassifier ─────────────────────────────────────────────────────

/// Classifies a completed tool result (RFC §3.8).
///
/// Synchronous by design. Classification runs on the turn's critical path once
/// per tool result, and an `async` signature would invite implementations that
/// perform I/O there — a network round trip to decide whether a network round
/// trip failed. A host needing external knowledge should load it up front and
/// consult it from memory here.
///
/// Implementations must be pure with respect to the runtime: no side effects
/// the loop depends on, and the same `(name, result)` pair should classify the
/// same way every time. The runtime may call `classify` more than once for one
/// result (for example when both the retry decision and a reporting path need
/// it) and does not memoize.
pub trait ToolOutcomeClassifier: Send + Sync {
    /// Classifies `result`, which was produced by the tool named `name`.
    ///
    /// `name` is passed separately from `result.name` because the per-tool
    /// judgement is the whole point of this seam — the classifier is expected
    /// to branch on it — and because the runtime knows the name it dispatched
    /// even when a misbehaving tool returns a result stamped with a different
    /// one. Treat `name` as authoritative.
    fn classify(&self, name: &str, result: &ToolResult) -> OutcomeClass;
}

// ── ErrorFieldClassifier ──────────────────────────────────────────────────────

/// The baseline classifier: reads [`ToolResult::is_error`] and nothing else.
///
/// `true` becomes [`OutcomeClass::PermanentFailure`], `false` becomes
/// [`OutcomeClass::Success`]. It ignores tool name and content, because
/// interpreting any of those would require knowing which tool ran — precisely
/// the knowledge this crate does not have.
///
/// **Why `PermanentFailure` rather than `RetryableFailure`.** Retrying is the
/// answer that can cause harm: the runtime does not know whether the tool had
/// side effects, so a blanket retry can double-send a message or double-charge
/// a card, and at best it spends turn iterations re-earning the same error.
/// Classifying as permanent surfaces the error to the model, which can then
/// choose to call again with different arguments — a strictly safer default,
/// and the retryable half is exactly what a host is expected to override.
///
/// Suitable for tests, local runs, and hosts that genuinely have no per-tool
/// error taxonomy. A host with one should implement
/// [`ToolOutcomeClassifier`] itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ErrorFieldClassifier;

impl ErrorFieldClassifier {
    /// Creates the classifier. Stateless — every instance behaves identically.
    pub fn new() -> Self {
        Self
    }
}

impl ToolOutcomeClassifier for ErrorFieldClassifier {
    fn classify(&self, _name: &str, result: &ToolResult) -> OutcomeClass {
        if result.is_error {
            OutcomeClass::PermanentFailure
        } else {
            OutcomeClass::Success
        }
    }
}

#[cfg(test)]
#[path = "tool_outcome_classifier_tests.rs"]
mod tests;
