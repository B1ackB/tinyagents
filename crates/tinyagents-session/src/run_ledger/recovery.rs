//! Advisory restart-recovery classification for dangling tool calls.
//!
//! After a crash, the tail of a transcript can end with assistant tool calls
//! that never received a result. Whether the loop may simply continue depends on
//! what actually happened to each call, and the per-call effect ledger
//! ([`super::tool_effects`]) is the durable record of that: it writes `started`
//! *before* a call executes and settles it afterwards, so
//!
//! - no row means the call never began,
//! - a `started` row means it may or may not have committed its side effects,
//! - a settled row means its outcome is known,
//! - a `deferred` row means the call paused and awaits an answer.
//!
//! [`classify_recovery`] turns that into one advisory [`RecoveryClass`] per
//! dangling call. It is pure — no database access, no mutation — and advisory:
//! the host decides what to do with the verdict, and nothing here re-runs a
//! tool. The classification assumes the effect ledger was attached to the run;
//! with no ledger every call looks "never began", so hosts without one should
//! not rely on [`RecoveryClass::Resume`].

use super::tool_effects::{ToolEffectRow, ToolEffectStatus};

/// A tool call at the transcript tail that has no recorded result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DanglingToolCall {
    /// The run that issued the call.
    pub run_id: String,
    /// The provider/harness call id.
    pub call_id: String,
    /// The tool name, for diagnostics.
    pub tool: String,
}

/// How safely a run can continue past one dangling call. Ordered from least to
/// most cautious, so the maximum over a set is the set's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecoveryClass {
    /// Continue normally: the call never began, so no side effect is in doubt.
    Resume,
    /// Continue, but only to *report*: the call settled (`Completed`/`Failed`)
    /// so its outcome is recorded. Surface that outcome; do not re-execute it.
    ResumeReportOnly,
    /// The call paused itself (`Deferred`: approval or a deferred result) and
    /// is waiting on an answer. Resume by *answering the deferral*; never
    /// re-execute the call, and never treat it as a plain resume — the call has
    /// begun and its answer must come from the same approver/source.
    AwaitingAnswer,
    /// The call started and never settled (or was already marked
    /// `Interrupted`): it may have committed. Verify the real-world state
    /// before continuing, and never blindly re-run it.
    NeedsVerification,
}

/// The verdict for one dangling call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRecovery {
    /// The call that was classified.
    pub call: DanglingToolCall,
    /// The advisory verdict.
    pub class: RecoveryClass,
    /// The ledger status the verdict was based on; `None` when no row exists.
    pub effect_status: Option<ToolEffectStatus>,
    /// One-line, host-safe justification.
    pub reason: &'static str,
}

fn status_rank(status: ToolEffectStatus) -> u8 {
    match status {
        ToolEffectStatus::Completed => 0,
        ToolEffectStatus::Failed => 1,
        ToolEffectStatus::Deferred => 2,
        ToolEffectStatus::Started => 3,
        ToolEffectStatus::Interrupted => 4,
    }
}

fn classify_status(status: Option<ToolEffectStatus>) -> (RecoveryClass, &'static str) {
    match status {
        None => (
            RecoveryClass::Resume,
            "no effect record: the call never began",
        ),
        Some(ToolEffectStatus::Deferred) => (
            RecoveryClass::AwaitingAnswer,
            "deferred: answer the pending deferral, do not re-execute",
        ),
        Some(ToolEffectStatus::Completed) => (
            RecoveryClass::ResumeReportOnly,
            "completed: outcome is recorded, report it and do not re-execute",
        ),
        Some(ToolEffectStatus::Failed) => (
            RecoveryClass::ResumeReportOnly,
            "failed: outcome is recorded, report it and do not re-execute",
        ),
        Some(ToolEffectStatus::Started) => (
            RecoveryClass::NeedsVerification,
            "started but never settled: side effects are uncertain",
        ),
        Some(ToolEffectStatus::Interrupted) => (
            RecoveryClass::NeedsVerification,
            "interrupted: marked unsafe to re-execute, verify before continuing",
        ),
    }
}

/// What a call with no effect row means. A missing row proves the call never
/// began only when every failed `started` write aborts the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingEffectRow {
    /// The call never began (every `started` write failure aborted execution).
    #[default]
    NeverBegan,
    /// The host runs with a policy that proceeds after a failed `started` write
    /// (e.g. `LedgerFailure::Continue`), so a missing row may hide an executed
    /// call: classify it [`RecoveryClass::NeedsVerification`].
    Uncertain,
}

/// Classifies each dangling call in `tail` against the recorded
/// `tool_effects`, matching on `(run_id, call_id)`. Output order follows `tail`.
/// A missing row is read as "never began"; use [`classify_recovery_with`] when
/// the run could have continued past a failed `started` write.
pub fn classify_recovery(
    tail: &[DanglingToolCall],
    tool_effects: &[ToolEffectRow],
) -> Vec<CallRecovery> {
    classify_recovery_with(tail, tool_effects, MissingEffectRow::NeverBegan)
}

/// Like [`classify_recovery`], with an explicit reading of missing rows.
pub fn classify_recovery_with(
    tail: &[DanglingToolCall],
    tool_effects: &[ToolEffectRow],
    missing: MissingEffectRow,
) -> Vec<CallRecovery> {
    tail.iter()
        .map(|call| {
            // Several rows can match one call (e.g. a replayed write); take the
            // most cautious verdict so the order of the rows never matters.
            let (effect_status, class, reason) = tool_effects
                .iter()
                .filter(|effect| effect.run_id == call.run_id && effect.call_id == call.call_id)
                .map(|effect| {
                    let (class, reason) = classify_status(Some(effect.status));
                    (Some(effect.status), class, reason)
                })
                // Equal classes (Completed vs Failed) tie-break on a fixed status
                // rank so the chosen row never depends on row order.
                .max_by_key(|(status, class, _)| (*class, status.map(status_rank)))
                .unwrap_or_else(|| match missing {
                    MissingEffectRow::NeverBegan => {
                        let (class, reason) = classify_status(None);
                        (None, class, reason)
                    }
                    MissingEffectRow::Uncertain => (
                        None,
                        RecoveryClass::NeedsVerification,
                        "no effect record, but the ledger may have failed to record the start: verify before continuing",
                    ),
                });
            tracing::debug!(
                "[run_ledger:recovery] run_id={} call_id={} tool={} class={class:?}",
                call.run_id,
                call.call_id,
                call.tool
            );
            CallRecovery {
                call: call.clone(),
                class,
                effect_status,
                reason,
            }
        })
        .collect()
}

/// The most cautious class among `classified`; [`RecoveryClass::Resume`] when
/// there is nothing dangling.
pub fn overall_recovery(classified: &[CallRecovery]) -> RecoveryClass {
    classified
        .iter()
        .map(|recovery| recovery.class)
        .max()
        .unwrap_or(RecoveryClass::Resume)
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod tests;
