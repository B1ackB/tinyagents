//! Per-turn recovery bookkeeping for the superstep loop.
//!
//! One logical turn can be re-issued several times before the model produces a
//! usable reply: a length-truncated empty reply is retried with a larger output
//! cap, a dropped or withheld tool call is re-prompted, an empty completion is
//! retried. Each of those recoveries is bounded by a counter on
//! [`crate::runtime::RunPolicy`], and the counters (plus the boosted output cap)
//! must be cleared the moment the turn *resolves* so a spent budget or a stale
//! cap never leaks into a later, unrelated turn.
//!
//! [`TurnRecovery`] owns all of that state and exposes one named reset per
//! kind of turn boundary, replacing the field-by-field resets that used to be
//! repeated at each exit of the loop body.

/// The recovery counters and boosted output cap of the turn in flight.
///
/// Every counter is consecutive-per-turn, not per-run (the output-validation
/// retry budget, which is run-wide, lives outside this struct).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct TurnRecovery {
    /// Retries of a length-truncated empty reply
    /// (`RunPolicy::truncated_empty_retries`).
    pub(super) truncated_empty_retries_used: u32,
    /// "Stop deliberating" re-prompts once the retries above are spent
    /// (`RunPolicy::truncated_empty_nudges`).
    pub(super) truncated_empty_nudges_used: u32,
    /// Retries of a normally-finished reply with no visible answer
    /// (`RunPolicy::empty_response_retries`).
    pub(super) empty_response_retries_used: u32,
    /// Consecutive "you said tool_calls but sent none" re-prompts
    /// (`RunPolicy::dropped_tool_call_nudges`).
    pub(super) dropped_tool_call_nudges_used: u32,
    /// Consecutive re-prompts after a call written on a turn with no callable
    /// tool (bounded by the same `dropped_tool_call_nudges`).
    pub(super) withheld_call_nudges_used: u32,
    /// Consecutive length-truncated tool turns answered with errors
    /// (`RunPolicy::truncated_tool_call_retries`); reset by any tool turn that
    /// was not cut off, never by the truncated turn itself.
    pub(super) truncated_tool_call_retries_used: u32,
    /// Overrides the next request's output cap after a length truncation.
    pub(super) boosted_max_tokens: Option<u32>,
    /// The original output cap, so growth stays clamped at 4x.
    pub(super) truncation_base: Option<u32>,
}

impl TurnRecovery {
    /// Grows the next request's output cap after a length-truncated reply:
    /// double the cap last sent, clamped at 4x the original. An unset cap
    /// stays unset (a plain retry is still worthwhile — the failure is
    /// stochastic).
    pub(super) fn boost_max_tokens(&mut self, attempt_max_tokens: Option<u32>) {
        if let Some(sent) = attempt_max_tokens {
            let base = *self.truncation_base.get_or_insert(sent);
            self.boosted_max_tokens = Some(
                self.boosted_max_tokens
                    .unwrap_or(sent)
                    .saturating_mul(2)
                    .min(base.saturating_mul(4)),
            );
        }
    }

    /// Clears the truncated-empty recovery state: both counters, the boosted
    /// output cap and the cap it grows from. The state is scoped to a single
    /// logical turn, so it must not carry into the turns after a recovered one.
    pub(super) fn reset_truncated_empty(&mut self) {
        self.truncated_empty_retries_used = 0;
        self.truncated_empty_nudges_used = 0;
        self.boosted_max_tokens = None;
        self.truncation_base = None;
    }

    /// Clears the re-prompt counters (dropped, withheld, empty-response).
    fn reset_nudges(&mut self) {
        self.dropped_tool_call_nudges_used = 0;
        self.withheld_call_nudges_used = 0;
        self.empty_response_retries_used = 0;
    }

    /// Clears everything that tracks a truncated turn: the truncated-tool-call
    /// retry counter and the truncated-empty state (including the boosted cap).
    fn reset_truncation(&mut self) {
        self.truncated_tool_call_retries_used = 0;
        self.reset_truncated_empty();
    }

    /// A tool-calling turn resolved (a plain one, or a mixed structured-output
    /// plus tool-call one): clear the re-prompt counters so a spent counter
    /// cannot leak into a later dropped-call turn.
    ///
    /// A turn whose call was cut off by the output limit
    /// (`turn_had_truncated_calls`) keeps its truncated-tool-call retry budget
    /// and boosted output cap for the retry.
    pub(super) fn reset_after_tool_turn(&mut self, turn_had_truncated_calls: bool) {
        self.reset_nudges();
        if !turn_had_truncated_calls {
            self.reset_truncation();
        }
    }

    /// The turn resolved without a tool call and without scheduling another
    /// retry (it is about to be taken as the answer): clear every counter and
    /// the boosted cap, which would otherwise override the caller's per-turn
    /// cap on every later call.
    pub(super) fn reset_after_final(&mut self) {
        self.reset_nudges();
        self.reset_truncation();
    }
}

#[cfg(test)]
#[path = "turn_recovery_tests.rs"]
mod tests;
