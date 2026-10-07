//! One run's repeat accounting: the successful-repeat tracker, the
//! warning-only pattern detectors and the post-compaction guard behind a
//! single host-neutral surface. A host middleware fingerprints results, feeds
//! each call in, and turns the answers into notes, blocks and halts.

use super::escalation::RepeatEscalation;
use super::loop_patterns::{
    ArgumentChurnDetector, DEFAULT_CHURN_CALLS_PER_VARIANT, DEFAULT_CHURN_VARIANTS,
    DEFAULT_PING_PONG_ALTERNATIONS, PingPongDetector,
};
use super::post_compaction::{DEFAULT_POST_COMPACTION_WINDOW, PostCompactionGuard, REPEATING_AT};
use super::successful_repeat::{DEFAULT_REPEAT_CALL_THRESHOLD, DEFAULT_REPEAT_OUTPUT_THRESHOLD};
use super::types::{CallGate, SuccessfulRepeat, SuccessfulRepeatTracker};

/// Settings for [`RepeatMonitor`] (and the middleware built on it).
///
/// The default is the staged ladder: warn at the first threshold (3 identical
/// results, 3 identical batches, 4 identical outputs), block two repeats
/// later, halt on the second block, plus the warning-only pattern detectors and
/// the post-compaction guard. [`immediate_halt`](Self::immediate_halt) restores
/// the historical halt-at-the-first-threshold behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeatProgressConfig {
    /// Consecutive identical output batches that trigger the first stage.
    pub output_threshold: u32,
    /// Identical results of one call, or identical call batches in a row, that
    /// trigger the first stage.
    pub call_threshold: u32,
    /// Staged escalation; `None` halts at the first threshold.
    pub escalation: Option<RepeatEscalation>,
    /// Alternating calls before the ping-pong warning; `0` disables it.
    pub ping_pong_alternations: u32,
    /// Argument variants before the churn warning; `0` disables it.
    pub churn_variants: u32,
    /// Calls per variant, with one result, that make a variant count.
    pub churn_calls_per_variant: u32,
    /// Calls watched after a compaction (and remembered before it); `0`
    /// disables the guard (warning-only).
    pub post_compaction_window: u32,
}

impl Default for RepeatProgressConfig {
    fn default() -> Self {
        Self {
            output_threshold: DEFAULT_REPEAT_OUTPUT_THRESHOLD,
            call_threshold: DEFAULT_REPEAT_CALL_THRESHOLD,
            escalation: Some(RepeatEscalation::default()),
            ping_pong_alternations: DEFAULT_PING_PONG_ALTERNATIONS,
            churn_variants: DEFAULT_CHURN_VARIANTS,
            churn_calls_per_variant: DEFAULT_CHURN_CALLS_PER_VARIANT,
            post_compaction_window: DEFAULT_POST_COMPACTION_WINDOW,
        }
    }
}

impl RepeatProgressConfig {
    /// The historical behaviour: halt the moment a repeat reaches its first
    /// threshold, with no warning stage, no blocking and no extra detectors.
    pub fn immediate_halt() -> Self {
        Self {
            escalation: None,
            ping_pong_alternations: 0,
            churn_variants: 0,
            post_compaction_window: 0,
            ..Self::default()
        }
    }
}

impl RepeatProgressConfig {
    /// Sets the identical-output streak that triggers the first stage.
    pub fn with_output_threshold(mut self, threshold: u32) -> Self {
        self.output_threshold = threshold;
        self
    }

    /// Sets the identical-call recurrence and batch streak that trigger the
    /// first stage.
    pub fn with_call_threshold(mut self, threshold: u32) -> Self {
        self.call_threshold = threshold;
        self
    }

    /// Sets staged escalation (`None` halts at the first threshold).
    pub fn with_escalation(mut self, escalation: Option<RepeatEscalation>) -> Self {
        self.escalation = escalation;
        self
    }

    /// Sets the ping-pong warning length; `0` disables it.
    pub fn with_ping_pong_alternations(mut self, alternations: u32) -> Self {
        self.ping_pong_alternations = alternations;
        self
    }

    /// Sets the argument-churn warning; `variants` of `0` disables it.
    pub fn with_churn(mut self, variants: u32, calls_per_variant: u32) -> Self {
        self.churn_variants = variants;
        self.churn_calls_per_variant = calls_per_variant;
        self
    }

    /// Sets the post-compaction watch window; `0` disables it.
    pub fn with_post_compaction_window(mut self, window: u32) -> Self {
        self.post_compaction_window = window;
        self
    }
}

/// What recording one successful call produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallObservation {
    /// The exact-repeat ledger's verdict for the call.
    pub verdict: SuccessfulRepeat,
    /// Warning notes to attach to this call's result, from the pattern
    /// detectors and the post-compaction guard.
    pub notes: Vec<String>,
}

/// See the module docs.
pub struct RepeatMonitor {
    tracker: SuccessfulRepeatTracker,
    ping_pong: Option<PingPongDetector>,
    churn: Option<ArgumentChurnDetector>,
    guard: Option<PostCompactionGuard>,
}

fn call_signature(tool: &str, arguments_fingerprint: &str) -> String {
    format!("{tool}\u{1}{arguments_fingerprint}")
}

impl RepeatMonitor {
    /// Builds a monitor for one run.
    pub fn new(config: &RepeatProgressConfig) -> Self {
        let mut tracker =
            SuccessfulRepeatTracker::new(config.output_threshold, config.call_threshold);
        if let Some(escalation) = config.escalation {
            tracker = tracker.with_escalation(escalation);
        }
        Self {
            tracker,
            ping_pong: (config.ping_pong_alternations > 0)
                .then(|| PingPongDetector::new(config.ping_pong_alternations)),
            churn: (config.churn_variants > 0).then(|| {
                ArgumentChurnDetector::new(config.churn_variants, config.churn_calls_per_variant)
            }),
            guard: (config.post_compaction_window > 0)
                .then(|| PostCompactionGuard::new(config.post_compaction_window)),
        }
    }

    /// See [`SuccessfulRepeatTracker::record_output`].
    pub fn record_output(&self, signature: &str, exempt: bool) -> SuccessfulRepeat {
        self.tracker.record_output(signature, exempt)
    }

    /// See [`SuccessfulRepeatTracker::record_call_batch`].
    pub fn record_call_batch(
        &self,
        signature: &str,
        all_successful: bool,
        exempt: bool,
    ) -> SuccessfulRepeat {
        self.tracker
            .record_call_batch(signature, all_successful, exempt)
    }

    /// Asked before `tool` runs with arguments fingerprinted as
    /// `arguments_fingerprint`; see [`SuccessfulRepeatTracker::pre_call`].
    pub fn pre_call(&self, tool: &str, arguments_fingerprint: &str) -> CallGate {
        self.tracker
            .pre_call(&call_signature(tool, arguments_fingerprint))
    }

    /// Records one successful, non-exempt call and the fingerprint of its
    /// result.
    ///
    /// `read_only` says the call cannot have changed state. After any other
    /// successful call the remembered results of the *other* calls are
    /// discarded, so [`pre_call`](Self::pre_call) never blocks a read on the
    /// prediction that an edit in between left it unchanged.
    pub fn record_call(
        &self,
        tool: &str,
        arguments_fingerprint: &str,
        outcome_identity: &str,
        read_only: bool,
    ) -> CallObservation {
        let signature = call_signature(tool, arguments_fingerprint);
        let verdict = self
            .tracker
            .record_call_identity(&signature, outcome_identity);
        let mut notes = Vec::new();
        if let Some(ping_pong) = &self.ping_pong {
            notes.extend(ping_pong.record(&signature, outcome_identity));
        }
        if let Some(churn) = &self.churn {
            notes.extend(churn.record(tool, arguments_fingerprint, outcome_identity));
        }
        if let Some(guard) = &self.guard {
            let repeating =
                self.tracker.recurrence_count(&signature, outcome_identity) >= REPEATING_AT;
            if guard.record(&signature, outcome_identity, repeating) {
                notes.push(
                    "this call and its result repeat what you were doing, over and over, right before the context was compacted; you may be looping. Use the result you have or take a different action."
                        .to_string(),
                );
            }
        }
        if !read_only {
            self.tracker.invalidate_predictions_except(&signature);
        }
        CallObservation { verdict, notes }
    }

    /// Context reduction removed results the model had seen: the ledger and
    /// detectors restart (the model cannot see those repeats any more), the
    /// block count survives, and the post-compaction guard starts watching.
    pub fn on_context_evicted(&self) {
        if let Some(guard) = &self.guard {
            guard.arm();
        }
        self.tracker.reset_ledger();
        if let Some(ping_pong) = &self.ping_pong {
            ping_pong.reset();
        }
        if let Some(churn) = &self.churn {
            churn.reset();
        }
    }

    /// Clears everything, for example when a paused run resumes.
    pub fn reset(&self) {
        self.tracker.reset();
        if let Some(ping_pong) = &self.ping_pong {
            ping_pong.reset();
        }
        if let Some(churn) = &self.churn {
            churn.reset();
        }
        if let Some(guard) = &self.guard {
            guard.reset();
        }
    }
}

#[cfg(test)]
#[path = "monitor_tests.rs"]
mod tests;
