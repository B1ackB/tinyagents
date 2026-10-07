//! One run's repeat accounting: the successful-repeat tracker, the
//! warning-only pattern detectors and the post-compaction guard behind a
//! single host-neutral surface. A host middleware fingerprints results, feeds
//! each call in, and turns the answers into notes, blocks and halts.

use super::loop_patterns::{
    DEFAULT_CHURN_CALLS_PER_VARIANT, DEFAULT_CHURN_VARIANTS, DEFAULT_PING_PONG_ALTERNATIONS,
};
use super::post_compaction::{DEFAULT_POST_COMPACTION_WINDOW, REPEATING_AT};
use super::successful_repeat::{DEFAULT_REPEAT_CALL_THRESHOLD, DEFAULT_REPEAT_OUTPUT_THRESHOLD};
use super::types::{
    ArgumentChurnDetector, CallGate, CallObservation, PingPongDetector, PostCompactionGuard,
    RepeatEscalation, RepeatMonitor, RepeatProgressConfig, SuccessfulRepeat,
    SuccessfulRepeatTracker,
};

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

/// `tool\u{1}arguments`, with the tool name prefixed by its length so a
/// `\u{1}` inside either part cannot make two distinct calls collide.
fn call_signature(tool: &str, arguments_fingerprint: &str) -> String {
    format!("{}:{tool}\u{1}{arguments_fingerprint}", tool.len())
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
                    format!(
                        "`{tool}` with these arguments, returning this result, is what you were repeating right before the context was compacted; you may be looping. Use the result you have or take a different action."
                    ),
                );
            }
        }
        if !read_only {
            self.tracker.invalidate_predictions_except(&signature);
        }
        CallObservation { verdict, notes }
    }

    /// A successful call this monitor does not track (an exempt polling tool)
    /// ran and may have changed state: stop predicting every remembered result.
    pub fn note_untracked_success(&self, read_only: bool) {
        if !read_only {
            self.tracker.invalidate_all_predictions();
        }
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
