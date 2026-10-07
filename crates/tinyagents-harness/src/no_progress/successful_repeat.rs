//! Successful-repeat progress detection.
//!
//! [`NoProgressTracker`] handles failing tool calls, but deliberately resets on
//! success. That leaves a second loop shape undetected: a model can repeatedly
//! emit the same response and successfully invoke the same no-op tool call.
//! This tracker owns the provider- and product-neutral streak accounting for
//! those loops; a harness middleware remains responsible for building canonical
//! signatures and deciding which polling tools are exempt.
//!
//! The streaks only see back-to-back repeats. A model that cycles through
//! several successful calls (A, B, A, B, ...) changes the signature every step
//! and never trips them, so the tracker also keeps a run-wide ledger of
//! `(call, result)` recurrences fed by
//! [`SuccessfulRepeatTracker::record_call_outcome`].

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::escalation::RepeatEscalation;
use super::fingerprint::{OutcomeFingerprinter, VolatileSpanNormalizer};
use super::types::{CallGate, Streak, SuccessfulRepeat, SuccessfulRepeatTracker};

/// Consecutive identical assistant-output batches required to halt.
pub const DEFAULT_REPEAT_OUTPUT_THRESHOLD: u32 = 4;
/// Consecutive identical successful tool-call batches required to halt.
pub const DEFAULT_REPEAT_CALL_THRESHOLD: u32 = 3;

impl Streak {
    /// Hashes `signature` and either extends the current run (same hash as
    /// last time) or starts a new one, returning the run length after the
    /// update. Hashing rather than storing the signature keeps the tracker
    /// cheap to hold for a whole turn.
    fn record(&mut self, signature: &str) -> u32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        signature.hash(&mut hasher);
        let hash = hasher.finish();
        if self.last_hash == Some(hash) {
            self.consecutive += 1;
        } else {
            self.last_hash = Some(hash);
            self.consecutive = 1;
        }
        self.consecutive
    }

    /// Clears the run back to its default (no streak).
    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for SuccessfulRepeatTracker {
    fn default() -> Self {
        Self::new(
            DEFAULT_REPEAT_OUTPUT_THRESHOLD,
            DEFAULT_REPEAT_CALL_THRESHOLD,
        )
    }
}

impl SuccessfulRepeatTracker {
    /// Builds a tracker. Thresholds are clamped to one so `0` cannot disable a
    /// safety guard accidentally; callers that do not want this guard should
    /// omit the tracker.
    pub fn new(output_threshold: u32, call_threshold: u32) -> Self {
        Self {
            output_threshold: output_threshold.max(1),
            call_threshold: call_threshold.max(1),
            fingerprinter: Arc::new(VolatileSpanNormalizer),
            output: std::sync::Mutex::new(Streak::default()),
            calls: std::sync::Mutex::new(Streak::default()),
            recurrences: std::sync::Mutex::new(std::collections::HashMap::new()),
            escalation: None,
            last_outcome: std::sync::Mutex::new(std::collections::HashMap::new()),
            blocks: std::sync::Mutex::new(0),
        }
    }

    /// Turns on staged escalation: each first threshold (`call_threshold`,
    /// `output_threshold`) only reports [`SuccessfulRepeat::Warn`], a call is
    /// then blocked through [`pre_call`](Self::pre_call), and the run halts on
    /// the second block. Without this the first threshold halts immediately.
    pub fn with_escalation(mut self, escalation: RepeatEscalation) -> Self {
        self.escalation = Some(escalation);
        self
    }

    /// Maps a streak length to its verdict: `Continue` below `threshold`;
    /// without escalation `Halt` from `threshold` on; with it `Warn` exactly at
    /// `threshold`, `Continue` until `threshold + gap`, then `Halt`.
    fn streak_verdict(
        &self,
        consecutive: u32,
        threshold: u32,
        warn: impl FnOnce() -> String,
        halt: impl FnOnce() -> String,
    ) -> SuccessfulRepeat {
        match self.escalation {
            None if consecutive >= threshold => SuccessfulRepeat::Halt(halt()),
            Some(escalation) if consecutive >= threshold + escalation.gap() => {
                SuccessfulRepeat::Halt(halt())
            }
            Some(_) if consecutive == threshold => SuccessfulRepeat::Warn(warn()),
            _ => SuccessfulRepeat::Continue,
        }
    }

    /// Replaces the fingerprinter that reduces a tool result to the identity
    /// [`record_call_outcome`](Self::record_call_outcome) keys on. The default
    /// is [`VolatileSpanNormalizer`], which ignores timestamps, durations and
    /// request ids; pass a different one to tighten or relax what counts as
    /// "the same result".
    pub fn with_fingerprinter(mut self, fingerprinter: Arc<dyn OutcomeFingerprinter>) -> Self {
        self.fingerprinter = fingerprinter;
        self
    }

    /// Stages the canonical visible-output plus tool-call signature produced
    /// by one assistant iteration. A threshold crossing is not reported until
    /// [`record_call_batch`](Self::record_call_batch) confirms that the
    /// associated batch succeeded and is not exempt.
    pub fn record_output(&self, signature: &str, exempt: bool) -> SuccessfulRepeat {
        let mut output = self.output.lock().unwrap();
        if exempt {
            output.reset();
            return SuccessfulRepeat::Continue;
        }
        output.record(signature);
        SuccessfulRepeat::Continue
    }

    /// Records the canonical tool-name/arguments signature after the whole
    /// batch completes. Failed or exempt batches reset the successful streak.
    pub fn record_call_batch(
        &self,
        signature: &str,
        all_successful: bool,
        exempt: bool,
    ) -> SuccessfulRepeat {
        if exempt || !all_successful {
            // Output is observed before the completed batch can be classified.
            // An exempt polling batch or a failure therefore resets both
            // trackers so its preceding output cannot leak into the next
            // progress-eligible iteration.
            self.output.lock().unwrap().reset();
            self.calls.lock().unwrap().reset();
            return SuccessfulRepeat::Continue;
        }
        let output_consecutive = self.output.lock().unwrap().consecutive;
        let output_verdict = self.streak_verdict(
            output_consecutive,
            self.output_threshold,
            || format!("the last {output_consecutive} iterations produced the identical response and tool call with no change; you are repeating the same step without making progress."),
            || format!("Stopping: the last {output_consecutive} iterations produced the identical response and tool call with no change; the run is stuck repeating the same step without making progress."),
        );
        // The call streak is recorded even when the output streak already
        // decided, so a warning does not leave it a batch behind.
        let consecutive = self.calls.lock().unwrap().record(signature);
        let call_verdict = self.streak_verdict(
            consecutive,
            self.call_threshold,
            || format!("the same successful tool-call batch was issued {consecutive} times in a row with identical arguments and no new information; you are repeating one action without making progress."),
            || format!("Stopping: the same successful tool-call batch was issued {consecutive} times in a row with identical arguments and no new information; the run is stuck repeating one action without making progress."),
        );
        // Halt outranks warn; the output verdict wins a tie, as it always has.
        match (output_verdict, call_verdict) {
            (halt @ SuccessfulRepeat::Halt(_), _) | (_, halt @ SuccessfulRepeat::Halt(_)) => halt,
            (warn @ SuccessfulRepeat::Warn(_), _) | (_, warn @ SuccessfulRepeat::Warn(_)) => warn,
            _ => SuccessfulRepeat::Continue,
        }
    }

    /// Records one successful, non-exempt tool call with the result it returned,
    /// and halts once that call has returned that same result `call_threshold`
    /// times in this run, whether or not the repeats were adjacent.
    ///
    /// An identical call that returns an identical result adds nothing the
    /// transcript does not already hold, however far apart the repeats are, so
    /// the ledger spans the run rather than a window: there is no size to tune,
    /// and it holds for two-step and longer cycles alike. Keying on the result
    /// means a re-read after state changed (same call, different result) does
    /// not count. The threshold is the same `call_threshold` the adjacent
    /// streak uses, because this is the same "identical successful call"
    /// verdict with a stricter key.
    ///
    /// Call it once per call, not per batch, and only for calls that succeeded
    /// and are not exempt polling calls; failures belong to
    /// [`NoProgressTracker`](super::NoProgressTracker). Failed or exempt batches
    /// deliberately do not clear this ledger: a failing sibling call does not
    /// make an identical re-read informative. A driver whose context compaction
    /// evicts earlier tool results should call [`reset`](Self::reset) when it
    /// does, since re-reading an evicted result is not a repeat the model can
    /// see.
    pub fn record_call_outcome(
        &self,
        call_signature: &str,
        outcome_signature: &str,
    ) -> SuccessfulRepeat {
        let outcome_identity = self.fingerprinter.fingerprint(outcome_signature);
        self.record_call_identity(call_signature, &outcome_identity)
    }

    /// [`record_call_outcome`](Self::record_call_outcome) for a caller that has
    /// already reduced the result to its identity with an
    /// [`OutcomeFingerprinter`]. Fingerprinting scans the whole result, so a
    /// host that shares this tracker behind a lock should fingerprint first
    /// and call this while holding the lock.
    pub fn record_call_identity(
        &self,
        call_signature: &str,
        outcome_identity: &str,
    ) -> SuccessfulRepeat {
        let key = ledger_key(call_signature, outcome_identity);
        let mut recurrences = self.recurrences.lock().unwrap();
        let count = recurrences.entry(key).or_insert(0);
        *count += 1;
        let count = *count;
        drop(recurrences);
        self.last_outcome
            .lock()
            .unwrap()
            .insert(hash_of(call_signature), key);
        let warn = || {
            format!(
                "the same successful tool call returned the identical result {count} times in this run; re-running steps whose results are already in the conversation adds no new information, so you are cycling without making progress."
            )
        };
        let halt = || {
            format!(
                "Stopping: the same successful tool call returned the identical result {count} times in this run; re-running steps whose results are already in the conversation adds no new information, so the run is cycling without making progress."
            )
        };
        match self.escalation {
            None if count >= self.call_threshold => SuccessfulRepeat::Halt(halt()),
            // Only reachable when the host never consults `pre_call`: the
            // block did not happen, so stop rather than loop on.
            Some(escalation) if count >= self.block_count(escalation) => {
                SuccessfulRepeat::Halt(halt())
            }
            Some(_) if count == self.call_threshold => SuccessfulRepeat::Warn(warn()),
            _ => SuccessfulRepeat::Continue,
        }
    }

    /// Ledger count at which a call is blocked (the Nth identical call is not
    /// executed).
    fn block_count(&self, escalation: RepeatEscalation) -> u32 {
        self.call_threshold + escalation.gap()
    }

    /// Asked *before* a call executes: whether running it would only repeat a
    /// result the model already holds.
    ///
    /// With staged escalation, a call whose last result has already recurred
    /// `block_after_warn` times past the warning is answered with
    /// [`CallGate::Block`] instead of running; the `blocks_before_halt`-th
    /// block in the run answers [`CallGate::Halt`]. A call never seen, or whose
    /// last result differed, is allowed. Without escalation this is always
    /// [`CallGate::Allow`].
    pub fn pre_call(&self, call_signature: &str) -> CallGate {
        let Some(escalation) = self.escalation else {
            return CallGate::Allow;
        };
        let Some(key) = self
            .last_outcome
            .lock()
            .unwrap()
            .get(&hash_of(call_signature))
            .copied()
        else {
            return CallGate::Allow;
        };
        let count = self
            .recurrences
            .lock()
            .unwrap()
            .get(&key)
            .copied()
            .unwrap_or(0);
        if count + 1 < self.block_count(escalation) {
            return CallGate::Allow;
        }
        let mut blocks = self.blocks.lock().unwrap();
        *blocks += 1;
        if *blocks >= escalation.halt_block() {
            return CallGate::Halt(format!(
                "Stopping: the same successful tool call was blocked {blocks} times for returning the identical result {count} times; the model kept re-issuing it after being warned, so the run is stuck cycling without making progress."
            ));
        }
        CallGate::Block(format!(
            "Blocked: this exact call has already returned the identical result {count} times and was not executed again. Its result is already in the conversation. Reassess: use that result, or take a different action. Issuing it again will stop the run."
        ))
    }

    /// Jumps a call straight to the block stage: its next attempt with the
    /// same result is blocked by [`pre_call`](Self::pre_call), without waiting
    /// for the ledger to count up. A no-op without staged escalation.
    pub fn escalate_to_block(&self, call_signature: &str, outcome_identity: &str) {
        let Some(escalation) = self.escalation else {
            return;
        };
        let key = ledger_key(call_signature, outcome_identity);
        let mut recurrences = self.recurrences.lock().unwrap();
        let count = recurrences.entry(key).or_insert(0);
        *count = (*count).max(self.block_count(escalation) - 1);
        drop(recurrences);
        self.last_outcome
            .lock()
            .unwrap()
            .insert(hash_of(call_signature), key);
    }

    /// Clears both streaks, the recurrence ledger and the block count, for
    /// example when a paused run is resumed.
    pub fn reset(&self) {
        self.reset_ledger();
        *self.blocks.lock().unwrap() = 0;
    }

    /// Clears the streaks and the recurrence ledger but keeps the run-wide
    /// block count: for a context eviction, where the model forgets the
    /// results it repeated but has still already been blocked once.
    pub fn reset_ledger(&self) {
        self.output.lock().unwrap().reset();
        self.calls.lock().unwrap().reset();
        self.recurrences.lock().unwrap().clear();
        self.last_outcome.lock().unwrap().clear();
    }
}

fn hash_of(value: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn ledger_key(call_signature: &str, outcome_identity: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (call_signature, outcome_identity).hash(&mut hasher);
    hasher.finish()
}
