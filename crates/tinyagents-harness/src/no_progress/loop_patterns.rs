//! Warning-only detectors for two loop shapes the exact-repeat ledger misses.
//!
//! - [`PingPongDetector`]: two different calls taking turns (A, B, A, B, ...),
//!   each returning the same result every time.
//! - [`ArgumentChurnDetector`]: one tool called with many *different*
//!   arguments that all come back with the same result, so the model is
//!   varying the input without changing what it learns.
//!
//! Neither blocks or halts. A legitimate workflow can look like either for a
//! while, so each only reports a note (once per pattern) for the host to
//! attach to the tool result the model is about to read. Callers feed in
//! successful, non-exempt calls with the result fingerprint
//! ([`OutcomeFingerprinter`](super::OutcomeFingerprinter)) so volatile spans
//! do not hide a repeat.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use super::util::{hash_of, lock};

/// Alternations (A,B,A,B,A,B is six) before [`PingPongDetector`] warns.
pub const DEFAULT_PING_PONG_ALTERNATIONS: u32 = 6;
/// Distinct argument variants [`ArgumentChurnDetector`] needs before it warns.
pub const DEFAULT_CHURN_VARIANTS: u32 = 3;
/// Calls each variant needs, all with the same result, to count as a variant.
pub const DEFAULT_CHURN_CALLS_PER_VARIANT: u32 = 3;

/// One call and the result it produced, as hashes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Step {
    call: u64,
    outcome: u64,
}

#[derive(Default)]
struct PingPongState {
    prev: Option<Step>,
    last: Option<Step>,
    /// Tool names of `prev` and `last`, for the warning text.
    names: (String, String),
    /// Length of the current strictly alternating tail ending at `last`.
    tail: u32,
    /// Pairs already warned about (order-independent hash of both calls).
    warned: HashSet<u64>,
}

/// Detects two calls alternating with a stable result on each side.
pub struct PingPongDetector {
    alternations: u32,
    state: Mutex<PingPongState>,
}

impl Default for PingPongDetector {
    fn default() -> Self {
        Self::new(DEFAULT_PING_PONG_ALTERNATIONS)
    }
}

impl PingPongDetector {
    /// Warns once the alternating tail reaches `alternations` calls, clamped
    /// up to four (two turns each) so a single repeat is never a ping-pong.
    pub fn new(alternations: u32) -> Self {
        Self {
            alternations: alternations.max(4),
            state: Mutex::default(),
        }
    }

    /// Records one successful call (`call_signature` identifies tool and
    /// arguments, `outcome_identity` the fingerprinted result). Returns a note
    /// the first time the pair has alternated enough.
    pub fn record(&self, call_signature: &str, outcome_identity: &str) -> Option<String> {
        let step = Step {
            call: hash_of(call_signature),
            outcome: hash_of(outcome_identity),
        };
        let mut state = lock(&self.state);
        state.tail = match (state.prev, state.last) {
            // Same two calls with the same results as two steps ago.
            (Some(prev), Some(last)) if prev == step && last.call != step.call => state.tail + 1,
            // A different call than last time starts a fresh pair.
            (_, Some(last)) if last.call != step.call => 2,
            _ => 1,
        };
        state.prev = state.last;
        state.last = Some(step);
        let tool = call_signature.split('\u{1}').next().unwrap_or_default();
        state.names = (std::mem::take(&mut state.names.1), tool.to_string());
        if state.tail < self.alternations {
            return None;
        }
        let other = state.prev.map_or(0, |prev| prev.call);
        let pair = step.call ^ other;
        if !state.warned.insert(pair) {
            return None;
        }
        Some(format!(
            "calls to `{}` and `{}` have been alternating {} times, each returning the same result every time; going back and forth between them is not making progress. Use the results you already have or change approach.",
            state.names.0, state.names.1, state.tail
        ))
    }

    /// Forgets the tail and the pairs already warned about.
    pub fn reset(&self) {
        *lock(&self.state) = PingPongState::default();
    }
}

/// Detects one tool called with many argument variants that all return the
/// same result.
pub struct ArgumentChurnDetector {
    variants: u32,
    calls_per_variant: u32,
    /// `(tool, outcome)` → argument variant → calls so far.
    groups: Mutex<HashMap<(String, u64), HashMap<u64, u32>>>,
    warned: Mutex<HashSet<(String, u64)>>,
}

impl Default for ArgumentChurnDetector {
    fn default() -> Self {
        Self::new(DEFAULT_CHURN_VARIANTS, DEFAULT_CHURN_CALLS_PER_VARIANT)
    }
}

impl ArgumentChurnDetector {
    /// Warns once `variants` distinct argument sets of one tool have each
    /// returned the same result `calls_per_variant` times. Both are clamped to
    /// at least two.
    pub fn new(variants: u32, calls_per_variant: u32) -> Self {
        Self {
            variants: variants.max(2),
            calls_per_variant: calls_per_variant.max(2),
            groups: Mutex::default(),
            warned: Mutex::default(),
        }
    }

    /// Records one successful call of `tool` with the fingerprint of its
    /// arguments and of its result. Returns a note the first time a tool and
    /// result qualify.
    pub fn record(
        &self,
        tool: &str,
        arguments_fingerprint: &str,
        outcome_identity: &str,
    ) -> Option<String> {
        let key = (tool.to_string(), hash_of(outcome_identity));
        let qualifying = {
            let mut groups = lock(&self.groups);
            let variants = groups.entry(key.clone()).or_default();
            *variants.entry(hash_of(arguments_fingerprint)).or_insert(0) += 1;
            variants
                .values()
                .filter(|calls| **calls >= self.calls_per_variant)
                .count() as u32
        };
        if qualifying < self.variants || !lock(&self.warned).insert(key) {
            return None;
        }
        Some(format!(
            "`{tool}` has been called with {qualifying} different sets of arguments, each at least {} times, and every one returned the same result; changing the arguments is not changing what you learn. Try a different tool or approach.",
            self.calls_per_variant
        ))
    }

    /// Forgets every count and warning.
    pub fn reset(&self) {
        lock(&self.groups).clear();
        lock(&self.warned).clear();
    }
}

#[cfg(test)]
#[path = "loop_patterns_tests.rs"]
mod tests;
