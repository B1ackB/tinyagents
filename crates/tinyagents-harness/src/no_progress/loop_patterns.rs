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

/// The tool name inside a call signature: `tool\u{1}args`, optionally with the
/// tool name prefixed by its byte length (`4:tool\u{1}args`).
fn tool_name(call_signature: &str) -> &str {
    if let Some((len, rest)) = call_signature.split_once(':')
        && let Ok(len) = len.parse::<usize>()
        && let Some(name) = rest.get(..len)
    {
        return name;
    }
    call_signature.split('\u{1}').next().unwrap_or_default()
}

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
            (Some(prev), Some(last)) if prev == step && last.call != step.call => state.tail.saturating_add(1),
            // A different call than last time starts a fresh pair.
            (_, Some(last)) if last.call != step.call => 2,
            _ => 1,
        };
        state.prev = state.last;
        state.last = Some(step);
        let tool = tool_name(call_signature);
        state.names = (std::mem::take(&mut state.names.1), tool.to_string());
        if state.tail < self.alternations {
            return None;
        }
        let other = state.prev.map_or(0, |prev| prev.call);
        let pair = step.call ^ other;
        if state.warned.len() >= MAX_WARNED && !state.warned.contains(&pair) {
            return None;
        }
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

/// Warned patterns remembered per detector; past it the detector stops
/// warning rather than grow without bound.
const MAX_WARNED: usize = 1024;
/// `(tool, outcome)` groups tracked at once; later groups are not tracked.
const MAX_CHURN_GROUPS: usize = 1024;
/// Argument variants tracked per group; later variants are not tracked.
const MAX_CHURN_VARIANTS_PER_GROUP: usize = 256;

#[derive(Default)]
struct ChurnState {
    /// `(tool, outcome)` → argument variant → calls so far.
    groups: HashMap<(String, u64), HashMap<u64, u32>>,
    /// Groups already warned about; their counts are dropped.
    warned: HashSet<(String, u64)>,
}

/// Detects one tool called with many argument variants that all return the
/// same result. Tracking is bounded ([`MAX_CHURN_GROUPS`],
/// [`MAX_CHURN_VARIANTS_PER_GROUP`]) so a run supplying unique values cannot
/// grow it without limit; past the bound new groups are simply not tracked.
pub struct ArgumentChurnDetector {
    variants: u32,
    calls_per_variant: u32,
    /// One lock for counts and warnings so `record` and `reset` are atomic.
    state: Mutex<ChurnState>,
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
            state: Mutex::default(),
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
        let argument = hash_of(arguments_fingerprint);
        let mut state = lock(&self.state);
        if state.warned.contains(&key) {
            return None;
        }
        if !state.groups.contains_key(&key) && state.groups.len() >= MAX_CHURN_GROUPS {
            return None;
        }
        let variants = state.groups.entry(key.clone()).or_default();
        if !variants.contains_key(&argument) && variants.len() >= MAX_CHURN_VARIANTS_PER_GROUP {
            return None;
        }
        let calls = variants.entry(argument).or_insert(0);
        *calls = calls.saturating_add(1);
        let qualifying = variants
            .values()
            .filter(|calls| **calls >= self.calls_per_variant)
            .count() as u32;
        if qualifying < self.variants {
            return None;
        }
        state.groups.remove(&key);
        if state.warned.len() < MAX_WARNED {
            state.warned.insert(key);
        }
        Some(format!(
            "`{tool}` has been called with {qualifying} different sets of arguments, each at least {} times, and every one returned the same result; changing the arguments is not changing what you learn. Try a different tool or approach.",
            self.calls_per_variant
        ))
    }

    /// Forgets every count and warning.
    pub fn reset(&self) {
        *lock(&self.state) = ChurnState::default();
    }
}

#[cfg(test)]
#[path = "loop_patterns_tests.rs"]
mod tests;
