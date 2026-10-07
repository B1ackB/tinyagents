//! Post-compaction loop guard.
//!
//! Context compaction can erase the evidence that a run was looping: the model
//! wakes up with a summary, repeats the call it was stuck on, gets the same
//! result and is back in the loop, with the repeat ledger cleared. This guard
//! remembers the last few `(call, result)` pairs recorded right before a
//! compaction. For a short window of calls after it, a call that repeats one
//! of those pairs is flagged so the host can escalate straight to blocking
//! instead of letting the ledger count up from zero again.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::util::{hash_pair, lock};

/// Tool calls watched after a compaction, and recent calls remembered before it.
pub const DEFAULT_POST_COMPACTION_WINDOW: u32 = 3;

#[derive(Default)]
struct GuardState {
    /// Most recent `(call, result)` hashes, oldest first, capped at the window.
    tail: VecDeque<u64>,
    /// The tail captured at compaction and the calls still to watch.
    armed: Option<(Vec<u64>, u32)>,
}

/// Flags calls that repeat the pre-compaction tail. See the module docs.
pub struct PostCompactionGuard {
    window: u32,
    state: Mutex<GuardState>,
}

impl Default for PostCompactionGuard {
    fn default() -> Self {
        Self::new(DEFAULT_POST_COMPACTION_WINDOW)
    }
}

impl PostCompactionGuard {
    /// `window` is both how many calls before a compaction are remembered and
    /// how many calls after it are watched. `0` disables the guard.
    pub fn new(window: u32) -> Self {
        Self {
            window,
            state: Mutex::default(),
        }
    }

    /// Records one successful call. Returns `true` when the guard is armed and
    /// this call repeats a pair from before the compaction (and disarms, so one
    /// compaction flags at most one repeat).
    pub fn record(&self, call_signature: &str, outcome_identity: &str) -> bool {
        if self.window == 0 {
            return false;
        }
        let pair = hash_pair(call_signature, outcome_identity);
        let mut state = lock(&self.state);
        if let Some((tail, remaining)) = state.armed.as_mut() {
            let repeated = tail.contains(&pair);
            *remaining -= 1;
            if repeated || *remaining == 0 {
                state.armed = None;
            }
            return repeated;
        }
        if state.tail.len() == self.window as usize {
            state.tail.pop_front();
        }
        state.tail.push_back(pair);
        false
    }

    /// A compaction just removed results from context: watch the next
    /// `window` calls against the tail recorded so far. With nothing recorded
    /// since the previous compaction, an already armed guard is left as is.
    pub fn arm(&self) {
        if self.window == 0 {
            return;
        }
        let mut state = lock(&self.state);
        if state.tail.is_empty() {
            return;
        }
        let tail = state.tail.drain(..).collect();
        state.armed = Some((tail, self.window));
    }

    /// Forgets the tail and disarms.
    pub fn reset(&self) {
        *lock(&self.state) = GuardState::default();
    }
}

#[cfg(test)]
#[path = "post_compaction_tests.rs"]
mod tests;
