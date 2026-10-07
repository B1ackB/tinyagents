//! Small helpers shared by the trackers and detectors in this module.

use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Hashes one string. The trackers store hashes rather than the content so a
/// whole run's accounting stays cheap to hold.
pub(super) fn hash_of(value: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Hashes a `(call signature, outcome identity)` pair.
pub(super) fn hash_pair(call_signature: &str, outcome_identity: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (call_signature, outcome_identity).hash(&mut hasher);
    hasher.finish()
}

/// Locks `mutex`, carrying on with the state a panicking holder left behind
/// instead of panicking too: a loop guard must never take the run down.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
