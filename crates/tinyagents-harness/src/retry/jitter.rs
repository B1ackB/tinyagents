//! Minimal, dependency-free randomness for retry jitter.
//!
//! Jitter exists to break up thundering herds: without it every client that
//! failed at the same instant retries at the same instant. That needs *some*
//! randomness, but not cryptographic randomness and not a statistically
//! rigorous generator — a couple of milliseconds of spread is the whole
//! requirement.
//!
//! The crate deliberately does not take a `rand` / `fastrand` dependency for
//! this, so this module ships a ~30-line `xorshift64*` generator behind a
//! thread-local. It is:
//!
//! - **Never used for anything security-relevant.** Backoff spread only.
//! - **Seeded per thread** from the wall clock plus a process-global counter, so
//!   two threads (and two processes started in the same nanosecond) do not walk
//!   the same sequence.
//! - **Bypassable for tests.** Every consumer routes through
//!   [`RetryPolicy::backoff_for_attempt_with`][crate::retry::RetryPolicy::backoff_for_attempt_with],
//!   which takes an explicit `rand01`, so no test ever has to observe this
//!   module's output.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-global seed disambiguator so two threads seeded within the same
/// clock tick still diverge.
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Per-thread generator state. Never zero (xorshift is degenerate at zero).
    static STATE: Cell<u64> = Cell::new(seed());
}

/// Builds a non-zero seed from the wall clock and a global counter.
fn seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let bump = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Golden-ratio odd constant keeps the mix well-distributed for small bumps.
    let mixed = nanos ^ bump.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if mixed == 0 {
        0xDEAD_BEEF_CAFE_F00D
    } else {
        mixed
    }
}

/// Advances the thread-local generator and returns the raw 64-bit output.
fn next_u64() -> u64 {
    STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        // xorshift64* output scrambler.
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    })
}

/// Returns a pseudo-random `f64` uniformly distributed over `[0, 1)`.
///
/// This is the value production retry paths feed to
/// [`RetryPolicy::backoff_for_attempt_with`][crate::retry::RetryPolicy::backoff_for_attempt_with].
pub(crate) fn rand01() -> f64 {
    // 53 bits is the full mantissa of an f64, so this covers [0, 1) evenly.
    ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
#[path = "jitter_test_tests.rs"]
mod test;
