//! The per-session turn lock: what keeps an out-of-band transcript append
//! from landing in the middle of a live turn.
//!
//! A live turn reads the head generation when it starts (resume), runs the
//! model, and then appends its delta against what it read
//! ([`TranscriptHistory::append_turn`](super::TranscriptHistory::append_turn)
//! validates that baseline under the file lock). A line written by anyone else
//! in between makes that baseline stale, and the turn's persist fails rather
//! than hide the foreign line behind a compaction. The file lock alone is held
//! only for one write, so it cannot prevent this; the turn lock is held for the
//! whole read → run → persist span.
//!
//! The lock is in-process and async (a tokio mutex), keyed by the locator's
//! [`destination_key`](TranscriptLocator::destination_key) and the session's
//! generation-0 stem, so every generation of one conversation and every
//! locator rebuilt over the same workspace share one lock. Entries are weak
//! and swept, so the registry does not grow with every session ever touched.

use super::history::TranscriptLocator;
use super::session::{SessionRef, session_stem};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Proof that the holder owns `session`'s turn lock. Released on drop.
#[must_use = "the turn lock is released as soon as the guard is dropped"]
#[derive(Debug)]
pub struct SessionTurnGuard {
    _guard: OwnedMutexGuard<()>,
}

/// Waits for and takes the turn lock for `session` under `locator`.
///
/// Returns `None` when `locator` cannot name its destination
/// ([`TranscriptLocator::destination_key`] is `None`): such a locator only
/// ever matches itself, so no other caller could find the same lock.
///
/// Not reentrant: awaiting it twice for one session from the same task (for
/// example, a turn that awaits a background append into its own session)
/// deadlocks.
pub async fn lock_session_turn(
    locator: &dyn TranscriptLocator,
    session: &SessionRef,
) -> Option<SessionTurnGuard> {
    let destination = locator.destination_key()?;
    let stem = session_stem(&session.first_generation());
    let lock = registry_lock(destination, stem.clone());
    tracing::debug!(session = %stem, "[transcript-turn-lock] acquiring");
    let guard = lock.lock_owned().await;
    tracing::debug!(session = %stem, "[transcript-turn-lock] acquired");
    Some(SessionTurnGuard { _guard: guard })
}

#[doc(hidden)]
pub(crate) async fn lock_session_turn_with_notification(
    locator: &dyn TranscriptLocator,
    session: &SessionRef,
    attempted: Option<&tokio::sync::Notify>,
) -> Option<SessionTurnGuard> {
    let destination = locator.destination_key()?;
    let stem = session_stem(&session.first_generation());
    let lock = registry_lock(destination, stem.clone());
    tracing::debug!(session = %stem, "[transcript-turn-lock] acquiring");
    let guard = NotifyOnPoll::new(lock.lock_owned(), attempted).await;
    tracing::debug!(session = %stem, "[transcript-turn-lock] acquired");
    Some(SessionTurnGuard { _guard: guard })
}

struct NotifyOnPoll<'a, F> {
    future: Pin<Box<F>>,
    notify: Option<&'a tokio::sync::Notify>,
    notified: bool,
}

impl<'a, F> NotifyOnPoll<'a, F> {
    fn new(future: F, notify: Option<&'a tokio::sync::Notify>) -> Self {
        Self {
            future: Box::pin(future),
            notify,
            notified: false,
        }
    }
}

impl<F: Future> Future for NotifyOnPoll<'_, F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Poll the wrapped future first: a waiter woken by the notification
        // then knows the lock attempt has already been made (and parked, if it
        // is held), instead of racing a not-yet-polled acquisition.
        let result = self.future.as_mut().poll(cx);
        if !self.notified {
            if let Some(notify) = self.notify {
                notify.notify_one();
            }
            self.notified = true;
        }
        result
    }
}

/// The shared mutex for `(destination, stem)`, created on first use.
fn registry_lock(destination: String, stem: String) -> Arc<AsyncMutex<()>> {
    type Registry = Mutex<HashMap<(String, String), Weak<AsyncMutex<()>>>>;
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    let mut locks = REGISTRY
        .get_or_init(Registry::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let key = (destination, stem);
    if let Some(existing) = locks.get(&key).and_then(Weak::upgrade) {
        return existing;
    }
    locks.retain(|_, weak| weak.strong_count() > 0);
    let fresh = Arc::new(AsyncMutex::new(()));
    locks.insert(key, Arc::downgrade(&fresh));
    fresh
}

#[cfg(test)]
#[path = "turn_lock_tests.rs"]
mod tests;
