//! Read-only late-attach replay over the durable journal and status seams.
//!
//! Every function is a *reader*: it never writes or mutates. A host resolves its
//! own storage layout into a [`HarnessEventJournal`] / [`HarnessStatusStore`]
//! and delegates here.

use super::{AgentObservation, HarnessEventJournal, HarnessStatusStore, file_status::is_active};
use crate::error::Result;
use crate::events::HarnessRunStatus;

/// Default page size for [`read_run_events_page`] when the caller omits a limit.
pub const DEFAULT_EVENTS_LIMIT: u64 = 200;

/// Hard cap on a single replay page so one call can never fan a whole run into
/// a single response.
pub const MAX_EVENTS_LIMIT: u64 = 1000;

/// One page of a run's durable event stream.
///
/// `events` are [`AgentObservation`]s in ascending `offset` order.
/// `next_offset` is the offset to pass back to fetch the following page, or
/// `None` once the stream is drained.
pub struct RunEventsPage {
    /// Observations in this page.
    pub events: Vec<AgentObservation>,
    /// Cursor for the next page, or `None` when drained.
    pub next_offset: Option<u64>,
}

/// Paged late-attach replay reader over a durable journal.
///
/// Returns up to `limit` (clamped into `[1, MAX_EVENTS_LIMIT]`) observations for
/// `run_id` whose stream offset is `>= offset`, plus a `next_offset` cursor.
/// Best-effort: an unknown run yields an empty page, not an error.
pub async fn read_run_events_page(
    journal: &dyn HarnessEventJournal,
    run_id: &str,
    offset: u64,
    limit: u64,
) -> Result<RunEventsPage> {
    let effective_limit = limit.clamp(1, MAX_EVENTS_LIMIT);
    tracing::debug!(
        "[agent] replay read_run_events_page run_id={run_id} offset={offset} \
         limit={limit} effective_limit={effective_limit}"
    );

    let mut events = journal.read_from(run_id, offset).await?;

    // Reading the whole tail lets us detect a further page without a second
    // store round-trip.
    let has_more = events.len() as u64 > effective_limit;
    if has_more {
        events.truncate(effective_limit as usize);
    }
    // Offsets are monotonic within a run: the cursor is one past the last
    // returned offset.
    let next_offset = if has_more {
        events.last().map(|obs| obs.offset + 1)
    } else {
        None
    };

    tracing::debug!(
        "[agent] replay read_run_events_page run_id={run_id} returned={} next_offset={:?}",
        events.len(),
        next_offset
    );
    Ok(RunEventsPage {
        events,
        next_offset,
    })
}

/// Latest durable status for `run_id`, or `None` when the run is unknown.
pub async fn read_run_status(
    store: &dyn HarnessStatusStore,
    run_id: &str,
) -> Result<Option<HarnessRunStatus>> {
    tracing::debug!("[agent] replay read_run_status run_id={run_id}");
    let status = store.get_status(run_id).await?;
    tracing::debug!(
        "[agent] replay read_run_status run_id={run_id} found={}",
        status.is_some()
    );
    Ok(status)
}

/// Active runs, optionally filtered by `thread_id` and/or `root_run_id`.
///
/// The thread/root store queries return *all* runs (active and terminal), so
/// the liveness predicate is always applied on top. When both filters are
/// supplied the base query uses `thread_id` and the result is further
/// restricted to `root_run_id`.
pub async fn list_active_runs(
    store: &dyn HarnessStatusStore,
    thread_id: Option<&str>,
    root_run_id: Option<&str>,
) -> Result<Vec<HarnessRunStatus>> {
    tracing::debug!(
        "[agent] replay list_active_runs thread_id={:?} root_run_id={:?}",
        thread_id,
        root_run_id
    );
    let base = match (thread_id, root_run_id) {
        (Some(thread), _) => store.list_by_thread(thread).await?,
        (None, Some(root)) => store.list_by_root(root).await?,
        (None, None) => store.list_active().await?,
    };

    let mut runs: Vec<HarnessRunStatus> = base.into_iter().filter(is_active).collect();
    if thread_id.is_some() {
        if let Some(root) = root_run_id {
            runs.retain(|s| s.root_run_id.as_str() == root);
        }
    }

    tracing::debug!("[agent] replay list_active_runs returned={}", runs.len());
    Ok(runs)
}

#[cfg(test)]
#[path = "replay_test.rs"]
mod tests;
