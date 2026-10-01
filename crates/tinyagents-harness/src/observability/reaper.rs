//! Startup reconciliation for orphaned runs.
//!
//! When a host process exits (a crash, a restart, a deploy) any run still in a
//! non-terminal state (`Pending` / `Running` / `Interrupted`) in the durable
//! status store is *orphaned*: no executor will ever advance it, yet it lingers
//! in the "active runs" listing. [`reap_orphaned_runs`] sweeps every
//! still-active run to the terminal `Cancelled` state with an explanatory error
//! so the listing reflects reality.

use std::time::SystemTime;

use super::HarnessStatusStore;

/// Error recorded on a run reaped by the startup sweep. Stable and
/// grep-friendly so an operator (or a test) can tell a reaped run from a
/// genuinely failed one.
pub const ORPHAN_REAP_REASON: &str = "run orphaned: core restarted while the run was in flight";

/// Reap every run left non-terminal by a previous process.
///
/// Lists the still-active runs in `store` and moves each to `Cancelled`.
/// `Cancelled` (not `Failed`) is deliberate: a restart is not a run failure and
/// `Failed` would render as a red error row. Best-effort: a per-run persistence
/// failure is logged and does not abort the sweep, and a failure to list logs
/// and yields `0`. Returns the number of runs reaped.
pub async fn reap_orphaned_runs(store: &dyn HarnessStatusStore) -> usize {
    // One id for the whole sweep so every line can be tied to the same boot.
    // Derived from the clock rather than a uuid so it stays ordered in a tail.
    let sweep_id = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    tracing::debug!("[agent] startup run sweep entry sweep_id={sweep_id}");

    let active = match store.list_active().await {
        Ok(runs) => runs,
        Err(err) => {
            tracing::warn!(
                "[agent] startup run sweep exit sweep_id={sweep_id} branch=list-failed: {err}"
            );
            return 0;
        }
    };

    if active.is_empty() {
        tracing::debug!("[agent] startup run sweep exit sweep_id={sweep_id} branch=none-active");
        return 0;
    }
    tracing::debug!(
        "[agent] startup run sweep sweep_id={sweep_id} active={} to reap",
        active.len()
    );

    let mut reaped = 0usize;
    for mut status in active {
        let run_id = status.run_id.as_str().to_string();
        // Read the source state before the mutation: `list_active` returns
        // whatever non-terminal state a run was left in.
        let from_status = format!("{:?}", status.status);
        status.mark_cancelled(ORPHAN_REAP_REASON);
        let to_status = format!("{:?}", status.status);
        tracing::debug!(
            "[agent] startup run sweep sweep_id={sweep_id} run_id={run_id} \
             transition={from_status}->{to_status} phase=Done persisting"
        );
        match store.put_status(status).await {
            Ok(()) => {
                reaped += 1;
                tracing::debug!(
                    "[agent] startup run sweep sweep_id={sweep_id} run_id={run_id} \
                     transition={from_status}->{to_status} persisted"
                );
            }
            Err(err) => {
                tracing::warn!(
                    "[agent] startup run sweep sweep_id={sweep_id} run_id={run_id} \
                     transition={from_status}->{to_status} failed: {err}"
                );
            }
        }
    }
    // One operator-visible line, and only when the sweep did something.
    if reaped > 0 {
        tracing::info!(
            "[agent] startup run sweep sweep_id={sweep_id} reaped {reaped} orphaned run(s)"
        );
    } else {
        tracing::debug!("[agent] startup run sweep exit sweep_id={sweep_id} branch=nothing-reaped");
    }
    reaped
}

#[cfg(test)]
#[path = "reaper_test.rs"]
mod tests;
