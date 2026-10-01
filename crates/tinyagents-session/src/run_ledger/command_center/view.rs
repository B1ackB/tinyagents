//! Read-only command-center projection over the durable run ledger.
//!
//! [`list_agent_work`] fetches recent background agent runs and projects them
//! into a [`CommandCenterView`] grouped by normalized [`AgentWorkBucket`]. The
//! grouping logic ([`build_view`]) is unit-testable without a database, while
//! [`list_agent_work`] owns the one ledger read. Agent display names come from
//! a host-supplied closure so this crate needs no agent registry.

use std::path::Path;

use crate::Result;
use crate::run_ledger::{AgentRun, AgentRunListRequest, AgentRunStatus, list_agent_runs};

use super::types::{AgentWorkBucket, AgentWorkRow, CommandCenterGroup, CommandCenterView};

/// Default number of recent runs scanned for the command center.
const DEFAULT_LIMIT: u32 = 200;
/// Hard ceiling, mirroring the ledger's own `list_agent_runs` cap.
const MAX_LIMIT: u32 = 500;

/// Map a fine-grained ledger status to its command-center bucket.
///
/// Exhaustive on [`AgentRunStatus`] so a new ledger status variant fails to
/// compile here until its bucket is decided.
pub fn bucket_for(status: AgentRunStatus) -> AgentWorkBucket {
    match status {
        AgentRunStatus::AwaitingUser => AgentWorkBucket::NeedsInput,
        AgentRunStatus::Pending | AgentRunStatus::Running | AgentRunStatus::Paused => {
            AgentWorkBucket::Working
        }
        AgentRunStatus::Completed => AgentWorkBucket::Completed,
        AgentRunStatus::Failed => AgentWorkBucket::Failed,
        AgentRunStatus::Cancelled | AgentRunStatus::Interrupted => AgentWorkBucket::Stopped,
    }
}

/// List recent background agent work, grouped by command-center bucket.
///
/// Reads at most `limit` (default 200, capped 500) most-recently-updated runs
/// across every parent thread and projects them. Read-only: no ledger writes.
pub fn list_agent_work(
    workspace_dir: &Path,
    limit: Option<u32>,
    display_name: &dyn Fn(&str) -> Option<String>,
) -> Result<CommandCenterView> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
    tracing::debug!(
        target: "command_center",
        "[command_center] list_agent_work.entry limit={limit}"
    );
    let request = AgentRunListRequest {
        status: None,
        kind: None,
        parent_run_id: None,
        parent_thread_id: None,
        limit: Some(limit),
        offset: None,
    };
    let response = list_agent_runs(workspace_dir, &request)?;
    let view = build_view(response.runs, display_name);
    tracing::debug!(
        target: "command_center",
        "[command_center] list_agent_work.done total={}",
        view.total
    );
    Ok(view)
}

/// Project + group a set of ledger runs into the command-center view.
///
/// Pure given `display_name` (maps an agent id to a human-friendly name when
/// the host's registry knows it): input order is preserved within each bucket, so callers that pass
/// runs already ordered most-recently-updated-first (as `list_agent_runs`
/// does) get recent-first rows per group. All five buckets are always present.
pub fn build_view(
    runs: Vec<AgentRun>,
    display_name: &dyn Fn(&str) -> Option<String>,
) -> CommandCenterView {
    let rows: Vec<AgentWorkRow> = runs
        .into_iter()
        .map(|run| project_row(run, display_name))
        .collect();
    let total = rows.len();
    let groups = AgentWorkBucket::ALL
        .iter()
        .map(|&bucket| {
            let bucket_rows: Vec<AgentWorkRow> = rows
                .iter()
                .filter(|r| r.bucket == bucket)
                .cloned()
                .collect();
            CommandCenterGroup {
                bucket,
                count: bucket_rows.len(),
                rows: bucket_rows,
            }
        })
        .collect();
    CommandCenterView { groups, total }
}

/// Project one ledger run into a lean command-center row.
///
/// `pub(super)` so the control verbs ([`super::control`]) can re-project a run
/// after a durable status transition without duplicating the mapping.
pub(super) fn project_row(
    run: AgentRun,
    display_name: &dyn Fn(&str) -> Option<String>,
) -> AgentWorkRow {
    let display_name = run.agent_id.as_deref().and_then(display_name);
    let telemetry = run.telemetry;
    AgentWorkRow {
        run_id: run.id,
        kind: run.kind.as_str().to_string(),
        agent_id: run.agent_id,
        display_name,
        bucket: bucket_for(run.status),
        status: run.status.as_str().to_string(),
        parent_thread_id: run.parent_thread_id,
        worker_thread_id: run.worker_thread_id,
        summary: run.summary,
        error: run.error,
        started_at: run.started_at.to_rfc3339(),
        updated_at: run.updated_at.to_rfc3339(),
        elapsed_ms: telemetry.as_ref().and_then(|t| t.elapsed_ms),
        input_tokens: telemetry.as_ref().map(|t| t.input_tokens).unwrap_or(0),
        output_tokens: telemetry.as_ref().map(|t| t.output_tokens).unwrap_or(0),
        cost_usd: telemetry.as_ref().map(|t| t.cost_usd).unwrap_or(0.0),
        tool_count: telemetry.as_ref().map(|t| t.tool_count).unwrap_or(0),
    }
}
