use tinyagents_graph::orchestration::{DetachedTaskRegistry, OrchestrationTaskRecord};
use tinyagents_harness::ids::TaskId;

use super::ledger::{record_agent_id, record_parent_session, record_subagent_session_id};
use super::status::{DetachedSubagentStatus, WaitError, wait_error_from_registry};

/// What a host's registry metadata must expose for roster and resolution.
pub trait SubagentIdentity {
    /// Worker type (not unique across parallel workers).
    fn agent_id(&self) -> &str;
    /// Durable, stable per-worker reference, if any.
    fn subagent_session_id(&self) -> Option<&str>;
}

/// Compact, read-only view of one registered subagent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentSnapshot {
    /// Worker type.
    pub agent_id: String,
    /// Durable per-worker reference.
    pub subagent_session_id: Option<String>,
    /// Transient registry key.
    pub task_id: String,
    /// Stable status label (see [`DetachedSubagentStatus::label`]).
    pub status: &'static str,
}

/// A subagent addressed for resumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentResumeRef {
    /// Transient task id.
    pub task_id: String,
    /// Worker type.
    pub agent_id: String,
    /// Durable per-worker reference.
    pub subagent_session_id: Option<String>,
}

/// Snapshot the subagents registered under `owner`, with live status, ordered
/// by `agent_id` then `task_id` so a rendered roster is stable across turns.
pub fn snapshot_for_owner<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    owner: &str,
) -> Vec<SubagentSnapshot>
where
    M: SubagentIdentity + Clone + Send + Sync + 'static,
{
    let mut out: Vec<SubagentSnapshot> = registry
        .snapshots(Some(owner))
        .expect("detached task registry lock poisoned")
        .into_iter()
        .map(|entry| SubagentSnapshot {
            agent_id: entry.metadata.agent_id().to_string(),
            subagent_session_id: entry.metadata.subagent_session_id().map(str::to_string),
            task_id: entry.task_id.as_str().to_string(),
            status: entry.status.label(),
        })
        .collect();
    out.sort_by(|a, b| {
        a.agent_id
            .cmp(&b.agent_id)
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
    out
}

/// Resolve a durable session id to the live task id, enforcing ownership. A
/// non-terminal entry wins over a terminal one.
pub fn task_id_for_session<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    subagent_session_id: &str,
    owner: &str,
) -> Result<String, WaitError>
where
    M: SubagentIdentity + Clone + Send + Sync + 'static,
{
    let mut saw_unowned = false;
    let mut owned_terminal: Option<String> = None;
    for snapshot in registry
        .snapshots(None)
        .expect("detached task registry lock poisoned")
        .into_iter()
        .filter(|snapshot| snapshot.metadata.subagent_session_id() == Some(subagent_session_id))
    {
        if snapshot.owner_id != owner {
            saw_unowned = true;
            continue;
        }
        let task_id = snapshot.task_id.as_str().to_string();
        if !snapshot.status.is_terminal() {
            return Ok(task_id);
        }
        owned_terminal.get_or_insert(task_id);
    }
    if let Some(task_id) = owned_terminal {
        return Ok(task_id);
    }
    if saw_unowned {
        return Err(WaitError::NotOwned);
    }
    Err(WaitError::Unknown)
}

/// Resolve a session id against durable `records` (most recently updated
/// first), enforcing parent ownership.
pub fn task_id_for_session_in_records(
    records: Vec<OrchestrationTaskRecord>,
    subagent_session_id: &str,
    parent_session: &str,
) -> Result<String, WaitError> {
    let mut saw_unowned = false;
    let mut matches: Vec<OrchestrationTaskRecord> = records
        .into_iter()
        .filter(|record| record_subagent_session_id(record) == Some(subagent_session_id))
        .collect();
    matches.sort_by_key(|item| std::cmp::Reverse(item.updated_at));
    for record in matches {
        if record_parent_session(&record) != Some(parent_session) {
            saw_unowned = true;
            continue;
        }
        return Ok(record.spec.task_id.as_str().to_string());
    }
    if saw_unowned {
        return Err(WaitError::NotOwned);
    }
    Err(WaitError::Unknown)
}

/// Resume reference for a live task, enforcing ownership.
pub fn resume_ref_for_task<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    task_id: &str,
    owner: &str,
) -> Result<SubagentResumeRef, WaitError>
where
    M: SubagentIdentity + Clone + Send + Sync + 'static,
{
    let snapshot = registry
        .snapshot(&TaskId::new(task_id), owner)
        .map_err(wait_error_from_registry)?;
    Ok(SubagentResumeRef {
        task_id: task_id.to_string(),
        agent_id: snapshot.metadata.agent_id().to_string(),
        subagent_session_id: snapshot.metadata.subagent_session_id().map(str::to_string),
    })
}

/// Resume reference recovered from a durable record.
pub fn resume_ref_from_record(
    task_id: &str,
    record: &OrchestrationTaskRecord,
) -> SubagentResumeRef {
    SubagentResumeRef {
        task_id: task_id.to_string(),
        agent_id: record_agent_id(record),
        subagent_session_id: record_subagent_session_id(record).map(ToOwned::to_owned),
    }
}
