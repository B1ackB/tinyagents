//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! [`SubAgent`], [`SubAgentTool`], and [`SubAgentSession`] compose harness
//! agents as direct child runs. The `Subagent*` lifecycle traits and driver
//! separately coordinate resume loading, preparation, execution, and one
//! mutually exclusive persistence action. Hosts still resolve policy, agent
//! definitions, prompts, tool allowlists, and persistence implementations.
//!
//! Dependency direction remains `orchestration -> {harness, runtime}`. Lower
//! TinyAgents layers must not depend on this module.

mod detached;
mod driver;
mod executor;
mod invocation;
mod persistence;
mod planner;
mod types;

pub use detached::{
    DETACHED_LEDGER_TIMEOUT_MS, DetachedSubagentStatus, FinishedOutcome, SpawnedSubagent,
    SteerAccess, SteerError, SteerReceipt, SteerRoute, SubagentIdentity, SubagentResumeRef,
    SubagentSnapshot, WaitError, WaitOutcome, cancel_for_thread, distinct_parent_threads,
    list_subagent_records, orphaned_subagent_reason, queue_lane_name, record_agent_id,
    record_cancelled, record_parent_session, record_spawned, record_status,
    record_subagent_session_id, record_to_wait_outcome, resume_ref_for_task,
    resume_ref_from_record, snapshot_for_owner, spawn_status_watcher, steer_detached,
    steer_detached_with_request_id, steering_command_for_lane, subagent_record_for_task,
    task_id_for_session, task_id_for_session_in_records, task_status_label, wait_detached,
    wait_error_from_registry,
};
pub use driver::{SubagentCapabilities, SubagentDriver};
pub use executor::SubagentExecutor;
pub use invocation::{
    ChildDataPolicy, SubAgent, SubAgentJob, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry,
    SubAgentJobStatus, SubAgentJobsTool, SubAgentMessageTool, SubAgentSession, SubAgentTool,
    register_subagent_job_tools,
};
pub use persistence::SubagentPersistence;
pub use planner::SubagentPlanner;
pub use types::{
    ArtifactReference, PersistedSubagentPause, PreparedSubagent, SubagentError, SubagentExecution,
    SubagentIncomplete, SubagentOutcome, SubagentPause, SubagentPausePersistenceDisposition,
    SubagentPersistenceDisposition, SubagentRequest, SubagentRequestParts, SubagentResume,
    SubagentRunResult, SubagentStatus, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
