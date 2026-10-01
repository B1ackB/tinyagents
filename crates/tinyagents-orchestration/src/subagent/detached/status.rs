use std::time::Duration;

use tinyagents_graph::orchestration::{
    DetachedTaskRegistry, DetachedTaskRegistryError, DetachedTaskWaitOutcome,
};
use tinyagents_harness::ids::TaskId;

/// Terminal/transient state of a detached subagent, published by the
/// spawner's background task and observed by waiters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetachedSubagentStatus {
    /// Still executing its inner tool-call loop.
    Running,
    /// Finished normally with a final response.
    Completed {
        /// Final response text.
        output: String,
        /// Loop iterations the run used (0 when recovered from a durable record).
        iterations: usize,
    },
    /// Paused on a clarification request; resumed by a follow-up.
    AwaitingUser {
        /// The question the child is waiting on.
        question: String,
    },
    /// The run errored out.
    Failed {
        /// Failure description.
        error: String,
    },
}

impl DetachedSubagentStatus {
    /// Everything except [`Self::Running`] is terminal (an awaiting run is
    /// paused, but it will not progress on its own).
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Stable wire label: `running` / `completed` / `awaiting_user` / `failed`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::AwaitingUser { .. } => "awaiting_user",
            Self::Failed { .. } => "failed",
        }
    }

    /// The status reported when a run's sender was dropped without a result.
    pub fn ended_without_result() -> Self {
        Self::Failed {
            error: "sub-agent task ended without reporting a result".to_string(),
        }
    }

    /// How a run had already ended, if it had. `AwaitingUser` is paused, not
    /// finished, so it yields `None`.
    pub fn finished_outcome(&self) -> Option<FinishedOutcome> {
        match self {
            Self::Completed { .. } => Some(FinishedOutcome::Completed),
            Self::Failed { .. } => Some(FinishedOutcome::Failed),
            Self::Running | Self::AwaitingUser { .. } => None,
        }
    }
}

/// The terminal outcome of a run that finished before its cancel arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishedOutcome {
    /// The run completed.
    Completed,
    /// The run failed.
    Failed,
}

impl FinishedOutcome {
    /// Wire name (`completed` / `failed`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// Why a wait or resolution could not be set up.
#[derive(Debug, PartialEq, Eq)]
pub enum WaitError {
    /// No such subagent.
    Unknown,
    /// The caller does not own it.
    NotOwned,
}

/// Map a registry error onto [`WaitError`] (anything but ownership is unknown).
pub fn wait_error_from_registry(error: DetachedTaskRegistryError) -> WaitError {
    match error {
        DetachedTaskRegistryError::NotOwned => WaitError::NotOwned,
        _ => WaitError::Unknown,
    }
}

/// Result of waiting on a subagent.
#[derive(Debug)]
pub enum WaitOutcome {
    /// Reached a terminal status (the registry entry is pruned).
    Terminal(DetachedSubagentStatus),
    /// The timeout elapsed first; the entry is intact so the caller can wait
    /// again. Carries the latest non-terminal snapshot.
    TimedOut(DetachedSubagentStatus),
}

/// Block until `task_id` reaches a terminal status or `timeout` elapses.
///
/// A closed status channel (aborted/panicked task) surfaces as a
/// [`DetachedSubagentStatus::ended_without_result`] failure instead of hanging.
pub async fn wait_detached<M>(
    registry: &DetachedTaskRegistry<M, DetachedSubagentStatus>,
    task_id: &str,
    owner: &str,
    timeout: Duration,
) -> Result<WaitOutcome, WaitError>
where
    M: Clone + Send + Sync + 'static,
{
    match registry.wait(&TaskId::new(task_id), owner, timeout).await {
        Ok(DetachedTaskWaitOutcome::Terminal(status)) => Ok(WaitOutcome::Terminal(status)),
        Ok(DetachedTaskWaitOutcome::TimedOut(status)) => Ok(WaitOutcome::TimedOut(status)),
        Err(DetachedTaskRegistryError::StatusChannelClosed) => Ok(WaitOutcome::Terminal(
            DetachedSubagentStatus::ended_without_result(),
        )),
        Err(error) => Err(wait_error_from_registry(error)),
    }
}
