//! The early-exit hook: pause a run when a designated tool succeeds.

use std::sync::{Arc, Mutex, PoisonError};

use crate::steering::{SteeringCommand, SteeringHandle};

/// A captured early-exit: a sub-agent invoked an early-exit tool (e.g.
/// `ask_user_clarification`), so the loop should pause and surface `question`
/// to the user.
#[derive(Debug, Clone)]
pub struct EarlyExit {
    /// Name of the tool that requested the early exit.
    pub tool: String,
    /// The question (the tool's output) to surface to the user.
    pub question: String,
}

/// Shared early-exit hook handed to the adapters for the early-exit tool names.
/// On a successful call to one of those tools it records the [`EarlyExit`] and
/// sends a [`SteeringCommand::Pause`] so the harness loop short-circuits at the
/// next checkpoint (before the next model call) — so a sub-agent that asks the
/// user a question stops the loop instead of guessing.
#[derive(Clone)]
pub struct EarlyExitHook {
    handle: SteeringHandle,
    slot: Arc<Mutex<Option<EarlyExit>>>,
}

impl std::fmt::Debug for EarlyExitHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EarlyExitHook").finish_non_exhaustive()
    }
}

impl EarlyExitHook {
    /// Build a hook that pauses `handle` and records into a fresh slot.
    #[must_use]
    pub fn new(handle: SteeringHandle) -> Self {
        Self {
            handle,
            slot: Arc::new(Mutex::new(None)),
        }
    }

    /// The captured early-exit, if one fired during the run.
    #[must_use]
    pub fn take(&self) -> Option<EarlyExit> {
        // Recover a poisoned slot rather than panic: a panic while some other
        // tool held this lock must not swallow the early-exit.
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Record an early-exit and request a cooperative pause. Only the first
    /// early-exit in a run is kept (halt on first).
    pub(crate) fn trigger(&self, tool: &str, question: String) {
        {
            // `into_inner` keeps early-exit recording working even if the slot
            // mutex was poisoned by an unrelated panic.
            let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(EarlyExit {
                    tool: tool.to_string(),
                    question,
                });
            }
        }
        tracing::info!(tool, "[tinyagents] early-exit tool requesting pause");
        self.handle.send(SteeringCommand::Pause);
    }
}
