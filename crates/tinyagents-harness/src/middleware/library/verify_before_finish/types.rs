//! Public activity, trigger, and middleware state types.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fewest model calls that must remain after the answer for the check to run.
/// The check takes one; leaving at least two more keeps it clear of a host's
/// penultimate/final-call wrap-up and leaves room to act on what it finds.
pub const MIN_REMAINING_MODEL_CALLS: usize = 3;

/// Default for
/// [`with_min_remaining_wall_clock`](VerifyBeforeFinishMiddleware::with_min_remaining_wall_clock).
pub const DEFAULT_MIN_REMAINING_WALL_CLOCK: Duration = Duration::from_secs(120);

/// What a run did before its answer, as seen by a [`FinishCheckTrigger`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FinishActivity {
    /// Model responses in this run that requested at least one tool call.
    pub tool_rounds: usize,
    /// Names of every tool those rounds requested.
    pub tools_called: BTreeSet<String>,
}

impl FinishActivity {
    /// Whether the run requested `tool` at least once.
    pub fn called(&self, tool: &str) -> bool {
        self.tools_called.contains(tool)
    }
}

/// Decides whether a run's activity warrants the check.
pub type FinishCheckTrigger = Arc<dyn Fn(&FinishActivity) -> bool + Send + Sync>;

/// Per-run bookkeeping, keyed by [`crate::context::RunContext::instance_id`].
#[derive(Default)]
pub(super) struct RunState {
    pub(super) activity: FinishActivity,
    pub(super) fired: bool,
    pub(super) restore_deferred: bool,
    pub(super) lifecycle: std::sync::Weak<()>,
}

/// Asks a multi-step run, once, to check its final answer against the request
/// before that answer ends the run. See the module docs.
///
/// Generic over the application state and run-context payload: nothing here
/// reads either. The host supplies the check text and decides which runs get
/// it ([`with_min_tool_rounds`](Self::with_min_tool_rounds) or
/// [`with_trigger`](Self::with_trigger)).
pub struct VerifyBeforeFinishMiddleware {
    pub(super) check: String,
    pub(super) trigger: FinishCheckTrigger,
    pub(super) min_remaining_wall_clock: Duration,
    pub(super) wall_clock_limit: Option<Duration>,
    pub(super) runs: Mutex<HashMap<u64, RunState>>,
}
