//! One timeout/retry/budget policy for every subagent path.
//!
//! [`SubAgentPolicy`] and [`SubAgentBudget`] are defined beside the graph
//! `SubAgentNode` (`tinyagents-graph`, which this crate depends on, so the
//! types cannot live here without a cycle) and re-exported so a host names
//! them from either place. [`SubagentDriver`](super::SubagentDriver) and
//! [`SubAgentTool`](super::SubAgentTool) apply them like the graph node does:
//!
//! - **timeout** cancels the child and ends the run `Incomplete(Timeout)`;
//!   a timed-out attempt is never retried (it may have run for the whole
//!   window and run tools).
//! - **retry** applies only to a failure the retry policy deems retryable
//!   and never after the attempt ran tools, unless
//!   [`SubAgentPolicy::retry_after_tool_calls`] is set.
//! - **budget** call caps tighten the child's `RunConfig` (enforced during the
//!   run); token caps are checked against the reported usage afterwards;
//!   `max_cost` is carried but not enforced (see [`SubAgentBudget`]).

pub use tinyagents_graph::{SubAgentBudget, SubAgentPolicy};

use tinyagents_harness::error::TinyAgentsError;

/// Whether a failed attempt `attempt` (0-based) may be retried.
pub(crate) fn may_retry(
    policy: &SubAgentPolicy,
    attempt: usize,
    error: &TinyAgentsError,
    tools_ran: bool,
) -> bool {
    policy.retry.should_retry_error(attempt, error) && (!tools_ran || policy.retry_after_tool_calls)
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
