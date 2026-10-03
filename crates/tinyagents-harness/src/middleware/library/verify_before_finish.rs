//! [`VerifyBeforeFinishMiddleware`]: before a multi-step run's first final
//! answer stands, ask the model once to check it against the request.
//!
//! # Why
//!
//! A run ends the moment the model returns text with no tool calls, i.e. when
//! the model *believes* it is done. On long tasks that belief is often built on
//! checks that cannot fail: a test derived from the implementation rather than
//! the spec, a benchmark measuring something other than what is graded, a
//! filter stated in the request that nothing ever re-read. One extra model call
//! that re-reads the request against the finished work is the cheapest lever
//! the harness has on that failure mode. This is a hypothesis about model
//! behaviour, so the middleware is opt-in and the trigger is the host's call.
//!
//! # How
//!
//! `after_model` sees the response before the loop decides it is final. When
//! it is a final answer (text, no tool calls, not truncated, not already
//! continued), the run's activity meets the trigger, the budget has room and
//! the check has not run yet in this run, the middleware sets
//! [`ModelResponse::continue_turn`] to the check message. The loop then keeps
//! the draft answer on the transcript, appends the check as the next **user**
//! turn (tail content, never a mid-conversation system message, so the cached
//! prefix is untouched) and asks for another reply. That reply is free to call
//! tools and fix what the check found; the next tool-less answer ends the run,
//! because the check fires at most once per run.
//!
//! The check rides `continue_turn` rather than a `JumpTo(Model)` plus a
//! `before_model` injection so the message is part of the transcript: a
//! follow-up call that fixes something still sees the instruction it is acting
//! on, and every later request keeps the same prefix.
//!
//! # Budget
//!
//! Skipped when two or fewer model calls remain after the answer, so it never
//! competes with a final-call wrap-up for the last calls, and when a configured
//! wall-clock deadline is closer than
//! [`with_min_remaining_wall_clock`](VerifyBeforeFinishMiddleware::with_min_remaining_wall_clock)
//! (the run context's deadline, and the policy cap a host declares through
//! [`with_wall_clock_limit`](VerifyBeforeFinishMiddleware::with_wall_clock_limit)).

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result;
use crate::events::AgentEvent;
use crate::middleware::{AgentRun, Middleware};
use tinyinference_llm::model::ModelResponse;

/// Fewest model calls that must remain after the answer for the check to run.
/// The check takes one; leaving at least two more keeps it clear of a host's
/// penultimate/final-call wrap-up and leaves room to act on what it finds.
pub const MIN_REMAINING_MODEL_CALLS: usize = 3;

/// Default for
/// [`with_min_remaining_wall_clock`](VerifyBeforeFinishMiddleware::with_min_remaining_wall_clock).
pub const DEFAULT_MIN_REMAINING_WALL_CLOCK: Duration = Duration::from_secs(120);

/// What a run did before its answer, as seen by a [`FinishCheckTrigger`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

/// Per-run bookkeeping, keyed by [`RunContext::instance_id`].
#[derive(Default)]
struct RunState {
    activity: FinishActivity,
    fired: bool,
}

/// Asks a multi-step run, once, to check its final answer against the request
/// before that answer ends the run. See the module docs.
///
/// Generic over the application state and run-context payload: nothing here
/// reads either. The host supplies the check text and decides which runs get
/// it ([`with_min_tool_rounds`](Self::with_min_tool_rounds) or
/// [`with_trigger`](Self::with_trigger)).
pub struct VerifyBeforeFinishMiddleware {
    check: String,
    trigger: FinishCheckTrigger,
    min_remaining_wall_clock: Duration,
    wall_clock_limit: Option<Duration>,
    runs: Mutex<HashMap<u64, RunState>>,
}

impl VerifyBeforeFinishMiddleware {
    /// A middleware that appends `check` as a user turn. By default it fires
    /// for any run that used at least one tool round, so a plain chat answer is
    /// never second-guessed.
    pub fn new(check: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            trigger: min_rounds_trigger(1),
            min_remaining_wall_clock: DEFAULT_MIN_REMAINING_WALL_CLOCK,
            wall_clock_limit: None,
            runs: Mutex::default(),
        }
    }

    /// Fire only for runs with at least `rounds` tool rounds. Replaces any
    /// trigger set before.
    pub fn with_min_tool_rounds(mut self, rounds: usize) -> Self {
        self.trigger = min_rounds_trigger(rounds);
        self
    }

    /// Fire only when `trigger` accepts the run's activity. Replaces any
    /// trigger set before.
    pub fn with_trigger(
        mut self,
        trigger: impl Fn(&FinishActivity) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.trigger = Arc::new(trigger);
        self
    }

    /// Skip the check when the run's wall-clock deadline is closer than `min`.
    /// A run with no deadline is never skipped for time.
    pub fn with_min_remaining_wall_clock(mut self, min: Duration) -> Self {
        self.min_remaining_wall_clock = min;
        self
    }

    /// The run's policy-level wall-clock cap
    /// ([`RunLimits::max_wall_clock_ms`](crate::limits::RunLimits::max_wall_clock_ms)),
    /// measured from the run's start. A `RunPolicy` cap is enforced by the loop
    /// but is not on the [`RunContext`] a middleware sees, so a host that sets
    /// one passes it here too; the context's own deadline
    /// ([`RunContext::remaining_wall_clock`]) is always honoured.
    pub fn with_wall_clock_limit(mut self, limit: Duration) -> Self {
        self.wall_clock_limit = Some(limit);
        self
    }

    /// The tighter of the context deadline and the declared policy cap.
    fn remaining_wall_clock<C>(&self, ctx: &RunContext<C>) -> Option<Duration> {
        let policy = self
            .wall_clock_limit
            .map(|limit| limit.saturating_sub(ctx.limits.elapsed()));
        match (ctx.remaining_wall_clock(), policy) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Why this response must not be held for the check, or `None` when it may.
    fn skip_reason<C>(
        &self,
        ctx: &RunContext<C>,
        response: &ModelResponse,
    ) -> Option<&'static str> {
        if !response.tool_calls().is_empty() {
            return Some("not_final");
        }
        if response.text().trim().is_empty() {
            return Some("empty_answer");
        }
        if response.finish_reason.as_deref() == Some("length") {
            return Some("truncated");
        }
        if response.continue_turn.is_some() {
            return Some("already_continued");
        }
        if ctx.limits.remaining_model_calls() < MIN_REMAINING_MODEL_CALLS {
            return Some("model_call_budget");
        }
        if self
            .remaining_wall_clock(ctx)
            .is_some_and(|left| left < self.min_remaining_wall_clock)
        {
            return Some("wall_clock");
        }
        None
    }
}

fn min_rounds_trigger(rounds: usize) -> FinishCheckTrigger {
    Arc::new(move |activity: &FinishActivity| activity.tool_rounds >= rounds)
}

#[async_trait]
impl<S: Send + Sync, C: Send + Sync> Middleware<S, C> for VerifyBeforeFinishMiddleware {
    fn name(&self) -> &str {
        "verify_before_finish"
    }

    async fn after_model(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        response: &mut ModelResponse,
    ) -> Result<()> {
        let Ok(mut runs) = self.runs.lock() else {
            tracing::warn!("[tinyagents::mw] verify_before_finish state poisoned; not checking");
            return Ok(());
        };
        let run = runs.entry(ctx.instance_id()).or_default();

        let calls = response.tool_calls();
        if !calls.is_empty() {
            run.activity.tool_rounds += 1;
            run.activity
                .tools_called
                .extend(calls.iter().map(|call| call.name.clone()));
            return Ok(());
        }
        if run.fired {
            return Ok(());
        }
        if let Some(reason) = self.skip_reason(ctx, response) {
            tracing::debug!(
                reason,
                tool_rounds = run.activity.tool_rounds,
                remaining_model_calls = ctx.limits.remaining_model_calls(),
                "[tinyagents::mw] verify_before_finish skipped"
            );
            return Ok(());
        }
        if !(self.trigger)(&run.activity) {
            tracing::debug!(
                tool_rounds = run.activity.tool_rounds,
                "[tinyagents::mw] verify_before_finish not triggered by this run's activity"
            );
            return Ok(());
        }

        run.fired = true;
        let tool_rounds = run.activity.tool_rounds;
        drop(runs);
        response.continue_turn = Some(self.check.clone());
        tracing::info!(
            tool_rounds,
            remaining_model_calls = ctx.limits.remaining_model_calls(),
            "[tinyagents::mw] verify_before_finish holding the first final answer for one check"
        );
        ctx.emit(AgentEvent::ControlApplied {
            control: "verify_before_finish".to_string(),
            detail: format!(
                "final answer after {tool_rounds} tool round(s) held for one check against the \
                 request"
            ),
        });
        Ok(())
    }

    async fn after_agent(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        _run: &mut AgentRun,
    ) -> Result<()> {
        if let Ok(mut runs) = self.runs.lock() {
            runs.remove(&ctx.instance_id());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "verify_before_finish_tests.rs"]
mod tests;
