//! Configuration and lifecycle hooks for final-answer verification.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tinyinference_llm::model::ModelResponse;

use super::types::{
    DEFAULT_MIN_REMAINING_WALL_CLOCK, FinishActivity, FinishCheckTrigger,
    MIN_REMAINING_MODEL_CALLS, RunState, VerifyBeforeFinishMiddleware,
};
use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::middleware::{AgentRun, Middleware};

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
        // Interrupted runs can bypass both lifecycle hooks. Evict the oldest
        // process-unique ID before retaining another run in a shared instance.
        const MAX_RETAINED_RUNS: usize = 1_024;
        if !runs.contains_key(&ctx.instance_id())
            && runs.len() >= MAX_RETAINED_RUNS
            && let Some(oldest) = runs.keys().copied().min()
        {
            runs.remove(&oldest);
        }
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

    async fn on_error(&self, ctx: &mut RunContext<C>, _error: &TinyAgentsError) -> Result<()> {
        if let Ok(mut runs) = self.runs.lock() {
            runs.remove(&ctx.instance_id());
        }
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
#[path = "middleware_tests.rs"]
mod tests;
