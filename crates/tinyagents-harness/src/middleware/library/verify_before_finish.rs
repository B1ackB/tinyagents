//! [`VerifyBeforeFinishMiddleware`] (skeleton).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result;
use crate::middleware::Middleware;
use tinyinference_llm::model::ModelResponse;

/// What a run did before its answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FinishActivity {
    /// Model responses that requested at least one tool call.
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

/// The middleware.
pub struct VerifyBeforeFinishMiddleware {
    check: String,
    trigger: FinishCheckTrigger,
    min_remaining_wall_clock: Duration,
}

impl VerifyBeforeFinishMiddleware {
    pub fn new(check: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            trigger: Arc::new(|_| false),
            min_remaining_wall_clock: Duration::ZERO,
        }
    }

    pub fn with_min_tool_rounds(self, _rounds: usize) -> Self {
        self
    }

    pub fn with_trigger(
        self,
        _trigger: impl Fn(&FinishActivity) -> bool + Send + Sync + 'static,
    ) -> Self {
        self
    }

    pub fn with_min_remaining_wall_clock(self, _min: Duration) -> Self {
        self
    }
}

#[async_trait]
impl<S: Send + Sync, C: Send + Sync> Middleware<S, C> for VerifyBeforeFinishMiddleware {
    fn name(&self) -> &str {
        "verify_before_finish"
    }

    async fn after_model(
        &self,
        _ctx: &mut RunContext<C>,
        _state: &S,
        _response: &mut ModelResponse,
    ) -> Result<()> {
        let _ = (&self.check, &self.trigger, self.min_remaining_wall_clock);
        Ok(())
    }
}

#[cfg(test)]
#[path = "verify_before_finish_tests.rs"]
mod tests;
