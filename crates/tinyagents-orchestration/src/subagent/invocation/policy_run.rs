//! Policy application around [`SubAgentTool`](super::SubAgentTool) children:
//! the retry/timeout/budget attempt loop and the leaf-role check.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::{AgentEvent, EventListener, EventRecord, EventSink};
use tinyagents_harness::middleware::AgentRun;

use super::SubAgent;
use crate::subagent::policy::may_retry;
use crate::subagent::{ResultPolicy, SubAgentJobId, SubAgentJobRegistry, SubAgentPolicy};

const LOG_PREFIX: &str = "[subagent-tool-policy]";

/// One pre-minted child attempt: its context and whether it ran any tool.
pub(crate) struct Attempt<Ctx> {
    pub(crate) child: RunContext<Ctx>,
    pub(crate) tools_ran: Arc<AtomicBool>,
}

/// Forwards every event to the parent sink while noting tool executions, so a
/// failed attempt can be classified as having had side effects.
struct ToolWatch {
    parent: EventSink,
    tools_ran: Arc<AtomicBool>,
}

impl EventListener for ToolWatch {
    fn on_event(&self, record: &EventRecord) {
        if matches!(record.event, AgentEvent::ToolStarted { .. }) {
            self.tools_ran.store(true, Ordering::SeqCst);
        }
        self.parent.emit(record.event.clone());
    }
}

impl<Ctx> Attempt<Ctx> {
    /// Wraps `child`; when `watch` is set its events pass through a
    /// [`ToolWatch`] (only needed when a retry is possible).
    pub(crate) fn new(child: RunContext<Ctx>, watch: bool) -> Self {
        let tools_ran = Arc::new(AtomicBool::new(false));
        let child = if watch {
            let sink = EventSink::with_stream_id(child.run_id().as_str());
            sink.subscribe(Arc::new(ToolWatch {
                parent: child.events.clone(),
                tools_ran: tools_ran.clone(),
            }));
            child.with_events(sink)
        } else {
            child
        };
        Self { child, tools_ran }
    }
}

/// Runs `attempts` in order under `policy` and returns the first result that is
/// not a retryable failure. A timeout cancels the child (via `cancel`) and is
/// never retried; token/call budgets are checked on success.
pub(crate) async fn run_attempts<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    subagent: &SubAgent<State, Ctx>,
    policy: &SubAgentPolicy,
    state: &State,
    attempts: Vec<Attempt<Ctx>>,
    input: String,
    streaming: bool,
    cancel: &tinyagents_harness::cancel::CancellationToken,
) -> Result<AgentRun> {
    let last = attempts.len().saturating_sub(1);
    for (index, attempt) in attempts.into_iter().enumerate() {
        let tools_ran = attempt.tools_ran.clone();
        let run = subagent.run_hosted_child(state, attempt.child, input.clone(), streaming);
        let result = match policy.timeout {
            Some(limit) => match tokio::time::timeout(limit, run).await {
                Ok(result) => result,
                Err(_) => {
                    cancel.cancel();
                    tracing::debug!("{LOG_PREFIX} timeout agent={}", subagent.name());
                    return Err(TinyAgentsError::Timeout(format!(
                        "sub-agent `{}` timed out after {limit:?}",
                        subagent.name()
                    )));
                }
            },
            None => run.await,
        };
        match result {
            Ok(run) => {
                let measured = tinyagents_graph::SubAgentOutput {
                    usage: run.usage,
                    model_calls: run.model_calls,
                    tool_calls: run.tool_calls,
                    ..Default::default()
                };
                policy.budget.check(&measured, subagent.name())?;
                return Ok(run);
            }
            Err(error)
                if index < last
                    && !cancel.is_cancelled()
                    && may_retry(policy, index, &error, tools_ran.load(Ordering::SeqCst)) =>
            {
                tracing::debug!(
                    "{LOG_PREFIX} retry agent={} attempt={}",
                    subagent.name(),
                    index + 1
                );
                policy.retry.sleep_backoff(index + 1).await;
            }
            Err(error) => return Err(error),
        }
    }
    Err(TinyAgentsError::Validation(
        "sub-agent had no attempt to run".into(),
    ))
}

/// Records `result` on the job, applying the result policy to a success first.
pub(crate) async fn settle(
    jobs: &SubAgentJobRegistry,
    id: &SubAgentJobId,
    result: Result<AgentRun>,
    result_policy: &ResultPolicy,
) {
    let applied = match (&result, result_policy.is_active()) {
        (Ok(run), true) => Some(
            result_policy
                .apply(
                    id.as_str(),
                    &run.text().unwrap_or_default(),
                    run.structured.as_ref(),
                )
                .await,
        ),
        _ => None,
    };
    jobs.mark_result(id, result);
    if let Some(applied) = applied {
        jobs.apply_result(id, applied);
    }
}

/// Names of delegation tools the child's harness exposes, for a leaf check.
pub(crate) fn delegation_tools_exposed<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    subagent: &SubAgent<State, Ctx>,
    host_delegation_tools: &[String],
) -> Vec<String> {
    subagent
        .harness()
        .tools()
        .names()
        .into_iter()
        .filter(|name| crate::subagent::is_delegation_tool(name, host_delegation_tools))
        .collect()
}
