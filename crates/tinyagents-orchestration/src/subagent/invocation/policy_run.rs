//! Policy application around [`SubAgentTool`](super::SubAgentTool) children:
//! the retry/timeout/budget attempt loop and the leaf-role check.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::events::{AgentEvent, EventListener, EventRecord, EventSink};
use tinyagents_harness::middleware::AgentRun;

use super::SubAgent;
use crate::subagent::policy::may_retry;
use crate::subagent::{ResultPolicy, SubAgentJobId, SubAgentJobRegistry, SubAgentPolicy};

const LOG_PREFIX: &str = "[subagent-tool-policy]";

/// One child attempt: its context, id, and whether it ran any tool.
///
/// Attempts after the first are given ids derived from the first child's
/// (`{first}-a{n}`), so enabling retries consumes no extra child ordinals from
/// the parent: sibling child ids are identical whether or not a retry happens.
pub(crate) struct Attempt<Ctx> {
    pub(crate) child: RunContext<Ctx>,
    pub(crate) tools_ran: Arc<AtomicBool>,
    pub(crate) run_id: String,
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
        let run_id = child.run_id().as_str().to_owned();
        let child = if watch {
            let sink = EventSink::with_stream_id(&run_id);
            sink.subscribe(Arc::new(ToolWatch {
                parent: child.events.clone(),
                tools_ran: tools_ran.clone(),
            }));
            child.with_events(sink)
        } else {
            child
        };
        Self {
            child,
            tools_ran,
            run_id,
        }
    }
}

/// How the attempts ended.
pub(crate) enum Finished {
    /// A run within budget.
    Run(AgentRun),
    /// The run completed but overshot a token/call budget: its work is kept.
    OverBudget { run: AgentRun, error: TinyAgentsError },
    /// No attempt produced a run.
    Failed(TinyAgentsError),
}

/// Runs `attempts` in order under `policy` and returns the first result that is
/// not a retryable failure. A timeout cancels the child (via `cancel`) and is
/// never retried; budgets are checked on success. Each attempt records its run
/// id on the job when it starts, so the job link names the current attempt.
///
/// Subagent retry compounds with the harness's own per-call model retry: each
/// subagent attempt may itself retry model calls first.
pub(crate) async fn run_attempts<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    subagent: &SubAgent<State, Ctx>,
    policy: &SubAgentPolicy,
    state: &State,
    attempts: Vec<Attempt<Ctx>>,
    input: String,
    streaming: bool,
    cancel: &tinyagents_harness::cancel::CancellationToken,
    jobs: &SubAgentJobRegistry,
    job_id: &SubAgentJobId,
) -> Finished {
    let last = attempts.len().saturating_sub(1);
    for (index, attempt) in attempts.into_iter().enumerate() {
        if index > 0 {
            jobs.set_attempt_run_id(job_id, &attempt.run_id);
        }
        let tools_ran = attempt.tools_ran.clone();
        let run = subagent.run_hosted_child(state, attempt.child, input.clone(), streaming);
        let result = match policy.timeout {
            Some(limit) => match tokio::time::timeout(limit, run).await {
                Ok(result) => result,
                Err(_) => {
                    cancel.cancel();
                    tracing::debug!("{LOG_PREFIX} timeout agent={}", subagent.name());
                    return Finished::Failed(TinyAgentsError::Timeout(format!(
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
                return match policy.budget.check(&measured, subagent.name()) {
                    Ok(()) => Finished::Run(run),
                    Err(error) => Finished::OverBudget { run, error },
                };
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
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Finished::Failed(TinyAgentsError::Cancelled),
                    _ = policy.retry.sleep_backoff(index + 1) => {}
                }
            }
            Err(error) => return Finished::Failed(error),
        }
    }
    Finished::Failed(TinyAgentsError::Validation(
        "sub-agent had no attempt to run".into(),
    ))
}

/// Records the finished attempts on the job, applying the result policy to any
/// output that was produced (including an over-budget one).
pub(crate) async fn settle(
    jobs: &SubAgentJobRegistry,
    id: &SubAgentJobId,
    finished: Finished,
    result_policy: &ResultPolicy,
) {
    match finished {
        Finished::Run(run) => {
            let applied = if result_policy.is_active() {
                Some(
                    result_policy
                        .apply(
                            id.as_str(),
                            &run.text().unwrap_or_default(),
                            run.structured.as_ref(),
                        )
                        .await,
                )
            } else {
                None
            };
            jobs.mark_result(id, Ok(run));
            if let Some(applied) = applied {
                jobs.apply_result(id, applied);
            }
        }
        Finished::OverBudget { run, error } => {
            let applied = result_policy
                .apply(
                    id.as_str(),
                    &run.text().unwrap_or_default(),
                    run.structured.as_ref(),
                )
                .await;
            jobs.mark_budget_overrun(id, applied, error.to_string());
        }
        Finished::Failed(error) => jobs.mark_result(id, Err(error)),
    }
}

/// Names of delegation tools the child's harness exposes, for a leaf check.
pub(crate) fn delegation_tools_exposed<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    subagent: &SubAgent<State, Ctx>,
    host_delegation_tools: &[String],
    own_tool_name: &str,
) -> Vec<String> {
    subagent
        .harness()
        .tools()
        .names()
        .into_iter()
        .filter(|name| {
            name == own_tool_name || crate::subagent::is_delegation_tool(name, host_delegation_tools)
        })
        .collect()
}
