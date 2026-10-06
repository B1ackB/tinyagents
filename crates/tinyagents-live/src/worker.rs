//! The tool worker: executes a live session's tool calls through a harness.
//!
//! One worker owns the session's [`RunContext`] (so limits, budgets and events
//! accumulate across the conversation, exactly as across one agent run) and
//! runs calls one at a time through
//! [`tinyagents_harness::agent_loop::phases::execute_tool_batch`] — the same
//! admission, `before_tool` / wrap / `after_tool` middleware and fold the agent
//! loop uses. A call the provider cancels before it starts is skipped; one
//! cancelled while running is allowed to finish (tools can have side effects)
//! but its result is not sent.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tinyagents_harness::agent_loop::phases::execute_tool_batch;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::{AgentEvent, EventListener, EventRecord, HarnessRunStatus};
use tinyagents_harness::ComponentId;
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::tinyinference_llm::message::Message;
use tinyagents_harness::tinyinference_llm::tool::ToolCall as HarnessToolCall;
use tinyagents_harness::runtime::AgentHarness;
use tinyliveagents::{LiveSender, ToolCall, ToolResult};
use tokio::sync::mpsc;

use crate::types::LiveAgentEvent;

/// Outcomes the harness reported for each call id, read off its event stream.
/// The fold's tool message carries no error flag, so this is where a failure,
/// a denial or a policy refusal is learned.
#[derive(Default)]
pub(crate) struct OutcomeRecorder {
    errors: Mutex<HashMap<String, bool>>,
}

impl OutcomeRecorder {
    /// Whether the harness reported `call_id` as failed. `None` when it
    /// reported nothing (the call never reached a tool).
    pub(crate) fn take(&self, call_id: &str) -> Option<bool> {
        self.errors
            .lock()
            .ok()
            .and_then(|mut map| map.remove(call_id))
    }
}

impl EventListener for OutcomeRecorder {
    fn on_event(&self, record: &EventRecord) {
        let outcome = match &record.event {
            AgentEvent::ToolCompleted { call_id, error, .. } => {
                Some((call_id.to_string(), error.is_some()))
            }
            AgentEvent::ToolFailed { call_id, .. } => Some((call_id.to_string(), true)),
            _ => None,
        };
        if let (Some((call_id, failed)), Ok(mut map)) = (outcome, self.errors.lock()) {
            map.insert(call_id, failed);
        }
    }
}

/// Call ids the provider cancelled.
pub(crate) type Cancelled = Arc<Mutex<HashSet<String>>>;

/// Everything the worker needs.
pub(crate) struct Worker<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) harness: Arc<AgentHarness<State, Ctx>>,
    pub(crate) state: Arc<State>,
    pub(crate) ctx: RunContext<Ctx>,
    pub(crate) recorder: Arc<OutcomeRecorder>,
    pub(crate) cancelled: Cancelled,
    pub(crate) sender: LiveSender,
    pub(crate) events: mpsc::Sender<LiveAgentEvent>,
    pub(crate) tool_timeout: Duration,
}

/// The text a tool message carries.
fn message_text(messages: &[Message]) -> Option<String> {
    messages.first().map(Message::text)
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> Worker<State, Ctx> {
    /// Runs calls from `calls` until the channel closes.
    pub(crate) async fn run(mut self, mut calls: mpsc::Receiver<ToolCall>) {
        let mut run = AgentRun::new();
        let mut status =
            HarnessRunStatus::new(self.ctx.run_id().clone(), ComponentId::new("live-session".to_string()));
        while let Some(call) = calls.recv().await {
            if self.is_cancelled(&call.call_id) {
                tracing::debug!(call_id = %call.call_id, "tinyagents-live: skipping cancelled call");
                continue;
            }
            let _ = self
                .events
                .send(LiveAgentEvent::ToolStarted {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                })
                .await;
            let started = Instant::now();
            let result = self.execute(&mut run, &mut status, &call).await;
            let cancelled = self.is_cancelled(&call.call_id);
            let is_error = result.is_error;
            if !cancelled {
                let _ = self.sender.send_tool_result(result).await;
            }
            tracing::debug!(
                call_id = %call.call_id,
                tool = %call.name,
                is_error,
                cancelled,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "tinyagents-live: tool call finished"
            );
            let _ = self
                .events
                .send(LiveAgentEvent::ToolFinished {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    is_error,
                    cancelled,
                    duration: started.elapsed(),
                })
                .await;
        }
    }

    fn is_cancelled(&self, call_id: &str) -> bool {
        self.cancelled
            .lock()
            .map(|set| set.contains(call_id))
            .unwrap_or(false)
    }

    async fn execute(
        &mut self,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        call: &ToolCall,
    ) -> ToolResult {
        let args = match &call.args {
            Value::Object(_) => call.args.clone(),
            Value::Null => Value::Object(serde_json::Map::new()),
            other => other.clone(),
        };
        let harness_call = HarnessToolCall::new(call.call_id.clone(), call.name.clone(), args);
        let mut messages = Vec::new();
        let batch = execute_tool_batch(
            &self.harness,
            &self.state,
            &mut self.ctx,
            run,
            status,
            &mut messages,
            vec![harness_call],
        );
        match tokio::time::timeout(self.tool_timeout, batch).await {
            Err(_) => ToolResult::error(
                call,
                format!("the tool did not finish within {}s", self.tool_timeout.as_secs()),
            ),
            Ok(Err(error)) => ToolResult::error(call, error.to_string()),
            Ok(Ok(outcome)) => {
                let failed = self.recorder.take(&call.call_id);
                match message_text(&outcome.results) {
                    // No tool message: the call was deferred (it needs an
                    // approval or an external answer the live session cannot
                    // wait on).
                    None => ToolResult::error(
                        call,
                        "the call was not run: it needs an approval that was not given",
                    ),
                    Some(text) => {
                        let mut result = ToolResult::ok(call, Value::String(text));
                        result.is_error = failed.unwrap_or(false);
                        result
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
