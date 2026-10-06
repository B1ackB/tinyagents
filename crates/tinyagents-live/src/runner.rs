//! Starting and driving a live agent session.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use tinyagents_harness::context::RunContext;
use tinyagents_harness::AgentHarness;
use tinyliveagents::{LiveConfig, LiveEvent, LiveEvents, LiveProvider, LiveSender};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::declarations::tool_declarations;
use crate::types::{LiveAgentEvent, LiveAgentOptions};
use crate::worker::{Cancelled, OutcomeRecorder, Worker};

/// Buffered events per session.
const EVENT_CAPACITY: usize = 512;

/// An agent — a harness with its tools and middleware — ready to hold live
/// voice sessions.
///
/// ```ignore
/// let agent = LiveAgent::new(harness, state);
/// let config = LiveConfig::new().with_system_instruction(prompt);
/// let mut session = agent.start(&provider, config, ctx, LiveAgentOptions::default()).await?;
/// let mic = session.sender();
/// while let Some(event) = session.recv().await { /* play audio, show captions */ }
/// ```
pub struct LiveAgent<State, Ctx> {
    harness: Arc<AgentHarness<State, Ctx>>,
    state: Arc<State>,
}

impl<State, Ctx> std::fmt::Debug for LiveAgent<State, Ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveAgent").finish_non_exhaustive()
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync + 'static> LiveAgent<State, Ctx> {
    /// An agent over `harness`, whose tools run against `state`.
    pub fn new(harness: Arc<AgentHarness<State, Ctx>>, state: Arc<State>) -> Self {
        Self { harness, state }
    }

    /// The harness behind this agent.
    pub fn harness(&self) -> &AgentHarness<State, Ctx> {
        &self.harness
    }

    /// Opens a live session on `provider`.
    ///
    /// The harness's tools are declared to the model (appended to any tools
    /// `config` already names). Every tool call the model makes runs through
    /// the harness pipeline under `ctx`, and its result is sent back to the
    /// provider; the host only plays audio and renders events.
    ///
    /// # Errors
    ///
    /// The provider's connection errors.
    pub async fn start(
        &self,
        provider: &dyn LiveProvider,
        mut config: LiveConfig,
        ctx: RunContext<Ctx>,
        options: LiveAgentOptions,
    ) -> tinyliveagents::Result<LiveAgentSession> {
        let declared: HashSet<String> = config.tools.iter().map(|t| t.name.clone()).collect();
        config.tools.extend(
            tool_declarations(&self.harness, options.include_deferred_tools)
                .into_iter()
                .filter(|tool| !declared.contains(&tool.name)),
        );
        tracing::debug!(
            provider = provider.id(),
            tools = config.tools.len(),
            "tinyagents-live: starting session"
        );
        let session = provider.connect(config).await?;
        let (sender, events) = session.split();

        let recorder = Arc::new(OutcomeRecorder::default());
        ctx.events.subscribe(recorder.clone());
        let cancelled: Cancelled = Arc::new(Mutex::new(HashSet::new()));
        let (event_tx, event_rx) = mpsc::channel(EVENT_CAPACITY);
        let (call_tx, call_rx) = mpsc::channel(64);

        let worker = Worker {
            harness: self.harness.clone(),
            state: self.state.clone(),
            ctx,
            recorder,
            cancelled: cancelled.clone(),
            sender: sender.clone(),
            events: event_tx.clone(),
            tool_timeout: options.tool_timeout,
        };
        let worker_task = tokio::spawn(worker.run(call_rx));
        let driver_task = tokio::spawn(drive(events, call_tx, cancelled, event_tx));
        Ok(LiveAgentSession {
            sender,
            events: event_rx,
            tasks: [driver_task, worker_task],
        })
    }
}

/// Forwards provider events to the host and queues tool calls for the worker.
async fn drive(
    mut events: LiveEvents,
    calls: mpsc::Sender<tinyliveagents::ToolCall>,
    cancelled: Cancelled,
    out: mpsc::Sender<LiveAgentEvent>,
) {
    while let Some(event) = events.recv().await {
        match &event {
            LiveEvent::ToolCall(call) => {
                let _ = calls.send(call.clone()).await;
            }
            LiveEvent::ToolCallCancelled { call_ids } => {
                if let Ok(mut set) = cancelled.lock() {
                    set.extend(call_ids.iter().cloned());
                }
            }
            _ => {}
        }
        let closed = matches!(event, LiveEvent::Closed(_));
        if out.send(LiveAgentEvent::Live(event)).await.is_err() || closed {
            break;
        }
    }
}

/// A running live agent session.
///
/// Dropping it stops the session: the provider connection closes and any
/// queued tool calls are abandoned.
pub struct LiveAgentSession {
    sender: LiveSender,
    events: mpsc::Receiver<LiveAgentEvent>,
    tasks: [JoinHandle<()>; 2],
}

impl std::fmt::Debug for LiveAgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveAgentSession").finish_non_exhaustive()
    }
}

impl LiveAgentSession {
    /// The sender for microphone audio, typed input, interruptions and close.
    /// Tool results are sent by the session itself.
    pub fn sender(&self) -> LiveSender {
        self.sender.clone()
    }

    /// The next event; `None` once the session has closed.
    pub async fn recv(&mut self) -> Option<LiveAgentEvent> {
        self.events.recv().await
    }
}

impl Drop for LiveAgentSession {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
