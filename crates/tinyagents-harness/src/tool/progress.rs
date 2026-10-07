//! Live mid-call progress from a running tool, and the guard that keeps it
//! ordered.
//!
//! A tool reports through [`tinytools::ToolRunContext::report_progress`]. The
//! loop gives every executing call a [`ToolProgressGate`]: the sink behind
//! that method. The gate turns each accepted update into an
//! [`AgentEvent::ToolProgress`] **immediately** (so a UI sees activity while
//! the tool is still running) and queues a matching
//! [`ToolDelta`] for the middleware stack's `on_tool_delta` hook.
//!
//! # Ordering
//!
//! The gate is the single authority on whether an update is still wanted:
//!
//! - Updates are accepted only while the call is *open*. The loop
//!   [closes](ToolProgressGate::close) the gate the moment the tool's future
//!   resolves (return, error, timeout, or cancellation) and **before** it
//!   emits that call's terminal `ToolCompleted` / `ToolFailed`. An update a
//!   detached task reports afterwards is dropped, never emitted late. This is
//!   the same `acceptingUpdates` guard pi's agent loop uses around `onUpdate`.
//! - The open check and the event emission happen under one lock, and so does
//!   closing. There is no window in which a late update passes the check, the
//!   terminal event is emitted, and the progress event then lands after it.
//! - Calls in one concurrent batch each have their own gate, so progress
//!   interleaves across calls but each call's progress precedes its own
//!   terminal event.
//!
//! # Middleware
//!
//! `on_tool_delta` needs `&mut RunContext`, which the loop lends to the
//! executing tool (serial path) or shares read-only with its siblings
//! (concurrent path), so the hook cannot run *during* the call. The gate
//! therefore queues deltas and the loop replays them, in order, to the stack
//! right after the call settles and before its terminal event. Middleware
//! observes progress; the live event is already out, so a rewrite of the delta
//! is not reflected in it.
//!
//! # Flooding
//!
//! A tool in a tight loop must not drown the event stream. Each gate admits at
//! most [`ToolProgressLimits::max_per_window`] events per
//! [`ToolProgressLimits::window`] (default 32 per second). Beyond that, updates
//! are **coalesced**: the newest value of each field replaces the held one, and
//! the held update is emitted when the next window opens or the gate closes, so
//! the final state is never lost. Coalesced-away updates produce no event and
//! no middleware delta.
//!
//! # Reaching the gate
//!
//! The loop does not hand the gate to a `ToolDispatch` — that trait is
//! implemented outside this crate. It scopes the gate in a task-local around
//! the dispatch future instead, and [`ToolExecutionContext::from_run_context`]
//! picks it up when the dispatch builds the tool's context for the matching
//! call id. The sink then lives in the context, so a tool may move it into a
//! spawned task; the gate's `open` flag, not the task-local, is what silences
//! it later.

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tinyinference_llm::tool::ToolDelta;
use tinytools::{ProgressSink, ToolProgress};

use crate::events::{AgentEvent, EventSink};
use crate::ids::CallId;

/// How many progress events one call may emit before coalescing starts.
///
/// See the module docs ("Flooding"). The default admits 32 events per second,
/// far above what a human-facing progress bar needs and far below what a tool
/// looping over a file can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolProgressLimits {
    /// Events admitted per window before further updates are coalesced.
    pub max_per_window: usize,
    /// Length of the window.
    pub window: Duration,
}

impl Default for ToolProgressLimits {
    fn default() -> Self {
        Self {
            max_per_window: 32,
            window: Duration::from_secs(1),
        }
    }
}

#[derive(Default)]
struct GateState {
    /// `false` once the call has settled; further updates are dropped.
    closed: bool,
    /// Deltas for events already emitted, awaiting the middleware replay.
    pending: Vec<ToolDelta>,
    window_start: Option<Instant>,
    emitted_in_window: usize,
    /// The newest coalesced update not yet emitted.
    held: Option<ToolProgress>,
}

/// The per-call destination for a tool's progress updates. See the module docs.
pub(crate) struct ToolProgressGate {
    call_id: CallId,
    tool_name: String,
    events: EventSink,
    limits: ToolProgressLimits,
    state: Mutex<GateState>,
}

tokio::task_local! {
    static CURRENT: Arc<ToolProgressGate>;
}

impl ToolProgressGate {
    pub(crate) fn new(
        call_id: CallId,
        tool_name: impl Into<String>,
        events: EventSink,
        limits: ToolProgressLimits,
    ) -> Arc<Self> {
        Arc::new(Self {
            call_id,
            tool_name: tool_name.into(),
            events,
            limits,
            state: Mutex::new(GateState::default()),
        })
    }

    /// The sink a tool reports through. Holds the gate alive, and stays safe
    /// to call after the gate is closed.
    pub(crate) fn sink(self: &Arc<Self>) -> ProgressSink {
        let gate = Arc::clone(self);
        ProgressSink::new(move |update| gate.accept(update))
    }

    /// Runs `future` with this gate visible to
    /// [`ToolExecutionContext::from_run_context`](super::ToolExecutionContext::from_run_context).
    pub(crate) fn scope<F: Future>(self: &Arc<Self>, future: F) -> impl Future<Output = F::Output> {
        CURRENT.scope(Arc::clone(self), future)
    }

    /// The sink for `call_id`, when a gate for exactly that call is in scope.
    pub(crate) fn current_sink_for(call_id: &CallId) -> Option<ProgressSink> {
        CURRENT
            .try_with(|gate| (&gate.call_id == call_id).then(|| gate.sink()))
            .ok()
            .flatten()
    }

    /// Stops accepting updates, first emitting any coalesced one so the
    /// call's final reported state is never lost. Idempotent.
    pub(crate) fn close(&self) {
        let mut state = self.lock();
        if state.closed {
            return;
        }
        if let Some(held) = state.held.take() {
            self.emit(&mut state, held);
        }
        state.closed = true;
    }

    /// Drains the deltas for events already emitted, for the middleware replay.
    pub(crate) fn take_pending(&self) -> Vec<ToolDelta> {
        std::mem::take(&mut self.lock().pending)
    }

    fn accept(&self, update: ToolProgress) {
        if update.is_empty() {
            return;
        }
        let mut state = self.lock();
        if state.closed {
            tracing::debug!(
                target: "tinyagents::tool_progress",
                call_id = %self.call_id,
                tool = %self.tool_name,
                "[tool_progress] dropped update reported after the call settled"
            );
            return;
        }
        let now = Instant::now();
        let window_open = state
            .window_start
            .is_some_and(|start| now.duration_since(start) < self.limits.window);
        if !window_open {
            state.window_start = Some(now);
            state.emitted_in_window = 0;
            if let Some(held) = state.held.take() {
                self.emit(&mut state, held);
            }
        }
        if state.emitted_in_window < self.limits.max_per_window {
            self.emit(&mut state, update);
        } else {
            tracing::trace!(
                target: "tinyagents::tool_progress",
                call_id = %self.call_id,
                tool = %self.tool_name,
                "[tool_progress] coalescing update past the per-window limit"
            );
            state.held = Some(match state.held.take() {
                Some(older) => merge(older, update),
                None => update,
            });
        }
    }

    fn emit(&self, state: &mut GateState, update: ToolProgress) {
        state.emitted_in_window += 1;
        state.pending.push(ToolDelta {
            call_id: self.call_id.as_str().to_string(),
            content: delta_content(&update),
            tool_name: Some(self.tool_name.clone()),
            ..ToolDelta::default()
        });
        self.events.emit(AgentEvent::ToolProgress {
            call_id: self.call_id.clone(),
            message: update.message.unwrap_or_default(),
            fraction: update.fraction,
            partial: update.partial,
        });
    }

    fn lock(&self) -> MutexGuard<'_, GateState> {
        // A poisoned lock only means a listener panicked mid-emit; the state
        // is still coherent, and progress must not take the run down.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Newer fields replace older ones; a field the newer update omits is kept.
fn merge(older: ToolProgress, newer: ToolProgress) -> ToolProgress {
    ToolProgress {
        message: newer.message.or(older.message),
        fraction: newer.fraction.or(older.fraction),
        partial: newer.partial.or(older.partial),
    }
}

/// The text the middleware sees: the status line, else the partial output,
/// else the fraction as a percentage.
fn delta_content(update: &ToolProgress) -> String {
    if let Some(message) = &update.message {
        return message.clone();
    }
    if let Some(partial) = &update.partial {
        return partial.to_string();
    }
    update
        .fraction
        .map(|fraction| format!("{:.0}%", fraction * 100.0))
        .unwrap_or_default()
}
