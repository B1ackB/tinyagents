//! Abort-on-drop steering forwarder: bridges a host-owned [`RunQueue`] into a
//! running turn's [`SteeringHandle`].
//!
//! A host that owns a session-level queue of messages typed while a turn is
//! active spawns a 50 ms poll loop per turn that drains the queue's **steer**
//! and **collect** lanes and forwards them into the run's steering handle as
//! injected user messages; the harness applies them at the next iteration
//! checkpoint.
//!
//! Cancellation in hosts is usually **drop-based** (the turn future is dropped by
//! a `select!`; detached sub-agents are hard-aborted). A forwarder that only
//! cleaned up on normal return would detach and loop forever, pinning the queue
//! and handle and racing the next turn's forwarder for the session-owned queue.
//! [`SteeringForwarderGuard`] fixes this with RAII: its `Drop` aborts the poll
//! task, runs the host's cleanup hook (e.g. deregistering a steering-registry
//! entry), and drains residual (delivered-but-unapplied) steers back into the
//! queue so a late steer becomes the *next* turn's input instead of vanishing.
//!
//! The queued payload, the delivery notifications and the cleanup are host
//! seams: [`QueuedMessage`] for the payload, [`ForwardEvent`] for notifications.

use std::sync::Arc;
use std::time::Duration;

use tinyinference_llm::message::Message as TaMessage;

use super::{QueueLane, RunQueue};
use crate::steering::{SteeringCommand, SteeringHandle};

/// Framing prepended to a queued **steer** message when it is injected as a user
/// turn. Shared so the residual-requeue path can strip it and avoid
/// double-prefixing when the next turn re-forwards the recovered text.
pub const STEER_PREFIX: &str = "[User steering message]: ";

/// Framing prepended to a queued **collect** message (orchestrator/monitor
/// context lines). See [`STEER_PREFIX`].
pub const COLLECT_PREFIX: &str = "[Additional context from user]: ";

/// Poll interval of the forwarder task.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// A host payload the forwarder can read text from and re-create when requeuing
/// a recovered steer.
pub trait QueuedMessage: Send + 'static {
    /// Stable id of the queued item.
    fn id(&self) -> &str;
    /// The message text to inject.
    fn text(&self) -> &str;
    /// Build a recovered item to push back onto the queue. `id` is freshly
    /// minted, `thread_label` is the forwarder's thread label, and `queued_at_ms`
    /// is the wall clock in epoch milliseconds.
    fn requeued(id: String, text: String, thread_label: &str, queued_at_ms: u64) -> Self;
}

/// A delivery or requeue the forwarder performed, for host observability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardEvent {
    /// Queued messages were forwarded into the running steering handle.
    Delivered {
        /// Thread label the forwarder was armed with.
        thread_label: String,
        /// `"steer"` or `"collect"`.
        mode: &'static str,
        /// Number of messages delivered.
        delivered: usize,
        /// Id of the first delivered item.
        item_id: Option<String>,
        /// Full text of the first delivered item (hosts clip it for display).
        text: Option<String>,
    },
    /// Delivered-but-unapplied steers were pushed back onto the queue.
    Requeued {
        /// Thread label the forwarder was armed with.
        thread_label: String,
        /// Number of messages requeued.
        requeued: usize,
        /// Minted id of the first requeued item.
        item_id: Option<String>,
        /// Full text of the first requeued item.
        text: Option<String>,
    },
}

/// Host callback receiving [`ForwardEvent`]s.
pub type ForwardEventSink = Arc<dyn Fn(ForwardEvent) + Send + Sync>;

/// Host cleanup run when the guard drops (e.g. deregistering a sub-agent's
/// steering handle from a registry).
pub type ForwarderCleanup = Box<dyn FnOnce() + Send>;

/// Milliseconds since the Unix epoch (best-effort; `0` on a pre-epoch clock).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Shared shutdown gate between the poll task and the guard's `Drop`.
///
/// The poll task sends into the steering handle only while holding this lock
/// and seeing `false`; `Drop` sets it to `true` under the same lock before it
/// drains the handle. So once `Drop` has drained, no later send can land in a
/// handle nobody will read again: the task pushes what it drained back onto the
/// queue instead.
type ClosedGate = std::sync::Mutex<bool>;

fn lock_gate(gate: &ClosedGate) -> std::sync::MutexGuard<'_, bool> {
    match gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Drain `lane` and forward it into `handle`. With a `gate`, delivery happens
/// under it, and items drained after the gate closed are pushed back onto the
/// queue; returns `false` in that case (the caller should stop polling).
#[allow(clippy::too_many_arguments)]
async fn forward_lane<T: QueuedMessage>(
    queue: &RunQueue<T>,
    handle: &SteeringHandle,
    thread_label: &str,
    lane: QueueLane,
    mode: &'static str,
    prefix: &str,
    sink: &ForwardEventSink,
    gate: Option<&ClosedGate>,
) -> bool {
    let drained = queue.drain(lane).await;
    if drained.is_empty() {
        return true;
    }
    let delivered = drained.len();
    let (item_id, text) = drained
        .first()
        .map(|msg| (Some(msg.id().to_string()), Some(msg.text().to_string())))
        .unwrap_or((None, None));
    let undelivered = {
        let closed = gate.map(lock_gate);
        if closed.as_deref().copied().unwrap_or(false) {
            Some(drained)
        } else {
            for msg in &drained {
                handle.send(SteeringCommand::InjectMessage(TaMessage::user(format!(
                    "{prefix}{}",
                    msg.text()
                ))));
            }
            None
        }
    };
    if let Some(items) = undelivered {
        tracing::debug!(
            thread_id = thread_label,
            returned = items.len(),
            mode,
            "[run_queue] forwarder closed mid-delivery; returned drained message(s) to the queue"
        );
        // Back at the front, in order: anything queued after the drain is
        // newer and must stay behind these.
        queue.requeue_front(lane, items).await;
        return false;
    }
    tracing::debug!(
        thread_id = thread_label,
        delivered,
        mode,
        "[run_queue] delivered message(s) into running steering handle"
    );
    sink(ForwardEvent::Delivered {
        thread_label: thread_label.to_string(),
        mode,
        delivered,
        item_id,
        text,
    });
    true
}

/// Drain the queue's pending **steer** messages and forward them to the
/// [`SteeringHandle`] as injected user turns (the harness applies them to the
/// working transcript at the next iteration checkpoint). Emits a
/// [`ForwardEvent::Delivered`] when at least one message is delivered.
pub async fn forward_steers<T: QueuedMessage>(
    queue: &RunQueue<T>,
    handle: &SteeringHandle,
    thread_label: &str,
    sink: &ForwardEventSink,
) {
    forward_lane(
        queue,
        handle,
        thread_label,
        QueueLane::Steer,
        "steer",
        STEER_PREFIX,
        sink,
        None,
    )
    .await;
}

/// Forward any queued **collect** messages as injected user turns so they reach
/// the next LLM call as additional context, framed with [`COLLECT_PREFIX`].
/// Emits a [`ForwardEvent::Delivered`] on delivery.
///
/// This drains the lane, so an item forwarded here is no longer handed back on
/// [`AgentRun::collected`][crate::middleware::AgentRun::collected]: a host
/// that arms a forwarder over a queue opts its collect lane into injection
/// (see [`QueueLane::Collect`]).
pub async fn forward_collects<T: QueuedMessage>(
    queue: &RunQueue<T>,
    handle: &SteeringHandle,
    thread_label: &str,
    sink: &ForwardEventSink,
) {
    forward_lane(
        queue,
        handle,
        thread_label,
        QueueLane::Collect,
        "collect",
        COLLECT_PREFIX,
        sink,
        None,
    )
    .await;
}

/// Stop-on-drop guard around the steering-forwarder poll task.
///
/// Held across the harness drive future so its `Drop` runs on **every** exit
/// path (normal return, error, and drop-cancellation), stopping the poll task,
/// running the host cleanup, and requeuing residual steers.
pub struct SteeringForwarderGuard<T: QueuedMessage> {
    /// The spawned poll task. `None` after drop so a double-drop is a no-op.
    forwarder: Option<tokio::task::JoinHandle<()>>,
    /// Closed by `Drop` before it drains the handle; see [`ClosedGate`].
    closed: Arc<ClosedGate>,
    /// A clone of the run's steering handle, drained on drop to recover
    /// delivered-but-unapplied steers.
    handle: SteeringHandle,
    /// The session-owned run queue, used to requeue residual steers on drop.
    /// `None` after the requeue so a double-drop is a no-op.
    run_queue: Option<Arc<RunQueue<T>>>,
    /// Host cleanup run exactly once on drop.
    cleanup: Option<ForwarderCleanup>,
    /// Best-effort thread label for observability + requeued-message metadata.
    thread_label: String,
    sink: ForwardEventSink,
}

impl<T: QueuedMessage> SteeringForwarderGuard<T> {
    /// Arm the guard: spawn the poll loop when a `run_queue` is present and wrap
    /// all cleanup in an abort-on-drop scope.
    ///
    /// `run_queue` is `None` for a steering-only run (a sub-agent controlled
    /// purely through a steering registry, with no queue lane): no poll task is
    /// spawned, but the guard still runs `cleanup` on every exit path so an
    /// aborted run cannot leak a registry entry. Must be called inside a tokio
    /// runtime when `run_queue` is `Some`.
    pub fn new(
        handle: SteeringHandle,
        run_queue: Option<Arc<RunQueue<T>>>,
        cleanup: Option<ForwarderCleanup>,
        thread_label: String,
        sink: ForwardEventSink,
    ) -> Self {
        let closed = Arc::new(ClosedGate::new(false));
        let forwarder = run_queue.as_ref().map(|queue| {
            let loop_queue = queue.clone();
            let loop_handle = handle.clone();
            let loop_label = thread_label.clone();
            let loop_sink = sink.clone();
            let loop_closed = closed.clone();
            tokio::spawn(async move {
                // Not aborted on drop: an abort could land between draining the
                // queue and delivering, losing what was drained. The loop exits
                // on its own once the gate is closed, returning anything it
                // drained after closing to the queue.
                for (lane, mode, prefix) in [
                    (QueueLane::Steer, "steer", STEER_PREFIX),
                    (QueueLane::Collect, "collect", COLLECT_PREFIX),
                ]
                .into_iter()
                .cycle()
                {
                    if lane == QueueLane::Steer {
                        tokio::time::sleep(POLL_INTERVAL).await;
                    }
                    if *lock_gate(&loop_closed) {
                        break;
                    }
                    let open = forward_lane(
                        &loop_queue,
                        &loop_handle,
                        &loop_label,
                        lane,
                        mode,
                        prefix,
                        &loop_sink,
                        Some(&loop_closed),
                    )
                    .await;
                    if !open {
                        break;
                    }
                }
            })
        });
        tracing::debug!(
            thread_id = thread_label.as_str(),
            has_queue = run_queue.is_some(),
            "[run_queue] steering forwarder guard armed (abort-on-drop)"
        );
        Self {
            forwarder,
            closed,
            handle,
            run_queue,
            cleanup,
            thread_label,
            sink,
        }
    }
}

impl<T: QueuedMessage> Drop for SteeringForwarderGuard<T> {
    fn drop(&mut self) {
        // 1. Close the gate so the poll loop can no longer deliver into this
        //    handle or race the next turn's forwarder for the shared,
        //    session-owned run queue. Anything it drains from now on goes back
        //    to the queue, and it exits at its next check.
        *lock_gate(&self.closed) = true;
        if self.forwarder.take().is_some() {
            tracing::debug!(
                thread_id = self.thread_label.as_str(),
                "[run_queue] closed steering forwarder (guard drop)"
            );
        }

        // 2. Host cleanup (e.g. deregister a sub-agent steering handle so an
        //    aborted run does not leak a registry entry keyed by a dead handle).
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
            tracing::debug!("[run_queue] ran forwarder cleanup (guard drop)");
        }

        // 3. Recover residual steers: messages the poll loop already delivered
        //    into the handle but that the harness ended/cancelled before
        //    applying at a checkpoint. Strip the delivery prefix so the next
        //    turn's forwarder re-frames them cleanly (no double prefix).
        //    Control-flow-only commands (Pause/Resume/Cancel/...) are meaningless
        //    once the run is gone and are intentionally dropped.
        let residual = self.handle.drain();
        // Each residual steer gets its requeued id minted here (Drop is
        // synchronous) so the `Requeued` event below and the item actually
        // pushed onto the queue in the spawned task agree on the same id.
        let requeue_items: Vec<(String, QueueLane, String)> = residual
            .into_iter()
            .filter_map(|cmd| match cmd {
                SteeringCommand::InjectMessage(msg) => {
                    let text = msg.text();
                    // The prefix that matched tells us the lane: preserve it so a
                    // delivered-but-unapplied collect line re-enters as Collect
                    // rather than being re-labeled as user Steer. Default to
                    // Steer when neither prefix is present (a raw, unframed steer).
                    let (text, lane) = if let Some(rest) = text.strip_prefix(STEER_PREFIX) {
                        (rest.to_string(), QueueLane::Steer)
                    } else if let Some(rest) = text.strip_prefix(COLLECT_PREFIX) {
                        (rest.to_string(), QueueLane::Collect)
                    } else {
                        (text.to_string(), QueueLane::Steer)
                    };
                    Some((text, lane, uuid::Uuid::new_v4().to_string()))
                }
                _ => None,
            })
            .collect();

        let Some(queue) = self.run_queue.take() else {
            return;
        };
        if requeue_items.is_empty() {
            return;
        }
        let requeued = requeue_items.len();
        let thread_label = self.thread_label.clone();
        let (item_id, text) = requeue_items
            .first()
            .map(|(text, _lane, id)| (Some(id.clone()), Some(text.clone())))
            .unwrap_or((None, None));

        // `RunQueue::push` is async (tokio `Mutex`); `Drop` is synchronous. Push
        // the recovered steers back on a detached task so they land in the
        // session queue and become the next turn's input. The forwarder poll
        // loop keeps re-draining the queue, so even if the requeue completes
        // after the next turn starts, its next tick picks them up.
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => {
                let label = thread_label.clone();
                rt.spawn(async move {
                    // Appended, not prepended: a later guard's residuals must
                    // not jump ahead of these, and successive drops each add
                    // theirs behind the last.
                    for (text, lane, id) in requeue_items {
                        queue
                            .push(lane, T::requeued(id, text, &label, now_ms()))
                            .await;
                    }
                });
            }
            Err(_) => {
                // No runtime to spawn on (should not happen on live paths, which
                // always drop inside a tokio task). The steers are lost rather
                // than silently mis-handled: log loudly.
                tracing::warn!(
                    thread_id = thread_label.as_str(),
                    requeued,
                    "[run_queue] could not requeue residual steers: no tokio runtime at guard drop"
                );
                return;
            }
        }

        tracing::debug!(
            thread_id = thread_label.as_str(),
            requeued,
            "[run_queue] requeued residual steer(s) as next-turn input (guard drop)"
        );
        (self.sink)(ForwardEvent::Requeued {
            thread_label,
            requeued,
            item_id,
            text,
        });
    }
}

#[cfg(test)]
#[path = "forwarder_test.rs"]
mod tests;
