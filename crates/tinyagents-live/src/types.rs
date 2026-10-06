//! Types for live agent sessions.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tinyliveagents::LiveEvent;

/// A boxed, sendable unit future.
pub type BoxedTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Wraps the tool worker's future before it is spawned.
///
/// Hosts that keep request-scoped state in task-locals — an approval channel,
/// a turn origin, a tracing span — install it here, because a spawned task
/// does not inherit the caller's task-locals:
///
/// ```ignore
/// let scope: TaskScope = Arc::new(move |task| Box::pin(MY_LOCAL.scope(value.clone(), task)));
/// ```
pub type TaskScope = Arc<dyn Fn(BoxedTask) -> BoxedTask + Send + Sync>;

/// How a live agent session runs its tools.
#[derive(Debug, Clone)]
pub struct LiveAgentOptions {
    /// Longest a single tool call may run before the model is told it timed
    /// out. The harness's own per-tool timeout policy still applies inside
    /// this bound.
    pub tool_timeout: Duration,
    /// Also declare the harness's deferred tools to the model, not only the
    /// directly exposed ones.
    pub include_deferred_tools: bool,
}

impl Default for LiveAgentOptions {
    fn default() -> Self {
        Self {
            tool_timeout: Duration::from_secs(120),
            include_deferred_tools: false,
        }
    }
}

/// Something that happened in a live agent session.
///
/// Every provider event is forwarded as [`LiveAgentEvent::Live`] — including
/// [`LiveEvent::ToolCall`], so a host can show the call — and the runner adds
/// a [`LiveAgentEvent::ToolStarted`] / [`LiveAgentEvent::ToolFinished`] pair
/// around each call it executes.
#[derive(Debug, Clone, PartialEq)]
pub enum LiveAgentEvent {
    /// A provider event, forwarded unchanged.
    Live(LiveEvent),
    /// The harness started executing a tool call.
    ToolStarted {
        /// The provider's call id.
        call_id: String,
        /// The tool name.
        name: String,
    },
    /// A tool call finished and its result was sent to the provider (or was
    /// dropped because the provider cancelled the call meanwhile).
    ToolFinished {
        /// The provider's call id.
        call_id: String,
        /// The tool name.
        name: String,
        /// Whether the result describes a failure, a denial or a timeout.
        is_error: bool,
        /// Whether the provider cancelled the call before it finished, so the
        /// result was not sent.
        cancelled: bool,
        /// How long the call took.
        duration: Duration,
    },
}
