//! [`MemoryProtocolMiddleware`]: nudge the model back onto the
//! read-index → dedupe → write → update-index memory protocol.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::memory_protocol::{MemoryOp, MemoryProtocolSpec, MemoryProtocolTracker};
use crate::context::RunContext;
use crate::error::Result as TaResult;
use crate::middleware::{AgentRun, Middleware, ToolInvocationIdentity};
use tinyinference_llm::tool::ToolCall as TaToolCall;
use tinytools::{ToolContent, ToolResult as TaToolResult};

/// Agents are told to follow a **read-index → dedupe → write → update-index**
/// cycle around durable memory, but the contract was never enforced, so it was
/// followed inconsistently: writes landed without a dedupe read (duplicating
/// entries) and the index update was skipped (so the index drifted from the
/// store). This middleware observes the ordered sequence of *successful* memory
/// tool calls via [`MemoryProtocolTracker`] and, on each memory write, appends a
/// corrective note to the tool result so the model is nudged back onto the
/// protocol — the same "structured correction surfaced to the model" pattern the
/// unknown-tool recovery (#4118) uses. At run end it warns when a write was never
/// followed by an index update (the index is left stale).
///
/// Only *successful* ops advance the state machine — a failed `memory_store`
/// neither creates an entry nor obliges an index update. Non-memory tools are
/// ignored, so this is a no-op on turns that never touch memory.
pub struct MemoryProtocolMiddleware {
    spec: Arc<MemoryProtocolSpec>,
    can_update_index: bool,
    tracker: Mutex<MemoryProtocolTracker>,
    /// call_id → classified op, captured in `before_tool` (the tool result carries
    /// no arguments, yet `update_memory_md` and `memory_tree` can only be
    /// classified from their `file` / `mode` argument). Correlated back by
    /// the invocation identity in `after_tool`.
    pending_ops: Mutex<HashMap<String, MemoryOp>>,
}

impl MemoryProtocolMiddleware {
    /// Enforce the protocol using `spec`'s tool vocabulary; the agent can call
    /// the index-update tool.
    pub fn new(spec: Arc<MemoryProtocolSpec>) -> Self {
        Self::with_index_update_tool(spec, true)
    }

    /// As [`Self::new`], but `can_update_index` states whether the index-update
    /// tool is actually available this turn. When it is not, dedupe guidance is
    /// kept and no note tells the model to call a tool it does not hold.
    pub fn with_index_update_tool(spec: Arc<MemoryProtocolSpec>, can_update_index: bool) -> Self {
        Self {
            tracker: Mutex::new(MemoryProtocolTracker::new(Arc::clone(&spec))),
            spec,
            can_update_index,
            pending_ops: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<C: Send + Sync> Middleware<(), C> for MemoryProtocolMiddleware {
    fn name(&self) -> &str {
        "memory_protocol"
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<C>,
        _state: &(),
        call: &mut TaToolCall,
    ) -> TaResult<()> {
        // Classify with the arguments in hand (the result won't carry them) and
        // stash the op keyed by call id. Only memory-relevant ops are stored, so
        // the map stays empty on turns that never touch memory.
        let op = self.spec.classify(&call.name, &call.arguments);
        if op != MemoryOp::Other
            && let Ok(mut ops) = self.pending_ops.lock()
        {
            ops.insert(call.id.clone(), op);
        }
        Ok(())
    }

    async fn after_tool(
        &self,
        _ctx: &mut RunContext<C>,
        _state: &(),
        invocation: &ToolInvocationIdentity,
        result: &mut TaToolResult,
    ) -> TaResult<()> {
        let tool_name = invocation.tool_name();
        // Consume the op captured for this call (removing it so the map can't
        // grow unbounded). Absent → a non-memory tool: nothing to enforce.
        let op = self
            .pending_ops
            .lock()
            .ok()
            .and_then(|mut ops| ops.remove(&invocation.call_id().to_string()));
        let Some(op) = op else {
            return Ok(());
        };
        // Only successful memory ops advance the protocol — a failed write did
        // not mutate memory and must not demand an index update.
        if result.is_error {
            return Ok(());
        }
        let observation = {
            let mut tracker = match self.tracker.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            tracker.observe(op)
        };
        if let Some(note) =
            observation.guidance_with_index_update(&self.spec, tool_name, self.can_update_index)
        {
            tracing::debug!(
                tool = tool_name,
                missing_index_read = observation.missing_index_read,
                index_drift = observation.index_drift,
                "[tinyagents::mw] memory-protocol guidance appended to tool result"
            );
            result.content.push(ToolContent::Text {
                text: format!("\n\n{note}"),
            });
            result.markdown_formatted = None;
        }
        Ok(())
    }

    async fn after_agent(
        &self,
        _ctx: &mut RunContext<C>,
        _state: &(),
        _run: &mut AgentRun,
    ) -> TaResult<()> {
        let pending = self
            .tracker
            .lock()
            .map(|tracker| tracker.pending_index_update())
            .unwrap_or(false);
        if self.can_update_index && pending {
            tracing::warn!(
                "[tinyagents::mw] memory-protocol: run ended with a memory write that was never \
                 followed by an index update — the memory index is left stale"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "memory_protocol_middleware_test.rs"]
mod tests;
