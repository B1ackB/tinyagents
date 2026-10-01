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
///
/// One instance can sit on a reusable harness and serve many runs, including
/// concurrent ones, so the protocol state is keyed by run id and released in
/// `after_agent`. A call that admission stopped before execution (rejected,
/// awaiting approval, deferred) never reaches `after_tool`; its pending entry
/// is dropped with the rest of its run's state.
pub struct MemoryProtocolMiddleware {
    spec: Arc<MemoryProtocolSpec>,
    can_update_index: bool,
    runs: Mutex<HashMap<String, RunProtocolState>>,
}

/// Protocol state for one run.
struct RunProtocolState {
    tracker: MemoryProtocolTracker,
    /// call_id → classified op, captured in `before_tool` (the tool result carries
    /// no arguments, yet `update_memory_md` and `memory_tree` can only be
    /// classified from their `file` / `mode` argument). Correlated back by
    /// the invocation identity in `after_tool`.
    pending_ops: HashMap<String, MemoryOp>,
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
            spec,
            can_update_index,
            runs: Mutex::new(HashMap::new()),
        }
    }

    /// Run `f` against `run_id`'s protocol state, creating it on first use.
    fn with_run<R>(&self, run_id: &str, f: impl FnOnce(&mut RunProtocolState) -> R) -> R {
        let mut runs = match self.runs.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let state = runs
            .entry(run_id.to_string())
            .or_insert_with(|| RunProtocolState {
                tracker: MemoryProtocolTracker::new(Arc::clone(&self.spec)),
                pending_ops: HashMap::new(),
            });
        f(state)
    }

    /// Number of runs currently holding protocol state (test observability).
    #[cfg(test)]
    fn tracked_runs(&self) -> usize {
        self.runs.lock().map(|runs| runs.len()).unwrap_or(0)
    }
}

#[async_trait]
impl<C: Send + Sync> Middleware<(), C> for MemoryProtocolMiddleware {
    fn name(&self) -> &str {
        "memory_protocol"
    }

    async fn before_tool(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        call: &mut TaToolCall,
    ) -> TaResult<()> {
        // Classify with the arguments in hand (the result won't carry them) and
        // stash the op keyed by call id. Only memory-relevant ops are stored, so
        // the map stays empty on turns that never touch memory.
        let op = self.spec.classify(&call.name, &call.arguments);
        if op != MemoryOp::Other {
            self.with_run(ctx.run_id().as_str(), |run| {
                run.pending_ops.insert(call.id.clone(), op);
            });
        }
        Ok(())
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<C>,
        _state: &(),
        invocation: &ToolInvocationIdentity,
        result: &mut TaToolResult,
    ) -> TaResult<()> {
        let tool_name = invocation.tool_name();
        // Consume the op captured for this call (removing it so the map can't
        // grow unbounded). Absent → a non-memory tool: nothing to enforce.
        let call_id = invocation.call_id().to_string();
        let is_error = result.is_error;
        let observation = self.with_run(ctx.run_id().as_str(), |run| {
            let op = run.pending_ops.remove(&call_id)?;
            // Only successful memory ops advance the protocol — a failed write
            // did not mutate memory and must not demand an index update.
            (!is_error).then(|| run.tracker.observe(op))
        });
        let Some(observation) = observation else {
            return Ok(());
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
        ctx: &mut RunContext<C>,
        _state: &(),
        _run: &mut AgentRun,
    ) -> TaResult<()> {
        // Release this run's state: its tracker and any pending entries left by
        // calls that admission stopped before they executed.
        let finished = match self.runs.lock() {
            Ok(mut runs) => runs.remove(ctx.run_id().as_str()),
            Err(poisoned) => poisoned.into_inner().remove(ctx.run_id().as_str()),
        };
        let pending = finished
            .map(|run| run.tracker.pending_index_update())
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
