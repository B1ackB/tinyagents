//! Nested tool calls (C9): the agent-loop side of
//! [`ToolExecutionContext::call_tool`](crate::tool::ToolExecutionContext::call_tool).
//!
//! # Shape
//!
//! A tool future holds only `&RunContext` (the concurrent path runs several at
//! once) and cannot own the harness or the run state, so the nested runner a
//! tool sees is a **channel**: [`ToolCallBase::call`] drives the tool's future
//! and, in the same poll loop, services the nested calls it sends. Servicing
//! borrows exactly what the model-issued path borrows (`&AgentHarness`,
//! `&State`, `&RunContext`), so a nested call runs through the same
//! lookup, validation, host authorization and tool-wrap onion — and, being a
//! plain [`ToolCallBase::call`] itself, can nest again.
//!
//! # What a nested call shares with the run
//!
//! - **Budget**: one `max_tool_calls` pool. The slot is reserved on the shared
//!   atomic in [`crate::limits::LimitTracker`], so concurrent parents cannot
//!   overspend it, and model-issued admission counts reserved nested slots.
//! - **Cancellation and wall clock**: the run's; a cancelled run refuses the
//!   next nested call, and dropping the parent drops its in-flight nested calls
//!   (each gets a `ToolFailed`, so every `ToolStarted` has a terminal event).
//! - **Events**: `ToolStarted`/`ToolCompleted`/`ToolFailed` with
//!   `parent_call_id` (the *immediate* parent), under the id
//!   `<parent call id>/<n>`.
//! - **Effect ledger**: one row per nested call, keyed by the nested id.
//!
//! # What applies to a nested call, exactly
//!
//! Applies: tool lookup and the host allow-list, argument preparation and
//! validation, the approval refusal, [`check_nested_tool`](crate::middleware::Middleware::check_nested_tool) on every
//! registered middleware, host authorization (with
//! `ToolCallRequest::parent_call_id` set), the tool-wrap onion
//! (`ToolMiddleware::wrap_tool`), timeouts, the run budget,
//! [`observe_nested_result`](crate::middleware::Middleware::observe_nested_result) after the call.
//!
//! Does **not** apply: `Middleware::before_tool` / `after_tool` proper (they
//! take `&mut RunContext`, which a tool future holding `&RunContext` cannot
//! lend), so enforcement that lives only in `before_tool` is *not* applied
//! unless the middleware also implements `check_nested_tool`; the progress
//! gate; host output screening; a result's `ToolControl`.
//!
//! # What a nested call deliberately does not do
//!
//! - **No transcript rows.** A nested call is not answered to the provider, so
//!   it must not appear in tool-call/tool-result pairing. A capped summary is
//!   attached to the *parent's* result metadata instead.
//! - **No deferral.** The parent is mid-execution, so a nested call that would
//!   need approval (or be deferred) fails with a clear error.
//! - **Unbounded refusals.** After [`MAX_NESTED_REFUSALS`] refused nested
//!   calls a parent gets no more answers but a refusal, so a tool cannot loop
//!   on free refusals (a refused call releases its budget slot).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{FutureExt, StreamExt};
use futures::channel::{mpsc, oneshot};
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use serde_json::{Value, json};

use super::model_call::ToolCallBase;
use super::tools::PreparedToolCall;
use super::*;
use crate::tool::{NestedToolRunner, ToolEffectStatus, provider_schema};
use tinytools::{ToolCall as CanonicalToolCall, ToolCallId};

/// Refused nested calls one parent call may make before every further call is
/// refused outright.
const MAX_NESTED_REFUSALS: usize = 8;
/// Longest `nested_calls` summary kept on a parent result's metadata.
const MAX_NESTED_SUMMARIES: usize = 32;
/// Longest serialized arguments kept in one summary entry, in bytes.
const MAX_SUMMARY_ARGS_BYTES: usize = 1024;

/// A nested call as the tool sent it.
struct NestedRequest {
    name: String,
    arguments: Value,
    reply: oneshot::Sender<Result<tinytools::ToolResult>>,
}

/// The runner a tool sees: forwards each call to the loop that drives it.
struct ChannelRunner {
    requests: mpsc::UnboundedSender<NestedRequest>,
}

impl NestedToolRunner for ChannelRunner {
    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        arguments: Value,
    ) -> BoxFuture<'a, Result<tinytools::ToolResult>> {
        Box::pin(async move {
            let settled = || {
                TinyAgentsError::ToolFailed(format!(
                    "nested call '{name}' could not run: the calling tool call already settled"
                ))
            };
            let (reply, answer) = oneshot::channel();
            self.requests
                .unbounded_send(NestedRequest {
                    name: name.to_string(),
                    arguments,
                    reply,
                })
                .map_err(|_| settled())?;
            answer.await.map_err(|_| settled())?
        })
    }
}

/// One entry of the `nested_calls` summary on a parent result's metadata.
struct NestedSummary {
    id: String,
    name: String,
    status: &'static str,
    duration_ms: u64,
    args: String,
    error: Option<String>,
}

impl NestedSummary {
    fn to_json(&self) -> Value {
        let mut entry = json!({
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "args": self.args,
        });
        if let Some(error) = &self.error {
            entry["error"] = Value::String(error.clone());
        }
        entry
    }
}

/// Everything needed to service the nested calls of one executing call.
pub(super) struct NestedCalls<'a, State: Send + Sync, Ctx: Send + Sync> {
    harness: &'a AgentHarness<State, Ctx>,
    ctx: &'a RunContext<Ctx>,
    state: &'a State,
    /// The call whose tool makes the nested calls.
    parent: CallId,
    /// Nesting level of that call: `0` for a model-issued call.
    level: usize,
    issued: AtomicUsize,
    refused: AtomicUsize,
    summaries: std::sync::Mutex<Vec<NestedSummary>>,
    dropped_summaries: AtomicUsize,
}

impl<'a, State: Send + Sync, Ctx: Send + Sync> NestedCalls<'a, State, Ctx> {
    pub(super) fn new(
        harness: &'a AgentHarness<State, Ctx>,
        ctx: &'a RunContext<Ctx>,
        state: &'a State,
        parent: CallId,
        level: usize,
    ) -> Self {
        Self {
            harness,
            ctx,
            state,
            parent,
            level,
            issued: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
            summaries: std::sync::Mutex::new(Vec::new()),
            dropped_summaries: AtomicUsize::new(0),
        }
    }

    /// Drives `execution` (the parent tool's future) to completion while
    /// servicing the nested calls it makes. Nested calls still in flight when
    /// the parent settles are dropped with it.
    pub(super) async fn drive<F>(&self, execution: F) -> Result<tinytools::ToolResult>
    where
        F: std::future::Future<Output = Result<tinytools::ToolResult>>,
    {
        let (requests, mut incoming) = mpsc::unbounded();
        let runner: Arc<dyn NestedToolRunner> = Arc::new(ChannelRunner { requests });
        let scoped = crate::tool::nested::scope(self.parent.clone(), runner, execution);
        tokio::pin!(scoped);
        let mut in_flight = FuturesUnordered::new();
        loop {
            tokio::select! {
                result = &mut scoped => {
                    // A tool that stopped waiting on a call dropped its half
                    // of the reply channel; give those calls one poll to
                    // observe it and record themselves as abandoned.
                    while let Some(Some(())) = in_flight.next().now_or_never() {}
                    return result;
                }
                Some(request) = incoming.next() => in_flight.push(self.serve(request)),
                Some(()) = in_flight.next(), if !in_flight.is_empty() => {}
            }
        }
    }

    /// Runs one nested call, records its summary, and answers the tool. If the
    /// tool drops its `call_tool` future first, the nested call is dropped with
    /// it (and reports a terminal event).
    async fn serve(&self, request: NestedRequest) {
        let NestedRequest {
            name,
            arguments,
            mut reply,
        } = request;
        let index = self.issued.fetch_add(1, Ordering::SeqCst) + 1;
        let id = CallId::new(format!("{}/{index}", self.parent));
        let args = truncated_json(&arguments);
        let started = std::time::Instant::now();
        let refused_so_far = self.refused.load(Ordering::SeqCst);
        let outcome = if refused_so_far >= MAX_NESTED_REFUSALS {
            Some(Err(TinyAgentsError::ToolFailed(format!(
                "nested call '{name}' refused: tool call '{}' already had {MAX_NESTED_REFUSALS} \
                 nested calls refused",
                self.parent
            ))))
        } else {
            let run = self.harness.run_nested_tool(
                self.state,
                self.ctx,
                &self.parent,
                id.clone(),
                self.level + 1,
                &name,
                arguments,
            );
            tokio::pin!(run);
            tokio::select! {
                (refused, outcome) = &mut run => {
                    if refused {
                        self.refused.fetch_add(1, Ordering::SeqCst);
                    }
                    Some(outcome)
                }
                () = reply.cancellation() => None,
            }
        };
        let (status, error) = match &outcome {
            Some(Ok(result)) if result.is_error => ("error", Some(truncate(&result.output(), 256))),
            Some(Ok(_)) => ("ok", None),
            Some(Err(error)) => ("failed", Some(truncate(&error.to_string(), 256))),
            None => (
                "abandoned",
                Some("the calling tool stopped waiting for the call".to_string()),
            ),
        };
        self.record(NestedSummary {
            id: id.to_string(),
            name,
            status,
            duration_ms: started.elapsed().as_millis() as u64,
            args,
            error,
        });
        if let Some(outcome) = outcome {
            // The tool may have been dropped (timeout, cancellation); nobody
            // left to answer is not an error.
            let _ = reply.send(outcome);
        }
    }

    fn record(&self, summary: NestedSummary) {
        let mut summaries = self
            .summaries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if summaries.len() < MAX_NESTED_SUMMARIES {
            summaries.push(summary);
        } else {
            self.dropped_summaries.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Attaches the capped `nested_calls` summary to the parent's result
    /// metadata (host-only, never rendered into the transcript).
    pub(super) fn attach_summary(&self, result: &mut tinytools::ToolResult) {
        let summaries = self
            .summaries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if summaries.is_empty() {
            return;
        }
        let metadata = result.metadata.get_or_insert_with(|| json!({}));
        let Some(object) = metadata.as_object_mut() else {
            tracing::debug!(
                target: "tinyagents::nested_tools",
                call_id = %self.parent,
                "[nested_tools] parent metadata is not an object; summary not attached"
            );
            return;
        };
        object.insert(
            "nested_calls".to_string(),
            Value::Array(summaries.iter().map(NestedSummary::to_json).collect()),
        );
        let dropped = self.dropped_summaries.load(Ordering::SeqCst);
        if dropped > 0 {
            object.insert("nested_calls_truncated".to_string(), json!(dropped));
        }
    }
}

fn approval_error(name: &str) -> TinyAgentsError {
    TinyAgentsError::ToolFailed(format!(
        "nested call '{name}' requires approval; nested calls cannot be deferred"
    ))
}

fn truncate(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

fn truncated_json(value: &Value) -> String {
    truncate(
        &serde_json::to_string(value).unwrap_or_default(),
        MAX_SUMMARY_ARGS_BYTES,
    )
}

/// Terminal-event guard for a started nested call: if the call's future is
/// dropped before it settles (its parent timed out, was cancelled, or stopped
/// waiting), the call still gets a `ToolFailed`, so every `ToolStarted` has
/// exactly one terminal partner.
struct StartedNested {
    events: crate::events::EventSink,
    call_id: CallId,
    tool_name: String,
    parent: CallId,
    started_at_ms: u64,
    armed: bool,
}

impl StartedNested {
    fn settle(&mut self) {
        self.armed = false;
    }
}

impl Drop for StartedNested {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.events.emit(AgentEvent::ToolFailed {
            call_id: self.call_id.clone(),
            tool_name: self.tool_name.clone(),
            started_at_ms: Some(self.started_at_ms),
            duration_ms: Some(crate::ids::now_ms().saturating_sub(self.started_at_ms)),
            error: "parent settled: the nested call was dropped before completing".to_string(),
            parent_call_id: Some(self.parent.clone()),
        });
    }
}

type Admitted<State, Ctx> = (Arc<dyn crate::tool::ToolDispatch<State, Ctx>>, ToolCall);

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Admits and executes one nested call of `parent`, as nesting `level`.
    ///
    /// Returns whether the call was *refused* before it ran (it counts toward
    /// [`MAX_NESTED_REFUSALS`]) alongside the outcome. The nested counterpart
    /// of `admit_tool_call` + the execution half of `execute_tool_serially`;
    /// see the module docs for what it shares with the model-issued path.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_nested_tool(
        &self,
        state: &State,
        ctx: &RunContext<Ctx>,
        parent: &CallId,
        call_id: CallId,
        level: usize,
        name: &str,
        arguments: Value,
    ) -> (bool, Result<tinytools::ToolResult>) {
        let call = ToolCall::new(call_id.to_string(), name.to_string(), arguments);
        let (dispatch, call) = match self.admit_nested(state, ctx, parent, level, call).await {
            Ok(admitted) => admitted,
            Err(error) => {
                tracing::debug!(
                    target: "tinyagents::nested_tools",
                    parent = %parent,
                    call_id = %call_id,
                    tool = name,
                    %error,
                    "[nested_tools] nested call refused at admission"
                );
                return (true, Err(error));
            }
        };

        let options = dispatch.call_options(&call.arguments);
        let captured_input = self.policy.capture.tool_io.then(|| call.arguments.clone());
        let started_at_ms = crate::ids::now_ms();
        let prepared = PreparedToolCall {
            call_id: call_id.clone(),
            tool_name: name.to_string(),
            call: call.clone(),
            options,
            captured_input: captured_input.clone(),
            started_at_ms,
            executed: true,
            output_origin: dispatch.output_origin(),
        };
        ctx.emit(AgentEvent::ToolStarted {
            call_id: call_id.clone(),
            tool_name: name.to_string(),
            input: captured_input.clone(),
            parent_call_id: Some(parent.clone()),
        });
        let mut guard = StartedNested {
            events: ctx.events.clone(),
            call_id: call_id.clone(),
            tool_name: name.to_string(),
            parent: parent.clone(),
            started_at_ms,
            armed: true,
        };
        // A durable row per nested call, so recovery sees the effects a parent
        // had through `call_tool`, not only the parent's own row.
        if let Err(error) = self
            .record_tool_effect_started(ctx, &call.arguments, &prepared)
            .await
        {
            guard.settle();
            ctx.limits.release_nested_tool_call();
            return (true, Err(self.fail_nested(ctx, &prepared, parent, error)));
        }
        let base = ToolCallBase {
            harness: self,
            dispatch,
            options,
            timeout_settings: self.tool_timeouts.clone(),
            level,
        };
        let execution = futures::FutureExt::map(
            self.middleware
                .run_wrapped_tool(ctx, state, call.clone(), &base),
            |result| result.map(|wrapped| wrapped.into_result_with_control()),
        );
        let outcome = Self::with_call_budget(
            self.call_budget(ctx),
            ctx.run_id().as_str(),
            "nested tool call",
            super::model_call::RUN_BOUND_LABEL,
            execution,
        )
        .await;
        guard.settle();
        let duration_ms = crate::ids::now_ms().saturating_sub(started_at_ms);

        match outcome {
            Ok((result, control)) => {
                if control.is_some() {
                    tracing::debug!(
                        target: "tinyagents::nested_tools",
                        call_id = %call_id,
                        tool = name,
                        "[nested_tools] a wrap middleware's control request on a nested call is ignored"
                    );
                }
                self.record_tool_effect_settled(ctx, &prepared, ToolEffectStatus::Completed)
                    .await;
                self.middleware
                    .run_observe_nested_result(ctx, state, &call, &result)
                    .await;
                let output = result.output_for_llm(options.prefer_markdown);
                let output_bytes = output.len() as u64;
                ctx.emit(AgentEvent::ToolCompleted {
                    call_id,
                    tool_name: name.to_string(),
                    started_at_ms: Some(started_at_ms),
                    input: captured_input,
                    output: self
                        .policy
                        .capture
                        .tool_io
                        .then(|| Value::String(output.clone())),
                    duration_ms: Some(duration_ms),
                    output_bytes: Some(output_bytes),
                    error: result.is_error.then_some(output),
                    metadata: result.metadata.clone(),
                    parent_call_id: Some(parent.clone()),
                });
                (false, Ok(result))
            }
            Err(error) => {
                // A tool the nested call reached may itself ask to be deferred;
                // the parent cannot pause, so that is the same refusal.
                let error = match error {
                    TinyAgentsError::ApprovalRequired { .. }
                    | TinyAgentsError::CallDeferred { .. } => approval_error(name),
                    other => other,
                };
                self.record_tool_effect_settled(ctx, &prepared, ToolEffectStatus::Failed)
                    .await;
                (false, Err(self.fail_nested(ctx, &prepared, parent, error)))
            }
        }
    }

    /// Emits the terminal `ToolFailed` of a started nested call.
    fn fail_nested(
        &self,
        ctx: &RunContext<Ctx>,
        prepared: &PreparedToolCall,
        parent: &CallId,
        error: TinyAgentsError,
    ) -> TinyAgentsError {
        ctx.emit(AgentEvent::ToolFailed {
            call_id: prepared.call_id.clone(),
            tool_name: prepared.tool_name.clone(),
            started_at_ms: Some(prepared.started_at_ms),
            duration_ms: Some(crate::ids::now_ms().saturating_sub(prepared.started_at_ms)),
            error: error.to_string(),
            parent_call_id: Some(parent.clone()),
        });
        error
    }

    /// Depth, cancellation, deadline and budget checks, then admission. Holds
    /// a budget slot on success; releases it on a refusal.
    async fn admit_nested(
        &self,
        state: &State,
        ctx: &RunContext<Ctx>,
        parent: &CallId,
        level: usize,
        call: ToolCall,
    ) -> Result<Admitted<State, Ctx>> {
        let name = call.name.clone();
        let max_depth = self.policy.limits.max_nested_depth;
        if level > max_depth {
            return Err(TinyAgentsError::ToolFailed(format!(
                "nested call '{name}' exceeds max_nested_depth ({max_depth})"
            )));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(TinyAgentsError::Cancelled);
        }
        if ctx.limits.check_wall_clock().is_err() {
            ctx.emit(AgentEvent::LimitReached {
                kind: LimitKind::WallClock,
            });
            return Err(TinyAgentsError::Timeout(format!(
                "run `{}` exceeded its wall-clock deadline",
                ctx.run_id()
            )));
        }
        if let Err(error) = ctx.limits.try_reserve_nested_tool_call() {
            ctx.emit(AgentEvent::LimitReached {
                kind: LimitKind::ToolCalls,
            });
            return Err(error);
        }
        match self.admit_nested_tool(state, ctx, parent, call).await {
            Ok(admitted) => Ok(admitted),
            Err(error) => {
                // A refused call never ran: give the slot back.
                ctx.limits.release_nested_tool_call();
                Err(error)
            }
        }
    }

    /// Lookup, argument preparation and validation, the approval refusal,
    /// middleware admission checks and host authorization for one nested call.
    async fn admit_nested_tool(
        &self,
        state: &State,
        ctx: &RunContext<Ctx>,
        parent: &CallId,
        mut call: ToolCall,
    ) -> Result<Admitted<State, Ctx>> {
        let name = call.name.clone();
        let allowed_tools = self.resolve_tool_allowlist(ctx)?;
        let is_allowed = allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.contains(&name));
        let Some(dispatch) = is_allowed
            .then(|| self.tools.model_dispatch(&name))
            .flatten()
        else {
            return Err(TinyAgentsError::ToolNotFound(name));
        };
        let tool = dispatch.tool();

        // Same ordering rule as `admit_tool_call`: strip host-injected keys,
        // inject the authoritative values, then validate the model-facing
        // schema. Host authorization below sees the arguments as the tool sent
        // them.
        let model_arguments = call.arguments.clone();
        let canonical_call = CanonicalToolCall::new(
            ToolCallId::new(call.id.clone()),
            call.name.clone(),
            call.arguments.clone(),
        );
        let injected_values = dispatch.injected_arguments(&canonical_call).map_err(|_| {
            TinyAgentsError::Validation(format!(
                "failed to prepare injected arguments for tool `{name}`"
            ))
        })?;
        let injected_declarations = tool.injected_arguments();
        call.arguments = if injected_declarations.is_empty() {
            canonical_call.arguments.clone()
        } else {
            tinytools::prepare_tool_arguments(
                &canonical_call,
                &injected_declarations,
                &injected_values,
            )
            .map_err(|error| {
                TinyAgentsError::ToolFailed(format!(
                    "invalid arguments for nested call '{name}': {error}"
                ))
            })?
        };
        let schema = provider_schema(tool.as_ref());
        if matches!(
            self.policy.invalid_args,
            InvalidArgsPolicy::NormalizeThenReturnToolError
        ) {
            super::tools::normalize_tool_arguments(&mut call, &schema);
        }
        schema.validate_call(&call).map_err(|error| {
            TinyAgentsError::ToolFailed(format!(
                "invalid arguments for nested call '{name}': {error}"
            ))
        })?;

        if crate::tool::is_external_tool(tool.as_ref()) || tool.policy().access.approval_required {
            return Err(approval_error(&name));
        }

        // The enforcement `before_tool` would have applied, over `&RunContext`.
        if let Err(error) = self
            .middleware
            .run_check_nested_tool(ctx, state, &call)
            .await
        {
            return Err(match error {
                TinyAgentsError::ToolFailed(_)
                | TinyAgentsError::Cancelled
                | TinyAgentsError::Timeout(_) => error,
                TinyAgentsError::ApprovalRequired { .. }
                | TinyAgentsError::CallDeferred { .. }
                | TinyAgentsError::Interrupted { .. } => approval_error(&name),
                other => TinyAgentsError::ToolFailed(format!(
                    "nested call '{name}' refused: {other}"
                )),
            });
        }

        if let Some(binding) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? {
            let request = crate::host::ToolCallRequest::new(
                name.clone(),
                model_arguments,
                binding.agent_id.clone(),
            )
            .with_call_id(CallId::new(call.id.clone()))
            .with_parent_call_id(parent.clone());
            let authorization = binding.host.security.authorize_tool(&request);
            let decision = ctx
                .bounded(self.call_budget(ctx), authorization, || {
                    format!(
                        "tool authorization for run `{}` exceeded its remaining wall-clock budget",
                        ctx.run_id()
                    )
                })
                .await?;
            if !decision.is_allowed() {
                let reason = decision
                    .denial_reason()
                    .unwrap_or("tool call was not approved")
                    .to_string();
                return Err(TinyAgentsError::ToolFailed(format!(
                    "nested call '{name}' denied: {reason}"
                )));
            }
        }
        Ok((dispatch, call))
    }
}
