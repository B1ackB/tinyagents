//! Host budget admission for one model call.
//!
//! Split out of `run_loop.rs`. An explicit host-driven run may carry a
//! [`crate::host::BudgetGate`]; each model call acquires a permit before it is
//! dispatched and the realised usage is recorded afterwards.

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Acquires the host budget permit for the call about to be dispatched.
    ///
    /// The permit is acquired after structured-output planning: a synthetic
    /// schema tool is part of the provider request and must be included in its
    /// estimate. Returns `None` for a run with no host budget. The permit stays
    /// alive through response accounting, so cancellation or a provider error
    /// still releases it through `Drop`.
    pub(super) async fn admit_host_budget(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &AgentRun,
        request: &mut ModelRequest,
        profile: Option<&tinyinference_llm::model::ModelProfile>,
        model_name: &str,
        offered_tool_count: usize,
    ) -> Result<Option<(Arc<dyn crate::host::BudgetGate>, crate::host::Permit)>> {
        Ok(
            if let Some(host_run) = crate::runtime::host_invocation_binding::<State, Ctx>(ctx)? {
                if let Some(budget) = host_run.host.budget.clone() {
                    let context_state = crate::host::ContextState {
                        message_count: request.messages.len(),
                        prompt_tokens: crate::token_estimation::estimate_slice_tokens(
                            &request.messages,
                        ),
                        context_window_tokens: profile.and_then(|profile| profile.max_input_tokens),
                        iterations: run.steps,
                    };
                    let hint = budget.compression_hint(&context_state);
                    if hint.is_advised() {
                        tracing::debug!(?hint, "[host] budget gate advised context compression");
                        apply_host_budget_compression(ctx, &mut request.messages, hint)?;
                    }
                    let estimate = crate::host::CallEstimate::new(
                        model_name,
                        crate::token_estimation::estimate_slice_tokens(&request.messages),
                        request.max_tokens.unwrap_or_default() as u64,
                    )
                    .with_agent(host_run.agent_id.clone())
                    .with_thread(
                        ctx.thread_id()
                            .cloned()
                            .unwrap_or_else(|| ctx.run_id().as_str().into()),
                    )
                    .with_tool_count(offered_tool_count);
                    let permit = ctx
                        .bounded(self.call_budget(ctx), budget.acquire(&estimate), || {
                            format!(
                                "budget admission for run `{}` exceeded its remaining wall-clock deadline",
                                ctx.run_id()
                            )
                        })
                        .await?;
                    Some((budget.clone(), permit))
                } else {
                    None
                }
            } else {
                None
            },
        )
    }

    /// Records realised provider usage without allowing host accounting I/O to
    /// outlive a cancelled or deadline-expired run. The local run totals are
    /// updated before this call, so a host-recording failure never erases spend
    /// that the provider has already incurred.
    pub(super) async fn record_host_usage(
        &self,
        ctx: &RunContext<Ctx>,
        budget: &Arc<dyn crate::host::BudgetGate>,
        usage: &tinyinference_llm::usage::Usage,
    ) -> Result<()> {
        let recording = budget.record(usage);
        ctx.bounded(self.call_budget(ctx), recording, || {
            format!(
                "budget usage recording for run `{}` exceeded its remaining wall-clock deadline",
                ctx.run_id()
            )
        })
        .await
    }
}

/// Applies a host budget hint before the provider sees the request.
///
/// This deliberately uses the harness's pairing-safe generic context reducer
/// instead of a host-specific transcript rewrite. `Soft` preserves the most
/// recent half of a multi-turn conversation (and all system messages) when it
/// can make progress. `Hard` uses a token budget and refuses a request that
/// cannot be reduced without discarding its whole conversational payload.
/// Both outcomes are observable through the canonical `context.compressed`
/// event so hosts can correlate a budget decision with the actual request.
fn apply_host_budget_compression<Ctx>(
    ctx: &mut RunContext<Ctx>,
    messages: &mut Vec<Message>,
    hint: crate::host::CompressionHint,
) -> Result<()> {
    use crate::host::CompressionHint;
    use crate::summarization::{TrimStrategy, trim_messages};

    let from_tokens = crate::token_estimation::estimate_slice_tokens(messages);
    let non_system = messages
        .iter()
        .filter(|message| !matches!(message, Message::System(_)))
        .count();
    let reduced = match hint {
        CompressionHint::None => return Ok(()),
        // Preserve a recent working window without perturbing a short prompt.
        CompressionHint::Soft if non_system < 3 => return Ok(()),
        CompressionHint::Soft => {
            trim_messages(messages, &TrimStrategy::KeepLast((non_system / 2).max(1)))
        }
        // A hard hint must create real headroom without ever treating system
        // instructions as expendable. The token trimmer removes oldest
        // conversational messages, preserves every system message verbatim,
        // and clears an orphaned tool-result prefix after its owning assistant
        // call was evicted.
        CompressionHint::Hard => crate::summarization::trim_messages_to_token_budget_with(
            messages,
            crate::summarization::TokenTrimPolicy::strict((from_tokens / 2).max(1))
                .preserve_system()
                .drop_leading_orphan_tools(),
            crate::token_estimation::estimate_message_tokens,
        ),
    };
    let to_tokens = crate::token_estimation::estimate_slice_tokens(&reduced);
    let has_conversation = reduced
        .iter()
        .any(|message| !matches!(message, Message::System(_)));
    if to_tokens >= from_tokens || reduced.is_empty() || (hint.is_required() && !has_conversation) {
        if hint.is_required() {
            return Err(TinyAgentsError::Validation(
                "host budget requires reducible conversational context before provider call".into(),
            ));
        }
        return Ok(());
    }
    *messages = reduced;
    ctx.emit(AgentEvent::Compressed {
        from_tokens,
        to_tokens,
    });
    Ok(())
}
