//! Post-response recovery for the superstep loop: what to do with a model
//! response that cannot be taken at face value.
//!
//! Split out of `run_loop.rs`. Two stages run on every response, both bounded
//! by the counters in [`TurnRecovery`]:
//!
//! 1. [`AgentHarness::reject_truncated_tool_calls`] — a length stop that may
//!    have cut a tool call mid-arguments: answer the suspect calls with an
//!    error instead of running them and let the model retry.
//! 2. [`AgentHarness::recover_unusable_response`] — a response with no tool
//!    call that is not a usable answer (a withheld call, a truncated-empty or
//!    empty reply, a dropped or undecodable call): re-issue or re-prompt.

use super::run_loop::{
    DROPPED_TOOL_CALL_NUDGE, TRUNCATED_EMPTY_ANSWER_NUDGE, TRUNCATED_EMPTY_TOOL_NUDGE,
    UNDECODABLE_TOOL_CALL_NUDGE, WITHHELD_TOOL_CALL_NUDGE, truncated_call_positions,
};
use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// A length stop means the output cap cut the reply off somewhere: the LAST
    /// call of the response may carry truncated (yet parseable) arguments —
    /// native or recovered from text alike, since a text grammar can close an
    /// open `{`/`[` or run a payload to end-of-text — as may any call the
    /// provider flagged invalid (a repair could make it look whole). Answer
    /// those with an error instead of running them and let the model retry;
    /// every earlier call was finished before the cut and runs normally. The
    /// call is chosen by position, not id, so duplicate or empty provider ids
    /// fail closed. Bounded per logical turn. A run resumed in a fresh context
    /// restarts this budget.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn reject_truncated_tool_calls(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        turn_recovery: &mut TurnRecovery,
        turn: &ResponseTurn<'_>,
    ) -> Result<TruncationOutcome> {
        let ResponseTurn {
            call_id,
            response,
            tool_calls,
            attempt_max_tokens,
            structured_call_names,
            ..
        } = *turn;
        ctx.truncated_call_positions.clear();
        if self.policy.reject_truncated_tool_calls
            && !tool_calls.is_empty()
            && crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
        {
            let truncated_positions = truncated_call_positions(tool_calls);
            if !truncated_positions.is_empty() {
                if turn_recovery.truncated_tool_call_retries_used
                    >= self.policy.truncated_tool_call_retries
                {
                    tracing::warn!(
                        target: "tinyagents::agent_loop",
                        run_id = %ctx.run_id(),
                        call_id = %call_id,
                        retries = turn_recovery.truncated_tool_call_retries_used,
                        "[agent_loop] length-truncated tool calls keep recurring; truncated-tool-call retry budget exhausted"
                    );
                    messages.pop();
                    return Err(TinyAgentsError::LimitExceeded(format!(
                        "run `{}` stopped: {} consecutive \
                             retries of a tool call truncated by the output token limit did not \
                             produce a complete call (RunPolicy::truncated_tool_call_retries)",
                        ctx.run_id(),
                        turn_recovery.truncated_tool_call_retries_used
                    )));
                }
                turn_recovery.truncated_tool_call_retries_used += 1;
                // Give the retry room to finish the call.
                turn_recovery.boost_max_tokens(attempt_max_tokens);
                tracing::info!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    call_id = %call_id,
                    calls = tool_calls.len(),
                    rejected = truncated_positions.len(),
                    attempt = turn_recovery.truncated_tool_call_retries_used,
                    max_tokens = ?turn_recovery.boosted_max_tokens,
                    finish_reason = ?response.finish_reason,
                    "[agent_loop] length-truncated response; failing its possibly-incomplete tool calls instead of running them"
                );
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "truncated_tool_calls".to_string(),
                    detail: format!(
                        "model call `{call_id}` hit its output limit mid-turn; {} of {} tool \
                             call(s) answered with an error, not run",
                        truncated_positions.len(),
                        tool_calls.len()
                    ),
                });
                status.set_last_event(record.id);
                if truncated_positions
                    .iter()
                    .any(|&index| structured_call_names.contains(&tool_calls[index].name))
                {
                    // The structured-output call itself was cut off: it
                    // cannot be extracted as the answer, and the turn has
                    // no other path that answers it. Fail the whole turn.
                    status.mark_running(HarnessPhase::Tools);
                    self.fail_truncated_tool_calls(state, ctx, run, status, messages, tool_calls)
                        .await?;
                    self.apply_queued_lane(
                        ctx,
                        status,
                        messages,
                        crate::run_queue::QueueLane::Steer,
                    )
                    .await;
                    return Ok(
                        match self.apply_pending_control(ctx, run, status, messages)? {
                            ControlEffect::Exit(exit) => TruncationOutcome::EndTurn(Some(exit)),
                            ControlEffect::None | ControlEffect::ContinueLoop => {
                                TruncationOutcome::EndTurn(None)
                            }
                        },
                    );
                }
                // Admission answers these calls with the error, in call
                // order, as the batch runs (see `admit_tool_call`). The
                // batch holds the non-structured calls only, so translate
                // each position into the batch's own index space.
                let mut batch_index = 0;
                for (index, call) in tool_calls.iter().enumerate() {
                    if structured_call_names.contains(&call.name) {
                        continue;
                    }
                    if truncated_positions.contains(&index) {
                        ctx.truncated_call_positions.insert(batch_index);
                    }
                    batch_index += 1;
                }
                return Ok(TruncationOutcome::CallsRejected);
            }
        }
        Ok(TruncationOutcome::Clean)
    }

    /// Recovery for a response with no real tool call that is not yet a usable
    /// answer. Returns `true` when a retry or re-prompt was scheduled and the
    /// loop must run another turn; `false` when the response stands and the
    /// caller goes on to resolve the turn.
    ///
    /// The checks run in a fixed order: a withheld call, a truncated-empty
    /// retry, a truncated-empty nudge, a non-truncated empty retry, and a
    /// dropped or undecodable call nudge.
    pub(super) fn recover_unusable_response(
        &self,
        ctx: &mut RunContext<Ctx>,
        run: &AgentRun,
        status: &mut HarnessRunStatus,
        messages: &mut Vec<Message>,
        turn_recovery: &mut TurnRecovery,
        turn: &ResponseTurn<'_>,
    ) -> bool {
        let ResponseTurn {
            call_id,
            response,
            tool_calls,
            attempt_max_tokens,
            recovery,
            tools_available: tools_available_this_turn,
            text_dialect_calls_recoverable,
            has_structured_plan,
            ..
        } = *turn;
        // A call written on a turn that could not take one (tools
        // withdrawn for a concluding answer, or `ToolChoice::None`).
        // It was scrubbed and not run; what is left is either nothing
        // or a lead-in to a step that never happened, so it is not the
        // answer the request asked for. Drop that row and ask once
        // more, telling the model plainly that tools are gone.
        // Replaying the bench request that leaked (DeepSeek V4, tools
        // withdrawn), the unchanged request leaked 6 times in 8; with
        // the row dropped and this re-prompt added it leaked 0 times
        // in 12. Runs before the empty-reply retries: a bare re-send
        // of the same transcript leaks the same way.
        let withheld_calls = recovery.dropped.withheld();
        if withheld_calls > 0
            && turn_recovery.withheld_call_nudges_used < self.policy.dropped_tool_call_nudges
            && ctx.limits.remaining_model_calls() > 0
        {
            turn_recovery.withheld_call_nudges_used += 1;
            messages.pop();
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                withheld_calls,
                attempt = turn_recovery.withheld_call_nudges_used,
                "[agent_loop] re-prompting after a tool call on a turn with no callable tools"
            );
            ctx.emit(AgentEvent::ControlApplied {
                control: "withheld_tool_call".to_string(),
                detail: format!(
                    "{withheld_calls} tool call(s) written while no tool was callable \
                             in model call `{call_id}`; scrubbed, not run, re-prompted"
                ),
            });
            messages.push(Message::user(WITHHELD_TOOL_CALL_NUDGE));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.withheld_call_nudges_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // Truncated-empty recovery (runs before structured extraction,
        // which would otherwise fail on the empty completion). A local
        // reasoning model can burn the whole token budget on its hidden
        // reasoning channel and return `finish_reason == "length"` with
        // no visible text, no tool calls, and no structured output — a
        // result useless to every caller. Retry the call (bumping the
        // token budget when one was set) instead of surfacing the blank.
        // A structured tool hit carries a real payload, so it is never
        // treated as truncated-empty.
        let truncated_empty = tool_calls.is_empty()
            && crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
            && response.text().trim().is_empty();
        if truncated_empty
            && turn_recovery.truncated_empty_retries_used < self.policy.truncated_empty_retries
            && ctx.limits.remaining_model_calls() > 0
        {
            // Drop the useless empty assistant row appended above so the
            // retry re-sends the identical transcript.
            messages.pop();
            turn_recovery.truncated_empty_retries_used += 1;
            // Grow the token budget when the request set one: double it,
            // clamped at 4x the original cap. An unset budget stays unset
            // (a plain retry is still worthwhile — the failure is
            // stochastic).
            turn_recovery.boost_max_tokens(attempt_max_tokens);
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.truncated_empty_retries_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // The boosted retry is spent and the model still deliberated
        // past its output budget. Re-sending the same transcript keeps
        // failing the same way (a high-effort reasoning model thinks
        // as long as it is allowed to), and finishing here hands the
        // host a blank reply it can only close as if the work were
        // done. Say plainly what happened and ask for the next step,
        // then carry on with the loop. The boosted cap stays in force.
        if truncated_empty
            && turn_recovery.truncated_empty_nudges_used < self.policy.truncated_empty_nudges
            && ctx.limits.remaining_model_calls() > 0
        {
            messages.pop();
            turn_recovery.truncated_empty_nudges_used += 1;
            let nudge = if tools_available_this_turn {
                TRUNCATED_EMPTY_TOOL_NUDGE
            } else {
                TRUNCATED_EMPTY_ANSWER_NUDGE
            };
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                attempt = turn_recovery.truncated_empty_nudges_used,
                tools_available = tools_available_this_turn,
                max_tokens = ?turn_recovery.boosted_max_tokens.or(attempt_max_tokens),
                "[agent_loop] truncated-empty retries spent; nudging model to act"
            );
            ctx.emit(AgentEvent::ControlApplied {
                control: "truncated_empty_nudge".to_string(),
                detail: format!(
                    "model call `{call_id}` ran out of output tokens while reasoning \
                             after {} retry(ies); re-prompted to act",
                    turn_recovery.truncated_empty_retries_used
                ),
            });
            messages.push(Message::user(nudge));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: (turn_recovery.truncated_empty_retries_used
                    + turn_recovery.truncated_empty_nudges_used) as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // A provider can also finish normally after sending only a
        // reasoning side channel (or no content at all). Retrying that
        // unusable answer is opt-in because it incurs another provider
        // call. Unlike a length-truncated reply, keep the same token
        // cap: there is no evidence that output space ran out.
        let nontruncated_empty = tool_calls.is_empty()
            && response.text().trim().is_empty()
            && response.continue_turn.is_none()
            && !has_structured_plan
            && run.structured.is_none()
            && !crate::finish_reason::is_length_stop(response.finish_reason.as_deref())
            && response.finish_reason.as_deref() != Some("tool_calls")
            && !response.served_from_cache;
        if nontruncated_empty
            && turn_recovery.empty_response_retries_used < self.policy.empty_response_retries
            && ctx.limits.remaining_model_calls() > 0
        {
            messages.pop();
            turn_recovery.empty_response_retries_used += 1;
            tracing::info!(
                target: "tinyagents::agent_loop",
                run_id = %ctx.run_id(),
                call_id = %call_id,
                attempt = turn_recovery.empty_response_retries_used,
                finish_reason = ?response.finish_reason,
                content_blocks = response.message.content.len(),
                "[agent_loop] retrying completion without visible answer"
            );
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.empty_response_retries_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        // Dropped tool call: the provider says the model stopped to
        // call a tool, but nothing arrived — structured or in text.
        // A bounded re-prompt asks for the call itself. The assistant
        // row stays on the transcript so the model sees what it did.
        //
        // A text dialect always finishes with `stop`, so its dropped
        // call is a block a grammar recognised that became no call:
        // one whose body did not decode (scrubbed, only the lead-in
        // prose left), or one the model stopped inside without a
        // closer. A `length` stop inside a block is truncation, not a
        // forgotten closer, and is left to the truncation handling.
        // Native models with text recovery on parse the same grammars
        // out of their prose, so the same drop applies to them.
        let malformed_blocks = recovery.dropped.malformed();
        let unterminated_blocks =
            if crate::finish_reason::is_length_stop(response.finish_reason.as_deref()) {
                0
            } else {
                recovery.dropped.unterminated()
            };
        let undecodable_text_call =
            text_dialect_calls_recoverable && malformed_blocks + unterminated_blocks > 0;
        if tool_calls.is_empty()
            && (response.finish_reason.as_deref() == Some("tool_calls") || undecodable_text_call)
            && tools_available_this_turn
            && turn_recovery.dropped_tool_call_nudges_used < self.policy.dropped_tool_call_nudges
        {
            turn_recovery.dropped_tool_call_nudges_used += 1;
            let nudge = if undecodable_text_call {
                tracing::info!(
                    target: "tinyagents::agent_loop",
                    run_id = %ctx.run_id(),
                    call_id = %call_id,
                    malformed_blocks,
                    unterminated_blocks,
                    attempt = turn_recovery.dropped_tool_call_nudges_used,
                    "[agent_loop] nudging after undecodable text-dialect tool call"
                );
                UNDECODABLE_TOOL_CALL_NUDGE
            } else {
                DROPPED_TOOL_CALL_NUDGE
            };
            messages.push(Message::user(nudge));
            let record = ctx.emit(AgentEvent::RetryScheduled {
                call_id: call_id.clone(),
                attempt: turn_recovery.dropped_tool_call_nudges_used as usize,
            });
            status.set_last_event(record.id);
            return true;
        }

        false
    }
}
