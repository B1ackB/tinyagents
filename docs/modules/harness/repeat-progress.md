# Repeat-progress guard (`RepeatProgressMiddleware`)

Catches loops that *succeed* but go nowhere, as the complement to the
repeated-failure ladder in `no_progress/`. It uses `before_tool` (admission),
`after_model` and `after_tool`, plus the companion `RepeatEvictionObserver`
registered after every context-reduction step.

Escalation is staged. The first threshold used to halt the run outright; it now
only warns, and the run halts on the second block:

| Stage | When (defaults) | Effect |
| --- | --- | --- |
| Warn | the same call returns the identical result 3 times, or an identical call batch repeats 3 times in a row, or identical output repeats 4 times | A `[repeat notice]` is appended to that tool result telling the model to change approach. Once per signature. |
| Block | the identical call is attempted a 5th time (warn + `block_after_warn` = 2) | `before_tool` refuses the call with `TinyAgentsError::ToolFailed`; the loop answers it with an error result asking the model to reassess and never runs the tool. This uses the existing admission-refusal path, so it does not force serial tool execution the way a `ToolMiddleware` would. |
| Halt | a second block *of the same call* (`blocks_before_halt` = 2) | The cause goes into the `HaltSummarySlot` and the run pauses through the steering handle, as before. Blocks are counted per call signature, so two different repeating calls in one batch are each blocked once first; the count survives context compaction and clears when the call returns a new result. |

Batch and output streaks (results that differ, so there is nothing to block)
warn at their threshold and halt `block_after_warn` repeats later.

Two warning-only detectors feed the same notes path and never block or halt:

- **Ping-pong**: two call signatures alternating (A,B,A,B,A,B; 6 calls) with a
  stable result on each side.
- **Argument churn**: one tool with at least 3 distinct argument variants, each
  called at least 3 times with the same stable result.

**Post-compaction guard** (warning-only): the last 3 `(call, result)` pairs
that were already repeating (2+ identical results) before a compaction are
remembered. If one of the next 3 calls repeats such a pair, the model gets a
note. One re-read of evicted content is correct behaviour and is neither warned
about nor blocked.

**Blocks are predictions.** A block assumes the call would return what it
returned last time. After any successful call that is not read-only, the
predictions for *other* calls are discarded, so a read repeated after an edit
runs. Tell the middleware which tools are pure reads with `with_read_only`
(from `ToolPolicy::read_only`); without it every tool counts as possibly
state-changing and a repeat blocks only while it is the most recent call.

**Marker.** Results the guard answers itself carry
`"tinyagents.repeat_guard": "blocked" | "halted"` in `ToolResult::metadata`
(host-only, never shown to the model; `repeat_guard_marker`). It is stamped
when the call is refused (`RunContext::set_refusal_metadata`), so every
`after_tool` hook sees it regardless of registration order. A host's
repeated-failure middleware should skip them.

At most one warning lands on a result, its own first. Extras (which name their
tool) wait for the next result and are dropped on a context compaction.

Configure all of it with `RepeatProgressConfig` (`with_config`); defaults are
`RepeatProgressConfig::default()`. `RepeatProgressConfig::immediate_halt()`
restores the historical behaviour: halt at the first threshold, no notes, no
blocking, no extra detectors. Polling tools (`RepeatExemption`) are exempt from
every stage.
