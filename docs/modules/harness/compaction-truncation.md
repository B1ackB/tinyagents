# Compaction: tool-result truncation

`with_tool_result_truncation(max_bytes)` (opt-in; unset never truncates) adds a
route in front of summarization. `CompactionPressure::route(prompt, budget,
reducible)` returns:

| Route | When | Action |
| --- | --- | --- |
| `Fits` | `prompt <= budget` | nothing |
| `Compact` | over budget, nothing reducible | summarize |
| `TruncateToolResults` | enough reducible tool-result content | cut results only |
| `CompactThenTruncate` | reducible, but not enough alone | summarize, then cut |

`reducible` is the bytes above `max_bytes` in every tool-result text block.
The cut keeps the head and appends the standard `truncated by
tool_result_budget` notice; it skips `trusted_verbatim` results, non-text
blocks and `[tool_result_preview]` envelopes, and is idempotent.

Truncation runs before the call once the prompt is over the trigger, and after
a reported overflow when the route says it can help. The provider's word
outranks the estimate, so a `Fits` verdict on a reported overflow still
compacts. A route that truncates switches the run into truncating mode: later
requests of that run have oversized results cut again before measurement, so
the measured prompt stays valid. The cut spares results after the last
assistant message, which is what the model just asked for.

A compaction summarizes the uncut results; the cut is applied on top when
sending. In truncating mode `wrap_model` sees an already-cut request, which no
longer fingerprint-aligns with the live transcript, so an overflow compaction
in that state is not persisted as a boundary (it still retries).

## Overflow reported by a successful response

Some providers report an oversized prompt through usage rather than an error.
`with_response_overflow_detection` controls this opt-in behaviour:

| Mode | Counts as overflow |
| --- | --- |
| `Off` (default) | nothing (errors only) |
| `Usage` | usage above the window, or a full-window zero-output `length` stop |
| `UsageAndShortLength` | the above plus a short `length` stop |

The window comes from response usage or `SummarizationPolicy`. Cache-served and
streamed responses are never discarded. A discarded response's usage is still
recorded in run totals and host budgets, and an `overflow_discarded_response`
custom event marks it in the event stream. If a retry fails, the billed usage
is flushed before the run returns the error.
