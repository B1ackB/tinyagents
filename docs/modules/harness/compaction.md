# Compaction: rules, split turns, iterative summaries, overflow recovery

This is the durable, rule-driven layer built on top of
[`summarization.md`](./summarization.md)'s trimming/summarization primitives.
It is the harness's port of pi's `compaction.ts` / `overflow.ts`
(`docs/runtime-comparison/pi.md` §4.5, feature-gap E2): cut points at safe
message boundaries, split-turn summarization, iterative summaries, a durable
`CompactionRecord`, and an overflow → compact → retry recovery loop.

Lives in `crates/tinyagents-harness/src/summarization/compaction.rs`
(`pub mod compaction`, re-exported at `summarization::*`) and
`middleware/library/context.rs` (`ContextCompressionMiddleware`).

## Cut points: `find_cut_point`

```rust
pub struct CutPoint {
    pub index: usize,        // into the non-system message slice
    pub tokens_before: u64,  // estimated tokens folded into a summary
    pub tokens_after: u64,   // estimated tokens kept verbatim
}

pub fn find_cut_point(
    messages: &[Message],
    keep_recent_tokens: u64,
    estimator: impl Fn(&Message) -> u64,
) -> Option<CutPoint>;
```

Walks `messages` newest-first (system messages excluded — they are always
implicitly kept), accumulating `estimator` tokens, until keeping one more
message would exceed `keep_recent_tokens`. The resulting boundary is then
repaired with `pairing::find_safe_cutoff_point`, so the returned cut **never**
splits an assistant tool-call turn from the tool results answering it — the
same invariant `SummarizationPolicy::plan` enforces for its count-based split.
`keep_recent_tokens` is therefore a floor, not an exact budget: the repair may
keep a few extra tokens to preserve pairing.

Returns `None` when there is nothing to cut: no non-system content, or the
whole non-system slice already fits under `keep_recent_tokens`.

`SummarizationPolicy::plan` splits by a fixed `keep_last` *message count*
unless `SummarizationPolicy::keep_recent_tokens` is set, in which case it uses
`find_cut_point` too. `find_cut_point` is also used by
`ContextCompressionMiddleware::wrap_model`'s overflow recovery path, where a
token budget (not a message count) is what needs to shrink under a provider's
context window. With `pin_turn_user_message`, both paths keep the turn's most
recent user message verbatim at the front of the kept tail when the tail has
none of its own.

## Split turns: `summarize_with_split`

```rust
pub async fn summarize_with_split(
    summarizer: &dyn Summarizer,
    messages: &[Message],
    max_turn_tokens: u64,
    previous_summary: Option<String>,
    estimator: impl Fn(&Message) -> u64,
) -> Result<SummaryRecord>;
```

When `messages`' estimated tokens fit under `max_turn_tokens`, this is exactly
`summarizer.summarize_request(&SummaryRequest { messages, previous_summary })`.
When they don't — a turn with a huge tool result, for instance — the slice is
split near its midpoint at a safe tool-pairing boundary
(`pairing::find_safe_cutoff_point`), each half is summarized independently
(`previous_summary` threads to the *first* half only), and the two summaries
are reconciled with `Summarizer::merge`. When no safe interior boundary exists
(the whole turn is a single indivisible tool-call/tool-result pair), the whole
slice is summarized in one call instead of forcing an unsafe split.

`ContextCompressionMiddleware::with_max_turn_tokens` wires `max_turn_tokens`
into both the threshold and overflow compaction paths; leaving it unset
(default) never splits.

## Iterative summaries

`Summarizer::summarize_request(&SummaryRequest) -> Result<SummaryRecord>` is a
default trait method. When `SummaryRequest::previous_summary` is set, the
default prepends it as a system message to the messages it hands `summarize`,
so a summarizer that only implements `summarize` still folds the prior summary
in; with none set it delegates unchanged. Compaction is incremental (below), so
dropping the previous summary would lose everything folded before.

Override `summarize_request` *instead of* relying on that default when the
prior summary belongs somewhere specific in your prompt (an LLM-backed
summarizer that asks the model to *refine* it): the override replaces the
default, so forward it exactly once. `ConcatSummarizer` overrides it to put the
previous summary verbatim ahead of its concatenation.

`ContextCompressionMiddleware` keeps the summary attached to each run's current
fold and passes it as `SummaryRequest::previous_summary` on that run's next
compaction — proactive or overflow-triggered. A summary from an overflow
compaction that could not extend the fold (below) is not kept.

## The fold: compaction carries across calls

The agent loop rebuilds every request from its own working transcript, and
`before_model` only rewrites that outgoing copy, so the loop never sees the
summary. The middleware therefore remembers what it compacted as a *fold*: the
number of leading non-system messages of the live transcript the summary stands
in for, a chained fingerprint of those messages, and the summary itself.

On every call it first re-applies the fold (system messages, then the summary,
then the live messages after the fold), and only then checks the threshold. A
new compaction plans over the unfolded remainder alone and hands the summarizer
just those messages plus the fold's summary as `previous_summary`; the result
replaces the old summary and extends the fold. If the transcript no longer
starts with the fingerprinted messages, the fold is dropped. If the request
carries the fold's summary itself (a host that spliced it into its own
transcript), that summary is lifted out and passed as `previous_summary`, so the
next summary replaces it rather than sitting beside it. Otherwise the history is
some other conversation: compaction starts over with no previous summary. A
host-carried summary keeps being recognized on every call until a compaction
replaces it, and boundaries for that run are not persisted (the host's
transcript no longer maps one-to-one onto the session's).

All of this state is per run, keyed by the context's process-unique
`RunContext::instance_id` (a `RunId` is a caller label two concurrent runs may
share), so invocations sharing one middleware instance never read each other's
fold or summary. `after_agent`
drops a finished run's state; a run that never reaches it is evicted
least-recently-used once 256 runs are tracked.

Without the fold, a long run compacted on every call after the first one and
re-sent the whole, growing history to the summarizer each time: in one 300-call
SWE task, 101 summarizer calls made up about 85% of the cost while the agent
saw only the summary and a few recent messages.

`CompactionRecord::first_kept_index` is in live-transcript coordinates (the fold
length after the compaction), which is what a session-backed sink maps to an
entry id. The overflow path extends the fold the same way, provided the
request's non-system messages still line up one-to-one with the live remainder.
That is checked by fingerprint, not count: when a later middleware dropped or
rewrote messages (a trim, or microcompact blanking tool bodies), the overflow
compaction still runs (without the prior summary, which then stays in the
request), but the fold is not extended and no boundary is persisted: a shifted
index would restore or duplicate the wrong messages on resume. With compression
installed only as model middleware there is no `before_model` state, and the
request is taken as the live transcript.

## `CompactionRecord` and `CompactionSink`

```rust
pub enum CompactionReason { Manual, Threshold, Overflow }

pub struct CompactionRecord {
    pub summary: String,
    pub first_kept_index: usize, // position in the non-system slice, matches CutPoint::index
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub usage: Option<tinyinference_llm::usage::Usage>,
    pub details: serde_json::Value,
    pub reason: CompactionReason,
}

pub trait CompactionSink: Send + Sync {
    fn persist(&self, record: &CompactionRecord) -> Result<()>;
}
```

Every successful compaction — threshold-triggered (`before_model`) or
overflow-triggered (`wrap_model`) — builds a `CompactionRecord` and, when a
`CompactionSink` is attached to the run
(`RunContext::compaction_sink` / `RunContext::with_compaction_sink`),
persists it. `tinyagents-harness` cannot depend on `tinyagents-session` (the
dependency runs the other way), so `CompactionSink` is a trait rather than a
concrete `Arc<EntryTree>` field; the session-backed implementation is
`tinyagents_session::entry_tree::SessionCompactionSink`, which translates
`first_kept_index` into a durable `EntryId` by re-walking the current tip's
ancestor chain and counting only the entries `EntryTree::build_context` itself
turns into messages (`EntryKind::Message`/`EntryKind::Custom`). An index that
doesn't line up with the session's own view of the conversation is skipped
rather than persisted as a corrupting boundary.

`AgentEvent::Compacted { reason, tokens_before, tokens_after }` is emitted for
every compaction, in addition to the pre-existing
`AgentEvent::Compressed { from_tokens, to_tokens }` on the threshold
(`before_model`) path — a listener that only cares whether the transcript
shrank can keep watching `Compressed`; one that cares *why* watches
`Compacted`.

## `before_compaction` hook

```rust
pub struct CompactionContext {
    pub reason: CompactionReason,
    pub tokens_before: u64,
    pub to_summarize_count: usize,
    pub to_keep_count: usize,
}

pub enum CompactionDecision {
    Proceed,
    Decline,
    UseSummary(String),
}
```

`ContextCompressionMiddleware::with_before_compaction` installs a hook of type
`Fn(&CompactionContext) -> CompactionDecision`, consulted before every
compaction this middleware runs (both `before_model` and `wrap_model`).
`Decline` leaves the transcript (and, for the overflow path, the original
provider error) untouched; `UseSummary(text)` skips the configured
`Summarizer` and installs `text` directly.

## Overflow classification: `OverflowClassifier`

```rust
pub struct OverflowInfo { pub requested: Option<u64>, pub limit: Option<u64> }
pub struct OverflowProbe<'a> { pub message: &'a str, pub code: Option<&'a str>, pub status: Option<u16> }

pub struct OverflowClassifier { /* patterns, checked in order */ }
impl OverflowClassifier {
    pub fn empty() -> Self;
    pub fn with_pattern(self, label: &'static str, matcher: impl Fn(&OverflowProbe) -> Option<OverflowInfo> + Send + Sync + 'static) -> Self;
    pub fn classify(&self, error: &TinyAgentsError) -> Option<OverflowInfo>;
}
```

`OverflowClassifier::default()` ships built-in patterns, checked in this
order: `"openai"` (`context_length_exceeded` code or message text),
`"anthropic"` (`"prompt is too long"`), `"llama_cpp"` (local llama.cpp `n_ctx`
messages), `"generic"` (a `"maximum context length"` phrasing several
providers share), `"http_body"` (a fallback for an HTTP 400/413 whose body
mentions the context window). `TinyAgentsError::ContextOverflow` classifies
directly regardless of pattern. `OverflowInfo`'s `requested`/`limit` fields
are extracted best-effort from the error text (never load-bearing for the
classification itself) and may both be `None`. Extend the table with
`with_pattern` for a provider or local server not covered by the defaults.

## `overflow → compact → retry`

`ContextCompressionMiddleware` implements both the lifecycle `Middleware`
trait (`before_model`, the proactive threshold path — unchanged from before
this landed, now backed by `summarize_with_split`/iterative summaries/
`CompactionRecord`) and the around-call `ModelMiddleware` trait (`wrap_model`,
the reactive overflow path). Register the same instance in both slots to get
both behaviors:

```rust
let mw = Arc::new(ContextCompressionMiddleware::new(policy));
stack.push(mw.clone());               // before_model: proactive threshold compaction
stack.push_model_middleware(mw.clone()); // wrap_model: overflow → compact → retry
```

`wrap_model`:

1. Calls the wrapped model once. On success, forwards the response.
2. On error, consults `OverflowClassifier`. A non-overflow error propagates
   unchanged — no compaction, no retry.
3. On a classified overflow, finds a cut point via `find_cut_point` using
   `SummarizationPolicy::trigger_budget()` as `keep_recent_tokens`. No safe
   cut (already-minimal transcript, or a single indivisible tool pair)
   propagates the original error.
4. Consults `before_compaction`. `Decline` propagates the original error.
   `Proceed`/`UseSummary` run the compaction (`CompactionReason::Overflow`),
   persist/emit as above, and retry the **same** turn exactly once more with
   the compacted transcript.
5. The retry's result (success or a second failure) is returned as-is — a
   second overflow is not compacted again. This bounds the loop to at most
   two model calls per turn even against a transcript that cannot be shrunk
   under the window.

## Tests

- `summarization::compaction::test` — cut points (never inside a tool pair,
  respects `keep_recent_tokens`), split-turn merge and `previous_summary`
  threading, `OverflowClassifier` per built-in pattern plus `with_pattern`.
- `middleware::test` — the retry path with a scripted model that overflows
  once then succeeds (asserts exactly two model calls and one `Compacted`
  event), the propagate-after-second-failure path, `Decline` leaving the
  transcript untouched (both `before_model` and `wrap_model`), persistence
  via a recording `CompactionSink`, and iterative-summary threading across
  two compactions on one middleware instance.
- `tinyagents_session::entry_tree::test` — `SessionCompactionSink` anchoring,
  tip advancement across repeated compactions, the empty-session and
  out-of-range-index no-op cases.
