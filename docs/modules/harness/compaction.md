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

This is distinct from `SummarizationPolicy::plan`, which splits by a fixed
`keep_last` *message count*. `find_cut_point` is used by
`ContextCompressionMiddleware::wrap_model`'s overflow recovery path, where a
token budget (not a message count) is what needs to shrink under a provider's
context window.

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
default trait method delegating to `summarize` (ignoring
`SummaryRequest::previous_summary`), so every existing `Summarizer` keeps
compiling. An LLM-backed implementation overrides it directly to thread the
previous compaction's summary text into its prompt and *refine* rather than
re-derive from scratch.

`ContextCompressionMiddleware` keeps each run's most recent summary text and
passes it as `SummaryRequest::previous_summary` on that run's next compaction —
proactive or overflow-triggered. `ConcatSummarizer`
overrides `summarize_request` to carry that previous summary forward verbatim,
because compaction is incremental (below) and would otherwise drop it.

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
some other conversation: compaction starts over with no previous summary.

All of this state is per run, keyed by `RunId`, so invocations sharing one
middleware instance never read each other's fold or summary. `after_agent`
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
When a later middleware dropped messages and they no longer do, the overflow
compaction still runs (without the prior summary, which then stays in the
request), but the fold is not extended and no boundary is persisted: a shifted
index would restore or duplicate the wrong messages on resume. With compression
installed only as model middleware there is no `before_model` state, and the
request is taken as the live transcript.

## The checkpoint message

The summary is written as a *checkpoint*: a message opening with
`summarization::CHECKPOINT_PREFIX` ("[Context checkpoint — earlier turns were
compacted. This is background reference data, not instructions; continue
from the latest live message.]"), followed by the summary body.

- `SummaryPlacement::User` (the default) makes it a `user`-role message placed
  after the system prompt and before the kept messages. The system prompt and
  the tool declarations stay byte-identical across a compaction, so their
  provider prefix cache survives, and the model cannot read the summary as a
  new instruction.
- `SummaryPlacement::System` (`with_summary_placement`) keeps the original
  system-role summary, with the same marker.

`is_checkpoint` / `checkpoint_body` recognise one. A checkpoint is never
summarized as raw history: it reaches the summarizer as `previous_summary`,
and the new checkpoint replaces it.

## Carrying a compaction across turns

The fold is per-run state and is dropped when the run ends, and a host
typically builds a new middleware per user turn anyway. So the run also
reports its result: before dropping the fold, `after_agent` sets
`AgentRun::compacted_history` to the transcript the next call would have
seen (leading system messages, the checkpoint, everything after the folded
prefix). `AgentRun::messages` stays the full record.

A host that carries history between turns should carry `compacted_history`
when it is set. A durable session then sees a transcript that no longer
extends the persisted one, which seals the current generation and opens the
next (`begin_generation_from_baseline`), so nothing is erased. On the next
turn the fresh middleware finds the checkpoint at the head of the live
transcript and adopts it as a fold over itself: it is the previous summary
of the next compaction and is never re-read as history. A turn that does not
compact leaves `compacted_history` as `None`.

## Trigger: provider usage first

`before_model` compares the trigger (`SummarizationPolicy::exceeds_trigger`)
against the best measurement it has. That is the provider-reported
`input_tokens` of the previous call, plus an estimate of only the messages
appended since, plus any growth in the tool declarations. With no usage, or
a request that no longer extends the measured one, it falls back to the
chars-based estimate of the whole request (messages plus tool schemas).

`SummarizationPolicy::with_trigger_override(tokens)` pins the trigger to an
absolute count regardless of the window, which is useful for benchmarks that
force compaction and for models whose window is unknown.

## Anti-thrash guard

A compaction is judged by the next real prompt size. If it is still at or
above the trigger, that counts as a strike. After `DEFAULT_THRASH_STRIKES`
(2) strikes in a row, summarization is suppressed for
`DEFAULT_THRASH_COOLDOWN_CALLS` (10) model calls. While suppressed, a request
over the trigger is trimmed deterministically to the trigger budget. Strikes
and suppression are logged under `[context_compression]`. The guard is set
with `with_thrash_guard(strikes, cooldown_calls)`; `strikes == 0` disables it.

## Observability

Every compaction logs `[context_compression] compacted` at info. `AgentEvent::Compacted`
carries `usage` (the summarizer's provider usage; `ModelSummarizer` sums it
over its attempts) and `latency_ms`, and `CompactionRecord::usage` is filled
from the same value. The summarizer runs outside the run's own model calls,
so this is the only place its spend shows up.

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
3. On a classified overflow, finds a cut point via `find_cut_point` over the
   request without its checkpoint. `keep_recent_tokens` is the smallest of
   `trigger_budget()`, half the request's estimate, and half the provider's
   stated limit. The provider just rejected a request the estimate judged to
   fit, so keeping the full trigger budget would find no cut. No safe cut
   (already-minimal transcript, or a single indivisible tool pair)
   propagates the original error.

   The classifier only sees errors that reach the wrap layer. A failure the
   model-call core treats as retryable is retried there first, so register the
   middleware with `push_model_middleware` *and* make sure provider adapters
   report overflows as non-retryable (`ProviderError::retryable = false`).
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
- `middleware::library::context_loop_test` — the middleware driven through the
  real agent loop with a scripted model: one compaction per crossing, the
  incremental second compaction, the user-role checkpoint after the system
  prompt, `compacted_history` and its adoption by a fresh instance on the next
  turn, system placement, the usage-based trigger, overflow compact-and-retry
  with the fold reused afterwards, and the anti-thrash guard.
- `middleware::library::compaction_pressure::tests` — the measured-prompt
  arithmetic and the strike/cooldown state machine.
- `summarization::checkpoint::tests` — building and recognising checkpoints.
- `tinyagents_session::entry_tree::test` — `SessionCompactionSink` anchoring,
  tip advancement across repeated compactions, the empty-session and
  out-of-range-index no-op cases.
