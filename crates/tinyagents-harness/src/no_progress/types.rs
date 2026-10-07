//! Type definitions for the no-progress escalation ladder
//! (`ToolAttempt`, `NoProgress`, `LadderState`, `NoProgressTracker`).
//!
//! Split out of `no_progress/mod.rs`; see that module's doc comment for
//! the full escalation-ladder design.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::OutcomeFingerprinter;

/// One recorded tool outcome, as the driver observed it.
///
/// Built with [`ToolAttempt::success`] / [`ToolAttempt::failure`] plus the
/// [`ToolAttempt::hard_reject`] / [`ToolAttempt::recoverable_miss`] modifiers,
/// so a driver never has to remember the field set. Borrows rather than owns so
/// an `after_tool` hook can record without allocating anything but the argument
/// fingerprint.
pub struct ToolAttempt<'a> {
    /// Tool name.
    pub tool: &'a str,
    /// Stable fingerprint of the call arguments (computed by the driver). Folded
    /// into the identical-repeat signature so the "identical arguments" ladder
    /// only trips when the args truly repeat.
    pub arg_fingerprint: &'a str,
    /// `None` on success; otherwise the tool's error text.
    pub error: Option<&'a str>,
    /// `true` when the result is a hard security/approval rejection that can
    /// never succeed re-issued unchanged.
    pub hard_reject: bool,
    /// `true` for the unknown-tool recovery sentinel — a correctable miss that
    /// must not feed the generic any-failure backstop (it still feeds the
    /// identical-repeat counter, so re-issuing the *same* unavailable tool
    /// halts).
    pub recoverable_miss: bool,
}

impl<'a> ToolAttempt<'a> {
    /// A tool call that succeeded. Clears every ladder counter when recorded.
    ///
    /// `arg_fingerprint` should come from
    /// [`fingerprint_arguments`][crate::no_progress::fingerprint_arguments]
    /// so every driver computes it the same way.
    pub fn success(tool: &'a str, arg_fingerprint: &'a str) -> Self {
        Self {
            tool,
            arg_fingerprint,
            error: None,
            hard_reject: false,
            recoverable_miss: false,
        }
    }

    /// A tool call that failed, with the error text the model saw.
    pub fn failure(tool: &'a str, arg_fingerprint: &'a str, error: &'a str) -> Self {
        Self {
            tool,
            arg_fingerprint,
            error: Some(error),
            hard_reject: false,
            recoverable_miss: false,
        }
    }

    /// Marks the failure as a hard security/approval rejection, which can never
    /// succeed re-issued unchanged and so trips the ladder fastest.
    pub fn hard_reject(mut self) -> Self {
        self.hard_reject = true;
        self
    }

    /// Marks the failure as the unknown-tool recovery sentinel: correctable
    /// feedback the model already received, which must not feed the generic
    /// any-failure backstop.
    pub fn recoverable_miss(mut self) -> Self {
        self.recoverable_miss = true;
        self
    }
}

/// The ladder's verdict for one recorded attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoProgress {
    /// Progress was made, or not enough repetition yet — carry on.
    Continue,
    /// Same-strategy repetition detected below the retry cap: feed this
    /// structured "no progress since step X" corrective back into the loop so
    /// the model picks a *different* next action.
    Nudge(String),
    /// Same-strategy retries exhausted (or the any-failure backstop tripped):
    /// halt with this root-cause summary.
    Halt(String),
}

impl NoProgress {
    /// The corrective/summary text carried by a [`NoProgress::Nudge`] or
    /// [`NoProgress::Halt`]; `None` for [`NoProgress::Continue`].
    ///
    /// Saves a driver from matching the enum just to reach the string it has to
    /// forward either way.
    pub fn message(&self) -> Option<&str> {
        match self {
            NoProgress::Continue => None,
            NoProgress::Nudge(message) | NoProgress::Halt(message) => Some(message),
        }
    }

    /// `true` when the loop should keep running but feed the corrective back to
    /// the model.
    pub fn is_nudge(&self) -> bool {
        matches!(self, NoProgress::Nudge(_))
    }

    /// `true` when the loop must stop.
    pub fn is_halt(&self) -> bool {
        matches!(self, NoProgress::Halt(_))
    }

    /// Stable, snake_case label for logs and telemetry dimensions.
    pub fn as_str(&self) -> &'static str {
        match self {
            NoProgress::Continue => "continue",
            NoProgress::Nudge(_) => "nudge",
            NoProgress::Halt(_) => "halt",
        }
    }
}

/// Mutable counters backing [`NoProgressTracker::record`][crate::no_progress::NoProgressTracker::record];
/// reset to default on any success or on a halt.
#[derive(Default)]
pub(super) struct LadderState {
    /// Signature of the previous failing call (tool + args + first error line).
    pub(super) last_sig: Option<String>,
    /// Consecutive repeats of `last_sig`.
    pub(super) same_count: usize,
    /// Consecutive failures of any kind (reset by any success).
    pub(super) consecutive: usize,
    /// Signature we have already nudged on, so a nudge fires at most once per
    /// distinct failing `(tool, args, error)` before escalating to a halt.
    pub(super) nudged_sig: Option<String>,
    /// `true` once the varied-failure nudge fired for the current streak.
    pub(super) nudged_streak: bool,
}

/// Tracks recent tool outcomes and drives the no-progress escalation ladder.
///
/// Cheap to construct and interior-mutable, so a middleware can hold one behind
/// a shared reference for the whole turn. `identical_halt_threshold` is the
/// same-strategy retry cap; it is clamped so a nudge always precedes a halt.
pub struct NoProgressTracker {
    pub(super) identical_halt_threshold: usize,
    /// Reduces a failure message to the identity the identical-repeat rung
    /// compares, so volatile spans (timestamps, attempt counters) do not make
    /// a repeated failure look novel.
    pub(super) fingerprinter: Arc<dyn OutcomeFingerprinter>,
    pub(super) state: Mutex<LadderState>,
}

/// Verdict returned after recording a successful-repeat signal.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SuccessfulRepeat {
    /// The signature changed, is exempt, failed, or remains below its threshold.
    Continue,
    /// Staged escalation only: the repeat just reached its first threshold.
    /// The message is a short note for the host to attach to the tool result
    /// the model is about to read, telling it to change approach. Reported once
    /// per signature.
    Warn(String),
    /// The same successful action has repeated enough times to be considered
    /// stuck. The message is suitable for steering or a halt summary.
    Halt(String),
}

/// Verdict from [`SuccessfulRepeatTracker::pre_call`], asked *before* a call
/// executes. Only staged escalation ever answers anything but `Allow`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CallGate {
    /// Run the call.
    Allow,
    /// Do not run the call: answer it with this error text asking the model to
    /// reassess.
    Block(String),
    /// The call's second block: do not run it and halt the run with this
    /// summary.
    Halt(String),
}

/// One run of consecutive identical signatures (assistant output or a
/// successful tool-call batch), keyed by a content hash rather than the
/// content itself to keep the tracker cheap to hold across a whole turn.
#[derive(Default)]
pub(super) struct Streak {
    pub(super) last_hash: Option<u64>,
    pub(super) consecutive: u32,
}

/// Tracks identical assistant-output and successful tool-call batches.
///
/// The two streaks are independent, but their verdict timing is coordinated:
/// output is staged before tools execute and can halt only after the matching
/// call batch is recorded as successful and non-exempt. Exempt polling batches
/// and failed batches reset both streaks so the failure ladder remains
/// authoritative.
///
/// A third, run-wide ledger counts how often each successful call returned the
/// same result, so repeats that are not back to back (a model cycling A, B, A,
/// B) are caught too. It is not cleared by failed or exempt batches; see
/// [`SuccessfulRepeatTracker::record_call_outcome`].
pub struct SuccessfulRepeatTracker {
    pub(super) output_threshold: u32,
    pub(super) call_threshold: u32,
    /// Reduces a tool result to the identity the recurrence ledger keys on, so
    /// volatile spans (timestamps, durations, request ids) do not make a
    /// repeated result look novel.
    pub(super) fingerprinter: Arc<dyn OutcomeFingerprinter>,
    pub(super) output: Mutex<Streak>,
    pub(super) calls: Mutex<Streak>,
    /// Hash of `(call signature, outcome signature)` → times recorded this run.
    pub(super) recurrences: Mutex<HashMap<u64, u32>>,
    /// Staged escalation settings; `None` halts at the first threshold.
    pub(super) escalation: Option<super::escalation::RepeatEscalation>,
    /// Call-signature hash → ledger key of the outcome that call returned
    /// last, so [`SuccessfulRepeatTracker::pre_call`] can tell how often the
    /// next attempt would repeat a result without executing it.
    pub(super) last_outcome: Mutex<HashMap<u64, u64>>,
    /// Calls whose `last_outcome` may still be predicted. Invalidation removes
    /// calls from here but keeps `last_outcome`, so a later new result is still
    /// recognised as progress.
    pub(super) predictable: Mutex<std::collections::HashSet<u64>>,
    /// Call-signature hash → times that call was blocked (survives context
    /// eviction; cleared when the call returns a new result).
    pub(super) blocks: Mutex<HashMap<u64, u32>>,
}

/// Staged escalation of a successful repeat: **warn**, then **block**, then
/// **halt**.
///
/// The first threshold a repeat reaches (the tracker's `call_threshold` /
/// `output_threshold`) only *warns*: the host appends a note to the tool result
/// telling the model it is repeating itself. If the model repeats anyway the
/// call is **blocked** (not executed, answered with an error asking it to
/// reassess); a second block in the same run **halts** the run.
///
/// Without escalation (`Option<RepeatEscalation>` = `None` on the tracker, the
/// default for [`SuccessfulRepeatTracker::new`](super::SuccessfulRepeatTracker::new))
/// the first threshold halts immediately, which is the historical behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeatEscalation {
    /// How many repeats past the warning the call is blocked at: with the
    /// default warning at 3 identical results, the 5th identical call is the
    /// first one blocked. Clamped to at least 1 so a warning always lands
    /// before a block.
    pub block_after_warn: u32,
    /// Blocks of one call tolerated before the run halts; the halting block is
    /// the `blocks_before_halt`-th. Clamped to at least 1 (halt on the first
    /// block).
    pub blocks_before_halt: u32,
}

/// Settings for [`RepeatMonitor`] (and the middleware built on it).
///
/// The default is the staged ladder: warn at the first threshold (3 identical
/// results, 3 identical batches, 4 identical outputs), block two repeats
/// later, halt on the second block, plus the warning-only pattern detectors and
/// the post-compaction guard. [`immediate_halt`](Self::immediate_halt) restores
/// the historical halt-at-the-first-threshold behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RepeatProgressConfig {
    /// Consecutive identical output batches that trigger the first stage.
    pub output_threshold: u32,
    /// Identical results of one call, or identical call batches in a row, that
    /// trigger the first stage.
    pub call_threshold: u32,
    /// Staged escalation; `None` halts at the first threshold.
    pub escalation: Option<RepeatEscalation>,
    /// Alternating calls before the ping-pong warning; `0` disables it.
    pub ping_pong_alternations: u32,
    /// Argument variants before the churn warning; `0` disables it.
    pub churn_variants: u32,
    /// Calls per variant, with one result, that make a variant count.
    pub churn_calls_per_variant: u32,
    /// Calls watched after a compaction (and remembered before it); `0`
    /// disables the guard (warning-only).
    pub post_compaction_window: u32,
}

/// What recording one successful call produced.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CallObservation {
    /// The exact-repeat ledger's verdict for the call.
    pub verdict: SuccessfulRepeat,
    /// Warning notes to attach to this call's result, from the pattern
    /// detectors and the post-compaction guard.
    pub notes: Vec<String>,
}

/// See the module docs.
pub struct RepeatMonitor {
    pub(super) tracker: SuccessfulRepeatTracker,
    pub(super) ping_pong: Option<PingPongDetector>,
    pub(super) churn: Option<ArgumentChurnDetector>,
    pub(super) guard: Option<PostCompactionGuard>,
}

/// One call and the result it produced, as hashes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Step {
    pub(super) call: u64,
    pub(super) outcome: u64,
}

#[derive(Default)]
pub(super) struct PingPongState {
    pub(super) prev: Option<Step>,
    pub(super) last: Option<Step>,
    /// Tool names of `prev` and `last`, for the warning text.
    pub(super) names: (String, String),
    /// Length of the current strictly alternating tail ending at `last`.
    pub(super) tail: u32,
    /// Pairs already warned about (order-independent hash of both calls).
    pub(super) warned: HashSet<u64>,
}

/// Detects two calls alternating with a stable result on each side.
pub struct PingPongDetector {
    pub(super) alternations: u32,
    pub(super) state: Mutex<PingPongState>,
}

#[derive(Default)]
pub(super) struct ChurnState {
    /// `(tool, outcome)` → argument variant → calls so far.
    pub(super) groups: HashMap<(String, u64), HashMap<u64, u32>>,
    /// Groups already warned about; their counts are dropped.
    pub(super) warned: HashSet<(String, u64)>,
}

/// Detects one tool called with many argument variants that all return the
/// same result. Tracking is bounded ([`MAX_CHURN_GROUPS`],
/// [`MAX_CHURN_VARIANTS_PER_GROUP`]) so a run supplying unique values cannot
/// grow it without limit; past the bound new groups are simply not tracked.
pub struct ArgumentChurnDetector {
    pub(super) variants: u32,
    pub(super) calls_per_variant: u32,
    /// One lock for counts and warnings so `record` and `reset` are atomic.
    pub(super) state: Mutex<ChurnState>,
}

#[derive(Default)]
pub(super) struct GuardState {
    /// Repeating `(call, result)` hashes with the call number they were seen
    /// at, oldest first. Only pairs from the last `window` calls are kept.
    pub(super) tail: VecDeque<(u64, u64)>,
    /// Successful calls recorded so far, repeating or not.
    pub(super) calls: u64,
    /// The tail captured at compaction and the calls still to watch.
    pub(super) armed: Option<(Vec<u64>, u32)>,
}

/// Flags calls that repeat the pre-compaction tail. See the module docs.
pub struct PostCompactionGuard {
    pub(super) window: u32,
    pub(super) state: Mutex<GuardState>,
}
