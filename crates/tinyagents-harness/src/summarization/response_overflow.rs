//! Context overflow detected from a *successful* model response.
//!
//! [`super::OverflowClassifier`] only sees errors. Some servers never raise
//! one: they accept an oversized prompt and either report usage above the
//! window (z.ai style "silent overflow") or truncate the input to fit and stop
//! with `length` and no output (Xiaomi MiMo style). Port of the response cases
//! of pi's `isContextOverflow` and `isRecoverableLength` (`overflow.ts`).
//!
//! Detection is pure; acting on it (compact and retry) belongs to
//! [`crate::middleware::ContextCompressionMiddleware`].

use tinyinference_llm::usage::Usage;

use super::compaction::OverflowInfo;

/// Fraction of the window a zero-output `length` stop must have consumed to
/// count as a truncated-input overflow (pi: 0.99).
const FULL_WINDOW_FRACTION: f64 = 0.99;

/// A `length` stop whose output stayed below `1 / SHORT_LENGTH_DIVISOR` of the
/// requested cap is "far below" it: the stop was imposed by something other
/// than the requested limit, typically the window leaving no room.
const SHORT_LENGTH_DIVISOR: u64 = 2;

/// Which successful-response signals count as a context overflow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResponseOverflowDetection {
    /// Only errors are classified. The default: discarding a successful
    /// response throws away billed work and, on a streamed call, output the
    /// consumer already saw, so a host opts in.
    #[default]
    Off,
    /// Reported prompt tokens above the window, and a zero-output `length`
    /// stop with the window full. Both need a known window and are
    /// unambiguous. The discarded response's usage is still accounted to the
    /// run; streamed calls are never discarded.
    Usage,
    /// [`Self::Usage`] plus a `length` stop whose output is far below the
    /// requested `max_tokens`. A model can also stop short for its own
    /// reasons, and each false positive costs a compaction, so this is opt-in.
    UsageAndShortLength,
}

/// Classifies a model response as a context overflow, or `None`.
///
/// `usage.input_tokens` is the whole prompt (cache reads are a subset of it).
/// `context_window` is the model's window when known; without one only the
/// short-`length` rule can fire. `requested_max_tokens` is the request's
/// output cap *before* any clamping.
pub fn detect_response_overflow(
    mode: ResponseOverflowDetection,
    usage: Option<&Usage>,
    finish_reason: Option<&str>,
    context_window: Option<u64>,
    requested_max_tokens: Option<u32>,
) -> Option<OverflowInfo> {
    if mode == ResponseOverflowDetection::Off {
        return None;
    }
    let usage = usage.filter(|usage| usage.input_tokens > 0)?;
    let info = OverflowInfo {
        requested: Some(usage.input_tokens),
        limit: context_window,
    };
    let length_stop = crate::finish_reason::is_length_stop(finish_reason);

    if let Some(window) = context_window {
        if !length_stop && usage.input_tokens > window {
            return Some(info);
        }
        if length_stop
            && usage.output_tokens == 0
            && usage.input_tokens as f64 >= window as f64 * FULL_WINDOW_FRACTION
        {
            return Some(info);
        }
    }

    if mode == ResponseOverflowDetection::UsageAndShortLength
        && length_stop
        && let Some(cap) = requested_max_tokens.filter(|cap| *cap > 0)
        && usage.output_tokens.saturating_mul(SHORT_LENGTH_DIVISOR) < u64::from(cap)
    {
        return Some(info);
    }
    None
}

#[cfg(test)]
#[path = "response_overflow_tests.rs"]
mod tests;
