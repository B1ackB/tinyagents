//! Best-effort repair of a model's structured-output text into strict JSON.
//!
//! # Why this exists
//!
//! Structured extraction used to be a bare [`serde_json::from_str`] on the
//! assistant text, and a single malformed brace on the final turn discarded the
//! whole run — every tool call and token already spent. Meanwhile the crate
//! already carried a repair ladder for the *other* place a model emits JSON:
//! tool-call arguments, repaired by
//! [`recover_tool_arguments`][rta] over
//! [`relaxed_json`][rj]. Structured output got none of it.
//!
//! [rta]: tinyinference_llm::providers::openai
//! [rj]: tinyinference_llm::providers::openai::relaxed_json
//!
//! # The ladder
//!
//! Each rung is tried in order and the first strict parse wins. Every rung is
//! *conservative*: it only ever runs after strict parsing has already failed,
//! and a rung that does not yield strictly-parseable JSON is discarded rather
//! than half-applied.
//!
//! | Rung | Repairs | Modelled on |
//! |------|---------|-------------|
//! | `Strict` | nothing — the input was already valid | — |
//! | `CodeFence` | ```` ```json … ``` ```` wrappers | ubiquitous |
//! | `Slice` | prose around the value (`Here is the JSON: {…}`) | — |
//! | `Relaxed` | unquoted keys, doubled braces, leaked chat-template quote tokens | [`relaxed_json`][rj] |
//! | `Closed` | truncated output: unterminated strings and unclosed brackets | LangChain `parse_partial_json` |
//!
//! # What it deliberately does not do
//!
//! It never *invents* structure. A rung is accepted only when the repaired text
//! parses strictly, so noise can never be laundered into a plausible-looking
//! value. Whether the parsed value is the *right shape* is a separate question,
//! answered by [`super::validate`] against the declared schema.

use serde_json::Value;

/// Which rung of the ladder produced a value.
///
/// Carried out of `parse_lenient` so the caller can log — and a
/// [`super::StructuredOutcome`] can record — that the model's text needed
/// repairing, instead of a repair silently masking a degrading model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonRepair {
    /// The text was already strict JSON.
    Strict,
    /// A markdown code fence was removed.
    CodeFence,
    /// The value was sliced out of surrounding prose.
    Slice,
    /// Relaxed-JSON repairs were applied (unquoted keys, doubled braces, leaked
    /// chat-template quote tokens).
    Relaxed,
    /// Truncated output was closed (unterminated string and/or open brackets).
    Closed,
}

impl JsonRepair {
    /// A stable, log- and event-friendly label.
    pub fn as_str(self) -> &'static str {
        match self {
            JsonRepair::Strict => "strict",
            JsonRepair::CodeFence => "code_fence",
            JsonRepair::Slice => "slice",
            JsonRepair::Relaxed => "relaxed",
            JsonRepair::Closed => "closed",
        }
    }

    /// Whether any repair was actually needed.
    pub fn is_repaired(self) -> bool {
        self != JsonRepair::Strict
    }
}

/// Maximum trailing characters trimmed while closing a truncated value.
///
/// A truncated completion usually stops mid-token, so the tail that has to go
/// is short. Bounding the search keeps the cost linear-ish on adversarial input
/// instead of quadratic over a multi-megabyte blob.
const MAX_TRAILING_TRIM: usize = 64;

/// Parses `raw` as JSON, climbing the repair ladder until something parses.
///
/// Returns the parsed value and the rung that produced it, or `None` when no
/// conservative repair yields strict JSON.
pub fn parse_lenient(raw: &str) -> Option<(Value, JsonRepair)> {
    let trimmed = raw.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some((value, JsonRepair::Strict));
    }

    let unfenced = strip_code_fence(trimmed);
    if unfenced != trimmed
        && let Ok(value) = serde_json::from_str::<Value>(unfenced)
    {
        tracing::debug!("[structured::repair] recovered JSON by removing a markdown code fence");
        return Some((value, JsonRepair::CodeFence));
    }

    if let Some(sliced) = slice_json_span(unfenced)
        && let Ok(value) = serde_json::from_str::<Value>(sliced)
    {
        tracing::debug!(
            "[structured::repair] recovered JSON by slicing it out of surrounding text"
        );
        return Some((value, JsonRepair::Slice));
    }

    // Reuses the crate's existing relaxed-JSON repairs rather than a second,
    // divergent implementation. It only yields objects, which is the shape a
    // JSON-Schema structured output almost always declares.
    if let Some(value) = tinytools_agent::repair::json::recover_object(unfenced) {
        tracing::debug!("[structured::repair] recovered JSON through the relaxed-JSON repairs");
        return Some((value, JsonRepair::Relaxed));
    }

    if let Some(value) = close_truncated(unfenced) {
        tracing::debug!("[structured::repair] recovered JSON by closing a truncated value");
        return Some((value, JsonRepair::Closed));
    }

    None
}

/// Removes a ```` ``` ```` fence, with or without a language tag.
fn strip_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let body = match after_open.find('\n') {
        Some(newline)
            if after_open[..newline]
                .chars()
                .all(|character| character.is_ascii_alphanumeric()) =>
        {
            &after_open[newline + 1..]
        }
        _ => after_open,
    };
    body.trim().strip_suffix("```").unwrap_or(body).trim()
}

/// Returns the span from the first `{`/`[` to the matching last `}`/`]`.
///
/// Handles the extremely common "chatty" completion — `Sure! Here is the
/// result: {"score": 4}` — without any structural rewriting: the span is
/// returned verbatim and still has to parse strictly to be accepted.
fn slice_json_span(raw: &str) -> Option<&str> {
    let open = raw.find(['{', '['])?;
    let close = raw.rfind(['}', ']'])?;
    if close <= open {
        return None;
    }
    let span = &raw[open..=close];
    (span != raw).then_some(span)
}

/// Closes a value truncated mid-flight: an unterminated string, then any
/// brackets still open.
///
/// Mirrors LangChain's `parse_partial_json`: walk the text tracking string and
/// escape state, remember the closing delimiters owed, then append them. When
/// the result still does not parse — the truncation landed mid-key, mid-number,
/// or on a dangling comma — trailing characters are dropped one at a time and
/// the close is retried, bounded by [`MAX_TRAILING_TRIM`].
fn close_truncated(raw: &str) -> Option<Value> {
    let chars: Vec<char> = raw.chars().collect();
    let floor = chars.len().saturating_sub(MAX_TRAILING_TRIM);
    let mut end = chars.len();
    while end > floor && end > 0 {
        let candidate: String = chars[..end].iter().collect();
        if let Some(closed) = close_once(&candidate)
            && let Ok(value) = serde_json::from_str::<Value>(&closed)
        {
            return Some(value);
        }
        end -= 1;
    }
    None
}

/// Appends the delimiters `raw` still owes: a closing quote when it ends inside
/// a string, then every unclosed `}`/`]` in reverse order.
fn close_once(raw: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for character in raw.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => stack.push('}'),
            '[' if !in_string => stack.push(']'),
            '}' | ']' if !in_string => {
                // A closer with no matching opener means this is not a
                // truncated value at all; refuse rather than guess.
                stack.pop()?;
            }
            _ => {}
        }
    }

    if stack.is_empty() && !in_string {
        // Nothing was owed, so closing cannot help — the caller already tried a
        // strict parse of this exact text.
        return None;
    }

    let mut closed = String::with_capacity(raw.len() + stack.len() + 1);
    closed.push_str(raw);
    if in_string {
        closed.push('"');
    }
    while let Some(closer) = stack.pop() {
        closed.push(closer);
    }
    Some(closed)
}

#[cfg(test)]
#[path = "repair_test_tests.rs"]
mod test;
