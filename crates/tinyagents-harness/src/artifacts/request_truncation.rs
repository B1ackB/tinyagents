//! Request-side truncation of oversized tool results.
//!
//! The cheap first route of context-overflow recovery: before paying for a
//! summary, cut the few huge tool outputs that usually cause the overflow.
//! Unlike [`super::tool_results::ToolResultArtifactStore`] this never touches
//! disk or the transcript — it rewrites the *outgoing request* only, with the
//! same head-plus-notice cut the per-result budget uses
//! (`truncated by tool_result_budget`), so the model sees one consistent
//! truncation notice whichever layer cut.
//!
//! The rewrite is a pure function of the messages and the byte cap, and it is
//! idempotent (a result that already carries the notice is left alone), so a
//! host that re-applies it to every request keeps a byte-stable prompt prefix.
//!
//! Never cut: results flagged `ToolMessage::trusted_verbatim`, non-text
//! blocks, and `[tool_result_preview]` envelopes (already bounded, and they
//! carry the artifact pointer the model needs).

use tinyinference_llm::message::{ContentBlock, Message};

use super::tool_results::{TRAILER_RESERVED, apply_tool_result_budget};

/// The grep handle every truncation layer shares; see `apply_tool_result_budget`.
const TRUNCATION_MARKER: &str = "truncated by tool_result_budget";
const PREVIEW_ENVELOPE_PREFIX: &str = "[tool_result_preview]\n";

/// What one [`truncate_tool_results`] pass changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequestTruncation {
    /// Text blocks that were cut.
    pub truncated: usize,
    /// Bytes removed from the request (cut bytes minus the added notices).
    pub saved_bytes: usize,
}

/// Whether `text` is a candidate for cutting at `max_bytes`.
fn is_reducible(text: &str, max_bytes: usize) -> bool {
    text.len() > max_bytes
        && !text.starts_with(PREVIEW_ENVELOPE_PREFIX)
        && !text.contains(TRUNCATION_MARKER)
}

/// Estimate, in bytes, of what [`truncate_tool_results`] could save at
/// `max_bytes`: every reducible text block's excess over the cap. The notice a
/// real cut appends is not modelled, so the true saving differs by up to
/// about one notice (a few hundred bytes) per block.
pub fn reducible_tool_result_bytes(messages: &[Message], max_bytes: usize) -> usize {
    if max_bytes == 0 {
        return 0;
    }
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool) if !tool.trusted_verbatim => Some(tool),
            _ => None,
        })
        .flat_map(|tool| tool.content.iter())
        .filter_map(|block| match block {
            ContentBlock::Text(text) if is_reducible(text, max_bytes) => {
                Some(text.len() - max_bytes)
            }
            _ => None,
        })
        .sum()
}

/// Cuts every reducible tool-result text block in `messages` to about
/// `max_bytes` (head kept, notice appended). `max_bytes == 0` disables it.
pub fn truncate_tool_results(messages: &mut [Message], max_bytes: usize) -> RequestTruncation {
    let mut outcome = RequestTruncation::default();
    if max_bytes == 0 {
        return outcome;
    }
    for message in messages {
        let Message::Tool(tool) = message else {
            continue;
        };
        if tool.trusted_verbatim {
            continue;
        }
        for block in &mut tool.content {
            let ContentBlock::Text(text) = block else {
                continue;
            };
            if !is_reducible(text, max_bytes) {
                continue;
            }
            // Leave room for the notice so the result lands near the cap.
            let budget = max_bytes.max(TRAILER_RESERVED + 1);
            let original = std::mem::take(text);
            let before = original.len();
            let (cut, _) = apply_tool_result_budget(original.clone(), budget);
            if cut.len() >= before {
                *text = original;
                continue;
            }
            outcome.truncated += 1;
            outcome.saved_bytes += before - cut.len();
            tracing::debug!(
                target: "tinyagents::artifacts",
                tool_call_id = %tool.tool_call_id,
                before,
                after = cut.len(),
                "[request_truncation] cut an oversized tool result in the outgoing request"
            );
            *text = cut;
        }
    }
    outcome
}

#[cfg(test)]
#[path = "request_truncation_tests.rs"]
mod request_truncation_test;
