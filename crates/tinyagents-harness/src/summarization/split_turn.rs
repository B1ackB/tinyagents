//! Split-turn summarization.
//!
//! A compaction cut can land inside a turn: the user's request and the first
//! steps of the answer are folded away while the rest of the answer stays
//! verbatim. Summarized as plain history, that prefix loses the one thing the
//! kept tail needs — what was asked and what has been done so far. Port of
//! pi's turn-prefix summary (`compaction.ts`, `generateTurnPrefixSummary`): the
//! history before the turn is summarized as usual, the turn's prefix gets its
//! own request ([`SummaryKind::TurnPrefix`], so an LLM summarizer can use a
//! prompt written for it), and the two are joined under a
//! `**Turn Context (split turn):**` heading.
//!
//! This replaces size-halving for the case a cut splits a turn. A single turn
//! that is merely too large for one summarization call is still handled by
//! [`super::summarize_with_split`].

use tinyinference_llm::message::Message;

use crate::error::Result;

use super::compaction::summarize_with_split;
use super::types::{CompressionProvenance, Summarizer, SummaryRecord, SummaryRequest};

/// Heading that introduces the turn-prefix summary inside a merged summary.
pub const SPLIT_TURN_HEADING: &str = "**Turn Context (split turn):**";

/// Where, in `to_summarize`, the turn the cut splits begins.
///
/// `Some(i)` when the kept tail's first non-system message is not a user
/// message (the cut is mid-turn) and an earlier user message at `i > 0`
/// opens that turn, leaving history before it. `None` when the cut is at a
/// turn boundary, when the turn has no earlier history, or when the turn's
/// user message is itself kept (pinned).
pub fn split_turn_start(to_summarize: &[Message], to_keep: &[Message]) -> Option<usize> {
    let first_kept = to_keep.iter().find(|m| !matches!(m, Message::System(_)))?;
    if matches!(first_kept, Message::User(_)) {
        return None;
    }
    let start = to_summarize
        .iter()
        .rposition(|m| matches!(m, Message::User(_)))?;
    (start > 0).then_some(start)
}

/// Summarizes `to_summarize`, giving the split turn's prefix (from
/// `turn_start`) its own [`SummaryKind::TurnPrefix`](super::types::SummaryKind)
/// request when `turn_start` is `Some`. With `None` it is
/// [`summarize_with_split`]. `previous_summary` goes to the history half only.
pub async fn summarize_split_turn(
    summarizer: &dyn Summarizer,
    to_summarize: &[Message],
    turn_start: Option<usize>,
    max_turn_tokens: u64,
    previous_summary: Option<String>,
    estimator: impl Fn(&Message) -> u64,
) -> Result<SummaryRecord> {
    let Some(start) = turn_start.filter(|s| *s > 0 && *s < to_summarize.len()) else {
        return summarize_with_split(
            summarizer,
            to_summarize,
            max_turn_tokens,
            previous_summary,
            estimator,
        )
        .await;
    };
    let (history, prefix) = to_summarize.split_at(start);
    tracing::debug!(
        history = history.len(),
        prefix = prefix.len(),
        "[summarize] cut splits a turn; summarizing the turn prefix separately"
    );
    let history = summarize_with_split(
        summarizer,
        history,
        max_turn_tokens,
        previous_summary,
        estimator,
    )
    .await?;
    let prefix = summarizer
        .summarize_request(&SummaryRequest::turn_prefix(prefix.to_vec()))
        .await?;

    let text = format!(
        "{}\n\n---\n\n{SPLIT_TURN_HEADING}\n\n{}",
        history.summary.text(),
        prefix.summary.text()
    );
    let mut source_ids = history.provenance.source_ids.clone();
    source_ids.extend(prefix.provenance.source_ids.iter().cloned());
    Ok(SummaryRecord {
        summary: Message::system(text),
        provenance: CompressionProvenance {
            source_ids,
            original_token_estimate: history.provenance.original_token_estimate
                + prefix.provenance.original_token_estimate,
            summary_token_estimate: history.provenance.summary_token_estimate
                + prefix.provenance.summary_token_estimate,
            reason: "split-turn summary: history plus turn prefix".to_string(),
        },
        usage: match (history.usage, prefix.usage) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        },
    })
}

#[cfg(test)]
#[path = "split_turn_tests.rs"]
mod tests;
