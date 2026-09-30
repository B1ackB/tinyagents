//! Transcript and tool-timeline bookkeeping on the in-memory snapshot: prose
//! coalescing, ordering keys, and sub-agent entry lookup. Hosts translate their
//! own progress events into calls on these helpers.

use super::TurnStateMirror;
use super::caps::append_capped_transcript_text;
use crate::turn_state::types::{SubagentTranscriptItem, ToolTimelineEntry, TranscriptItem};

impl TurnStateMirror {
    /// Append a visible-narration delta to the transcript, coalescing into
    /// the trailing [`TranscriptItem::Narration`] when it's the most recent
    /// item and from the same round — so a streamed paragraph stays one item
    /// instead of one-per-token. A new round (or any intervening thinking /
    /// tool item) starts a fresh narration block.
    pub fn push_transcript_narration(&mut self, round: u32, delta: &str) {
        if let Some(TranscriptItem::Narration { round: r, text, .. }) =
            self.state.transcript.last_mut()
        {
            if *r == round {
                append_capped_transcript_text(text, delta);
                return;
            }
        }
        let seq = self.next_seq();
        let mut text = String::new();
        append_capped_transcript_text(&mut text, delta);
        self.state
            .transcript
            .push(TranscriptItem::Narration { round, seq, text });
    }

    /// Append a hidden-reasoning delta to the transcript, with the same
    /// coalescing rule as [`Self::push_transcript_narration`].
    ///
    /// Stamps the block's first delta as `started_at` and every appended delta
    /// as `ended_at` (epoch ms), which is what the "Thought for Ns" label reads.
    pub fn push_transcript_thinking(&mut self, round: u32, delta: &str) {
        let now = now_epoch_ms();
        if let Some(TranscriptItem::Thinking {
            round: r,
            text,
            ended_at,
            ..
        }) = self.state.transcript.last_mut()
        {
            if *r == round {
                append_capped_transcript_text(text, delta);
                *ended_at = Some(now);
                return;
            }
        }
        let seq = self.next_seq();
        let mut text = String::new();
        append_capped_transcript_text(&mut text, delta);
        tracing::trace!("[turn_state] thinking block opened round={round} seq={seq}");
        self.state.transcript.push(TranscriptItem::Thinking {
            round,
            seq,
            text,
            started_at: Some(now),
            ended_at: Some(now),
        });
    }

    /// Record a tool call in the transcript at the point it occurred, as a
    /// pointer into [`TurnState::tool_timeline`] (the row's status/label live
    /// there). Skips a duplicate if the same `call_id` was already recorded
    /// (e.g. a start event after an args-delta placeholder).
    pub fn push_transcript_tool(&mut self, round: u32, call_id: &str) {
        let already = self.state.transcript.iter().any(
            |item| matches!(item, TranscriptItem::ToolCall { call_id: c, .. } if c == call_id),
        );
        if already {
            return;
        }
        let seq = self.next_seq();
        self.state.transcript.push(TranscriptItem::ToolCall {
            round,
            seq,
            call_id: call_id.to_string(),
        });
    }

    /// Return the next monotonic transcript ordering key and advance it.
    pub fn next_seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        seq
    }

    /// Return the next monotonic tool-timeline ordering key and advance it.
    pub fn next_tool_seq(&mut self) -> u64 {
        let seq = self.next_tool_seq;
        self.next_tool_seq = self.next_tool_seq.saturating_add(1);
        seq
    }

    pub fn find_subagent_entry_mut(&mut self, task_id: &str) -> Option<&mut ToolTimelineEntry> {
        let needle = format!("subagent:{task_id}");
        self.state
            .tool_timeline
            .iter_mut()
            .rev()
            .find(|entry| entry.id == needle)
    }

    /// Append a sub-agent prose delta (narration when `is_thinking == false`,
    /// reasoning otherwise) to that sub-agent's transcript, coalescing into the
    /// trailing same-kind, same-iteration item so a streamed paragraph stays
    /// one entry (mirrors the frontend `appendSubagentStreamDelta`). Mutate-
    /// only (no flush) — high-frequency like the parent's `TextDelta`; the
    /// accumulated prose is persisted at the next sub-agent tool boundary.
    pub fn push_subagent_prose(
        &mut self,
        task_id: &str,
        iteration: u32,
        delta: &str,
        is_thinking: bool,
    ) {
        let Some(entry) = self.find_subagent_entry_mut(task_id) else {
            return;
        };
        let Some(activity) = entry.subagent.as_mut() else {
            return;
        };
        match activity.transcript.last_mut() {
            Some(SubagentTranscriptItem::Thinking {
                iteration: it,
                text,
            }) if is_thinking && *it == Some(iteration) => {
                append_capped_transcript_text(text, delta);
                return;
            }
            Some(SubagentTranscriptItem::Text {
                iteration: it,
                text,
            }) if !is_thinking && *it == Some(iteration) => {
                append_capped_transcript_text(text, delta);
                return;
            }
            _ => {}
        }
        let mut text = String::new();
        append_capped_transcript_text(&mut text, delta);
        activity.transcript.push(if is_thinking {
            SubagentTranscriptItem::Thinking {
                iteration: Some(iteration),
                text,
            }
        } else {
            SubagentTranscriptItem::Text {
                iteration: Some(iteration),
                text,
            }
        });
    }
}

/// Wall-clock epoch milliseconds for transcript timing (0 if the clock is
/// before the epoch, which only a broken system clock produces).
fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
