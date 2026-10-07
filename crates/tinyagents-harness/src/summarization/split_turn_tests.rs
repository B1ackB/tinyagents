use super::*;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyinference_llm::usage::Usage;

use super::super::types::{CompressionProvenance, SummaryKind, SummaryRequest};

fn user(t: &str) -> Message {
    Message::user(t)
}

fn assistant(t: &str) -> Message {
    Message::assistant(t)
}

/// Answers `kind:first-message` and records every request.
#[derive(Default)]
struct Recording {
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
}

#[async_trait]
impl Summarizer for Recording {
    async fn summarize(&self, messages: &[Message]) -> crate::error::Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(
        &self,
        request: &SummaryRequest,
    ) -> crate::error::Result<SummaryRecord> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(SummaryRecord {
            summary: Message::system(format!("{:?}:{}", request.kind, request.messages[0].text())),
            provenance: CompressionProvenance {
                source_ids: vec![format!("{}", request.messages.len())],
                original_token_estimate: 10,
                summary_token_estimate: 2,
                reason: "test".into(),
            },
            usage: Some(Usage {
                input_tokens: 5,
                ..Usage::default()
            }),
        })
    }
}

#[test]
fn a_cut_inside_a_turn_starts_at_that_turns_user_message() {
    let summarized = vec![user("u1"), assistant("a1"), user("u2"), assistant("a2")];
    let kept = vec![Message::system("sys"), assistant("a3")];
    assert_eq!(split_turn_start(&summarized, &kept), Some(2));
}

#[test]
fn a_cut_at_a_user_message_is_not_a_split_turn() {
    let summarized = vec![user("u1"), assistant("a1")];
    let kept = vec![Message::system("sys"), user("u2")];
    assert_eq!(split_turn_start(&summarized, &kept), None);
}

#[test]
fn a_turn_with_no_earlier_history_is_not_split() {
    let summarized = vec![user("u1"), assistant("a1")];
    assert_eq!(split_turn_start(&summarized, &[assistant("a2")]), None);
}

#[tokio::test]
async fn the_prefix_gets_its_own_summary_request_and_section() {
    let summarizer = Recording::default();
    let summarized = vec![user("u1"), assistant("a1"), user("u2"), assistant("a2")];
    let record = summarize_split_turn(
        &summarizer,
        &summarized,
        Some(2),
        u64::MAX,
        Some("earlier".into()),
        |_| 1,
    )
    .await
    .unwrap();

    let seen = summarizer.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].kind, SummaryKind::Full);
    assert_eq!(seen[0].messages.len(), 2);
    assert_eq!(seen[0].previous_summary.as_deref(), Some("earlier"));
    assert_eq!(seen[1].kind, SummaryKind::TurnPrefix);
    assert_eq!(seen[1].messages.len(), 2);
    assert_eq!(seen[1].previous_summary, None);

    let text = record.summary.text();
    assert_eq!(
        text,
        "Full:u1\n\n---\n\n**Turn Context (split turn):**\n\nTurnPrefix:u2"
    );
    assert_eq!(record.usage.unwrap().input_tokens, 10);
    assert_eq!(record.provenance.original_token_estimate, 20);
}

#[tokio::test]
async fn without_a_split_the_whole_slice_is_one_full_summary() {
    let summarizer = Recording::default();
    let summarized = vec![user("u1"), assistant("a1")];
    let record = summarize_split_turn(&summarizer, &summarized, None, u64::MAX, None, |_| 1)
        .await
        .unwrap();
    assert_eq!(summarizer.seen.lock().unwrap().len(), 1);
    assert_eq!(record.summary.text(), "Full:u1");
}

#[tokio::test]
async fn an_oversized_turn_prefix_is_halved_into_turn_prefix_requests() {
    let summarizer = Recording::default();
    let summarized = vec![
        user("u1"),
        assistant("a1"),
        user("u2"),
        assistant("a2"),
        user("u3"),
        assistant("a3"),
    ];
    // Each message is 10 tokens; the 4-message prefix (40) is over the 25 cap.
    summarize_split_turn(&summarizer, &summarized, Some(2), 25, None, |_| 10)
        .await
        .unwrap();
    let seen = summarizer.seen.lock().unwrap();
    let prefix_requests = seen
        .iter()
        .filter(|r| r.kind == SummaryKind::TurnPrefix)
        .count();
    assert_eq!(prefix_requests, 2, "{seen:?}");
}
