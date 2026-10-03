//! Tests for how [`ContextCompressionMiddleware`] carries a compaction across
//! calls.
//!
//! The agent loop rebuilds every request from its own working transcript and
//! never sees the summary `before_model` splices in. These tests drive the
//! middleware the same way: each call gets the full, growing transcript.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::middleware::ContextCompressionMiddleware;
use crate::middleware::{Middleware, MiddlewareStack};
use crate::summarization::{
    CompactionRecord, CompactionSink, CompressionProvenance, SummarizationPolicy, Summarizer,
    SummaryRecord, SummaryRequest,
};
use tinyinference_llm::message::{ContentBlock, Message, UserMessage};
use tinyinference_llm::model::ModelRequest;

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

/// ~30 estimated tokens (chars / 4) tagged with `tag`.
fn chunk(tag: &str) -> Message {
    user(&format!("{tag}:{}", "x".repeat(116)))
}

/// The checkpoint the middleware writes for `summary` (default user placement).
fn cp(summary: &str) -> Message {
    crate::summarization::checkpoint_message(crate::summarization::SummaryPlacement::User, summary)
}

/// Answers every request with a short summary naming how many requests it has
/// seen, and records each request.
#[derive(Default)]
struct ShortSummarizer {
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
}

#[async_trait]
impl Summarizer for ShortSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        let mut seen = self.seen.lock().unwrap();
        seen.push(request.clone());
        Ok(SummaryRecord {
            summary: Message::system(format!("summary #{}", seen.len())),
            provenance: CompressionProvenance {
                source_ids: Vec::new(),
                original_token_estimate: 0,
                summary_token_estimate: 0,
                reason: "test".into(),
            },
            usage: None,
        })
    }
}

#[derive(Default)]
struct RecordingSink {
    records: Mutex<Vec<CompactionRecord>>,
}

impl CompactionSink for RecordingSink {
    fn persist(&self, record: &CompactionRecord) -> Result<()> {
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }
}

/// The middleware under test, what its summarizer and sink saw, and its context.
struct Fixture {
    stack: MiddlewareStack<()>,
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
    sink: Arc<RecordingSink>,
    c: RunContext,
}

/// A 100-token window at 0.5 → a 50-token trigger, keeping the newest message.
fn fixture() -> Fixture {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(100)
    .with_threshold_fraction(0.5);
    let summarizer = ShortSummarizer::default();
    let seen = summarizer.seen.clone();
    let mw: Arc<dyn Middleware<()>> = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(summarizer),
    ));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);
    let sink = Arc::new(RecordingSink::default());
    let c = ctx().with_compaction_sink(sink.clone());
    Fixture {
        stack,
        seen,
        sink,
        c,
    }
}

async fn send(
    stack: &MiddlewareStack<()>,
    c: &mut RunContext,
    transcript: &[Message],
) -> Vec<Message> {
    let mut request = ModelRequest {
        messages: transcript.to_vec(),
        ..Default::default()
    };
    stack.run_before_model(c, &(), &mut request).await.unwrap();
    request.messages
}

#[tokio::test]
async fn reapplies_the_fold_instead_of_recompacting_every_call() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture();
    let mut transcript = vec![chunk("m1"), chunk("m2"), chunk("m3")];

    // ~90 tokens: over the 50-token trigger, so the first call compacts m1, m2.
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(sent, vec![cp("summary #1"), chunk("m3")]);

    // The loop's transcript still holds m1 and m2 (it never saw the summary)
    // and grows by a small message. The fold is re-applied, so the request
    // stays small and nothing is summarized again.
    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1, "no second summarizer call");
    assert_eq!(
        sent,
        vec![cp("summary #1"), chunk("m3"), user("ok")]
    );
    assert_eq!(sink.records.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn compacts_only_history_newer_than_the_fold() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture();
    let mut transcript = vec![chunk("m1"), chunk("m2"), chunk("m3")];
    send(&stack, &mut c, &transcript).await;

    // Grow past the trigger again: the next compaction must summarize only
    // what came after m2, building on the first summary rather than re-reading
    // m1 and m2.
    transcript.extend([chunk("m4"), chunk("m5")]);
    let sent = send(&stack, &mut c, &transcript).await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages, vec![chunk("m3"), chunk("m4")]);
    assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    // The new summary replaces the one it was built on.
    assert_eq!(sent, vec![cp("summary #2"), chunk("m5")]);

    // Persisted boundaries are positions in the live transcript, which is
    // what a session-backed sink maps to entry ids: m3, then m5.
    let firsts: Vec<usize> = sink
        .records
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.first_kept_index)
        .collect();
    assert_eq!(firsts, vec![2, 4]);
}

#[tokio::test]
async fn drops_the_fold_when_the_transcript_no_longer_matches() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    send(&stack, &mut c, &[chunk("m1"), chunk("m2"), chunk("m3")]).await;

    // A different history (rewritten or replaced): splicing the old summary
    // over it would be wrong, so it is compacted from scratch.
    let other = vec![chunk("n1"), chunk("n2"), chunk("n3")];
    let sent = send(&stack, &mut c, &other).await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages, vec![chunk("n1"), chunk("n2")]);
    assert_eq!(sent, vec![cp("summary #2"), chunk("n3")]);
}

#[tokio::test]
async fn keeps_system_prompts_ahead_of_the_reapplied_summary() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    let system = Message::system("You are a coding agent.");
    let mut transcript = vec![system.clone(), chunk("m1"), chunk("m2"), chunk("m3")];
    send(&stack, &mut c, &transcript).await;

    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(
        sent,
        vec![
            system,
            cp("summary #1"),
            chunk("m3"),
            user("ok"),
        ]
    );
}

#[tokio::test]
async fn concat_summarizer_carries_the_previous_summary_forward() {
    let record = crate::summarization::ConcatSummarizer
        .summarize_request(&SummaryRequest {
            messages: vec![user("new")],
            previous_summary: Some("earlier".into()),
        })
        .await
        .unwrap();
    let text = record.summary.text();
    assert!(text.starts_with("earlier\n"), "{text}");
    assert!(text.contains("new"), "{text}");
}
