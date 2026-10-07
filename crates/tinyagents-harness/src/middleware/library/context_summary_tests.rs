//! Tests for what [`ContextCompressionMiddleware`] puts in a compaction
//! summary: file-operation lists and the split-turn prefix.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::middleware::{ContextCompressionMiddleware, Middleware};
use crate::summarization::{
    CompressionProvenance, FileOpExtractor, FileOperations, SummarizationPolicy, Summarizer,
    SummaryKind, SummaryRecord, SummaryRequest,
};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, UserMessage};
use tinyinference_llm::model::ModelRequest;
use tinyinference_llm::tool::ToolCall;

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

/// ~60 estimated tokens tagged with `tag`.
fn chunk(tag: &str) -> Message {
    user(&format!("{tag}:{}", "x".repeat(236)))
}

fn call(id: &str, tool: &str, path: &str) -> Vec<Message> {
    vec![
        Message::Assistant(AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("y".repeat(236))],
            tool_calls: vec![ToolCall::new(id, tool, json!({ "path": path }))],
            usage: None,
            origin: None,
        }),
        Message::tool(id, "z".repeat(236)),
    ]
}

#[derive(Clone, Default)]
struct Recording {
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
}

#[async_trait]
impl Summarizer for Recording {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        self.seen.lock().unwrap().push(request.clone());
        let text = match request.kind {
            SummaryKind::Full => "HISTORY",
            SummaryKind::TurnPrefix => "PREFIX",
        };
        Ok(SummaryRecord {
            summary: Message::system(text),
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

/// 300-token window at 0.5: a 150-token trigger, keeping only the newest message.
fn policy() -> SummarizationPolicy {
    SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(300)
    .with_threshold_fraction(0.5)
}

fn middleware(summarizer: &Recording) -> ContextCompressionMiddleware {
    ContextCompressionMiddleware::with_summarizer(policy(), Box::new(summarizer.clone()))
}

/// Runs `before_model` over `messages` and returns the checkpoint text spliced in.
async fn compact(mw: &ContextCompressionMiddleware, messages: Vec<Message>) -> String {
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    let mut c = RunContext::new(RunConfig::new("r"), ());
    Middleware::<(), ()>::before_model(mw, &mut c, &(), &mut request)
        .await
        .unwrap();
    request
        .messages
        .iter()
        .find(|m| crate::summarization::is_checkpoint(m))
        .map(Message::text)
        .expect("a checkpoint was written")
}

fn transcript_with_file_calls() -> Vec<Message> {
    let mut messages = vec![chunk("m1")];
    messages.extend(call("c1", "read_file", "src/a.rs"));
    messages.extend(call("c2", "edit_file", "src/b.rs"));
    messages.push(chunk("m2"));
    messages.push(user("newest"));
    messages
}

#[tokio::test]
async fn a_summary_lists_the_files_its_tool_calls_touched() {
    let mw = middleware(&Recording::default());
    let text = compact(&mw, transcript_with_file_calls()).await;
    assert!(
        text.contains("<read-files>\nsrc/a.rs\n</read-files>"),
        "{text}"
    );
    assert!(
        text.contains("<modified-files>\nsrc/b.rs\n</modified-files>"),
        "{text}"
    );
}

#[tokio::test]
async fn file_lists_can_be_turned_off() {
    let mw = middleware(&Recording::default()).without_file_operations();
    let text = compact(&mw, transcript_with_file_calls()).await;
    assert!(!text.contains("<read-files>") && !text.contains("<modified-files>"));
}

#[tokio::test]
async fn the_extractor_is_pluggable() {
    struct Everything;
    impl FileOpExtractor for Everything {
        fn extract(&self, call: &ToolCall, ops: &mut FileOperations) {
            ops.add_modified(&call.name);
        }
    }
    let mw = middleware(&Recording::default()).with_file_op_extractor(Everything);
    let text = compact(&mw, transcript_with_file_calls()).await;
    assert!(
        text.contains("<modified-files>\nread_file\nedit_file\n</modified-files>"),
        "{text}"
    );
}

#[tokio::test]
async fn file_lists_accumulate_across_compactions_and_stay_out_of_the_summarizer_input() {
    let summarizer = Recording::default();
    let mw = middleware(&summarizer);
    let mut messages = transcript_with_file_calls();
    let first = compact(&mw, messages.clone()).await;
    assert!(first.contains("src/a.rs"));

    // The loop carries the checkpoint forward; more tool calls follow it.
    messages.pop();
    messages.extend(call("c3", "read_file", "src/c.rs"));
    messages.push(chunk("m3"));
    messages.push(user("newest again"));
    let mut next = vec![Message::user(first)];
    next.extend(messages.split_off(5));
    let second = compact(&mw, next).await;

    assert!(
        second.contains("src/a.rs") && second.contains("src/b.rs"),
        "{second}"
    );
    assert!(second.contains("src/c.rs"), "{second}");
    assert_eq!(second.matches("<read-files>").count(), 1, "{second}");
    let seen = summarizer.seen.lock().unwrap();
    let previous = seen
        .last()
        .unwrap()
        .previous_summary
        .as_deref()
        .unwrap_or("");
    assert!(!previous.contains("<read-files>"), "{previous}");
}

#[tokio::test]
async fn a_cut_inside_a_turn_summarizes_the_turn_prefix_separately() {
    let summarizer = Recording::default();
    let mw = middleware(&summarizer);
    // u1 a1 | u2 a2 | a3 (kept): the cut splits u2's turn.
    let messages = vec![
        chunk("u1"),
        Message::assistant(format!("a1 {}", "x".repeat(236))),
        chunk("u2"),
        Message::assistant(format!("a2 {}", "x".repeat(236))),
        Message::assistant(format!("a3 {}", "x".repeat(236))),
    ];
    let text = compact(&mw, messages).await;

    let seen = summarizer.seen.lock().unwrap();
    let kinds: Vec<SummaryKind> = seen.iter().map(|r| r.kind).collect();
    assert_eq!(kinds, vec![SummaryKind::Full, SummaryKind::TurnPrefix]);
    assert_eq!(seen[0].messages.len(), 2);
    assert_eq!(seen[1].messages.len(), 2);
    assert!(text.contains("HISTORY"), "{text}");
    assert!(
        text.contains("**Turn Context (split turn):**\n\nPREFIX"),
        "{text}"
    );
}

#[tokio::test]
async fn a_cut_at_a_turn_boundary_is_one_ordinary_summary() {
    let summarizer = Recording::default();
    let mw = middleware(&summarizer);
    let messages = vec![chunk("u1"), chunk("u2"), chunk("u3"), chunk("u4")];
    let text = compact(&mw, messages).await;
    assert_eq!(summarizer.seen.lock().unwrap().len(), 1);
    assert!(!text.contains("split turn"));
}

#[tokio::test]
async fn the_turn_prefix_request_can_be_turned_off() {
    let summarizer = Recording::default();
    let mw = middleware(&summarizer).with_split_turn_prefix(false);
    let messages = vec![
        chunk("u1"),
        Message::assistant(format!("a1 {}", "x".repeat(236))),
        chunk("u2"),
        Message::assistant(format!("a2 {}", "x".repeat(236))),
        Message::assistant(format!("a3 {}", "x".repeat(236))),
    ];
    compact(&mw, messages).await;
    let kinds: Vec<SummaryKind> = summarizer
        .seen
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.kind)
        .collect();
    assert_eq!(kinds, vec![SummaryKind::Full]);
}

#[tokio::test]
async fn a_hook_supplied_summary_still_gets_the_file_lists() {
    let summarizer = Recording::default();
    let mw = middleware(&summarizer).with_before_compaction(|_| {
        crate::summarization::CompactionDecision::UseSummary("HOOK SUMMARY".into())
    });
    let text = compact(&mw, transcript_with_file_calls()).await;
    assert!(text.contains("HOOK SUMMARY"), "{text}");
    assert!(
        text.contains("<read-files>\nsrc/a.rs\n</read-files>"),
        "{text}"
    );
    assert!(
        text.contains("<modified-files>\nsrc/b.rs\n</modified-files>"),
        "{text}"
    );
    assert!(
        summarizer.seen.lock().unwrap().is_empty(),
        "the hook replaced the summarizer"
    );
}
