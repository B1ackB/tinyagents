//! Tests for [`ModelSummarizer`], [`FaultTolerantCachingSummarizer`] and the
//! context-window-aware policy builders.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use tinyinference_llm::message::Message;

use super::{
    DEFAULT_SUMMARIZE_KEEP_LAST, DEFAULT_SUMMARIZE_THRESHOLD_FRACTION,
    FaultTolerantCachingSummarizer, ModelSummarizer, SummarizationPolicy, Summarizer,
    SummaryRecord, summarization_policy, summarization_policy_with,
};
use crate::error::{Result, TinyAgentsError};
use crate::testkit::ScriptedModel;

#[test]
fn policy_is_context_window_aware_at_the_default_threshold() {
    let policy = summarization_policy(200_000);
    assert_eq!(policy.context_window, Some(200_000));
    assert_eq!(
        policy.threshold_fraction,
        DEFAULT_SUMMARIZE_THRESHOLD_FRACTION
    );
    assert_eq!(policy.keep_last, DEFAULT_SUMMARIZE_KEEP_LAST);
}

#[test]
fn default_threshold_leaves_headroom_below_the_window() {
    assert!(
        DEFAULT_SUMMARIZE_THRESHOLD_FRACTION > 0.0 && DEFAULT_SUMMARIZE_THRESHOLD_FRACTION < 1.0
    );
    let policy = summarization_policy(100_000);
    let effective = (policy.context_window.unwrap() as f64 * policy.threshold_fraction) as u64;
    assert_eq!(effective, 90_000);
}

#[test]
fn explicit_threshold_and_tail_override_the_defaults() {
    let policy = summarization_policy_with(10_000, 0.5, 3);
    assert_eq!(policy.threshold_fraction, 0.5);
    assert_eq!(policy.keep_last, 3);
}

#[tokio::test]
async fn model_summarizer_wraps_the_reply_and_records_provenance() {
    let model = Arc::new(ScriptedModel::replies(vec!["  the gist  "]));
    let summarizer = ModelSummarizer::new(model.clone(), "m-1").with_threshold_fraction(0.8);
    let messages = vec![Message::user("hello there"), Message::assistant("hi back")];
    let record = summarizer.summarize(&messages).await.unwrap();

    assert!(
        record
            .summary
            .text()
            .contains("=== Conversation Summary (compacted) ===")
    );
    assert!(record.summary.text().contains("the gist"));
    assert_eq!(record.provenance.source_ids, vec!["msg-0", "msg-1"]);
    assert!(record.provenance.reason.contains("80%"));
    assert!(record.provenance.reason.contains("m-1"));
    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].messages[1].text().contains("user: hello there"));
}

#[tokio::test]
async fn model_summarizer_rejects_empty_input_and_empty_replies() {
    let summarizer = ModelSummarizer::new(Arc::new(ScriptedModel::replies(vec!["   "])), "m");
    assert!(summarizer.summarize(&[]).await.is_err());
    let err = summarizer
        .summarize(&[Message::user("x")])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("empty response"));
}

struct CountingFailing(Arc<AtomicUsize>);

#[async_trait]
impl Summarizer for CountingFailing {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::Model("boom".into()))
    }
}

fn long_slice() -> Vec<Message> {
    (0..6)
        .map(|i| Message::user(format!("message {i} {}", "word ".repeat(40))))
        .collect()
}

#[tokio::test]
async fn failure_falls_back_to_a_deterministic_trim_and_trips_the_breaker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(CountingFailing(calls.clone())), &policy);

    let first = guarded.summarize(&long_slice()).await.unwrap();
    assert!(first.summary.text().contains("deterministic trim"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // A different slice: the breaker is open, so the inner summarizer is skipped.
    let other = vec![Message::user("something else entirely")];
    let second = guarded.summarize(&other).await.unwrap();
    assert!(second.provenance.reason.contains("circuit breaker open"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_identical_slice_is_served_from_the_cache() {
    let model = Arc::new(ScriptedModel::replies(vec!["one summary"]));
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded = FaultTolerantCachingSummarizer::new(
        Box::new(ModelSummarizer::new(model.clone(), "m")),
        &policy,
    );
    let slice = long_slice();
    let a = guarded.summarize(&slice).await.unwrap();
    let b = guarded.summarize(&slice).await.unwrap();
    assert_eq!(a.summary.text(), b.summary.text());
    assert_eq!(
        model.requests().len(),
        1,
        "second call must not reach the model"
    );
}

#[tokio::test]
async fn the_fallback_front_drops_oldest_messages_to_fit_its_budget() {
    let policy = SummarizationPolicy::default().with_context_window(1_000);
    let guarded =
        FaultTolerantCachingSummarizer::new(Box::new(CountingFailing(Arc::default())), &policy);
    // A 1_000-token window gives a floor budget of 1_024 tokens; oversize the slice.
    let big: Vec<Message> = (0..40)
        .map(|i| Message::user(format!("m{i} {}", "x".repeat(400))))
        .collect();
    let record = guarded.summarize(&big).await.unwrap();
    assert!(record.summary.text().contains("older message(s) dropped"));
    assert!(record.summary.text().contains("m39"));
}
