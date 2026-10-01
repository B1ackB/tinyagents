use std::collections::BTreeMap;

use super::*;
use crate::observability::trace_export::{SpanKind, SpanStatus, TraceSpan, trace_session_id};

fn one_turn_spans() -> Vec<TraceSpan> {
    vec![TraceSpan {
        trace_id: "sess-42".into(),
        span_id: "root".into(),
        parent_span_id: None,
        name: "agent.turn".into(),
        kind: SpanKind::Turn,
        start_unix_ms: 1_000,
        end_unix_ms: Some(2_000),
        status: SpanStatus::Ok,
        attributes: BTreeMap::new(),
        input: None,
        output: None,
    }]
}

#[test]
fn oversized_model_content_remains_structured() {
    let big = "x".repeat(MAX_MODEL_CONTENT_CHARS + 100);
    let captured = capture_model_content(&serde_json::json!([
        { "role": "system", "content": big },
        { "role": "user", "content": "the latest question" },
    ]));
    let messages = captured.as_array().expect("messages stay structured");
    assert_eq!(messages.last().unwrap()["role"], "user");
    assert_eq!(messages.last().unwrap()["content"], "the latest question");
    assert!(captured.to_string().chars().count() <= MAX_MODEL_CONTENT_CHARS + 128);
}

#[test]
fn object_model_content_is_truncated_without_panicking() {
    let captured = capture_model_content(&serde_json::json!({
        "role": "user",
        "content": "x".repeat(MAX_MODEL_CONTENT_CHARS),
    }));
    assert_eq!(captured["role"], "user");
    assert!(captured["content"].as_str().unwrap().contains("…"));
}

#[test]
fn newest_nested_message_over_budget_is_kept_as_a_preview() {
    let older = serde_json::json!({ "role": "user", "content": "earlier" });
    let newest = serde_json::json!({
        "user": { "content": [{ "type": "text", "text": "x".repeat(MAX_MODEL_CONTENT_CHARS) }] }
    });
    let captured = capture_model_content(&serde_json::json!([older, newest]));
    let messages = captured.as_array().expect("still structured");
    assert_eq!(messages.len(), 2, "marker plus the newest message preview");
    assert_eq!(messages[1]["role"], "user");
    let content = messages[1]["content"].as_str().unwrap();
    assert!(content.ends_with("…[message truncated]"));
    assert!(content.chars().count() <= MAX_MODEL_CONTENT_CHARS);
    assert_eq!(messages[0]["content"], "[1 earlier messages omitted from telemetry]");
}

#[test]
fn trace_session_id_prefers_ui_session_else_thread() {
    assert_eq!(trace_session_id(Some(99), "thread-x"), "99");
    assert_eq!(trace_session_id(None, "thread-x"), "thread-x");
}

#[test]
fn ndjson_otel_emits_one_line_per_span() {
    let spans = one_turn_spans();
    let out = spans_to_ndjson(SpanEnvelope::Otel, &spans);
    assert_eq!(out.lines().count(), spans.len());
    // Bare OTel span body has the fields directly.
    let first: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(first["trace_id"], serde_json::json!("sess-42"));
    assert_eq!(first["kind"], serde_json::json!("turn"));
}

#[test]
fn ndjson_langfuse_wraps_each_span_in_an_observation_envelope() {
    let spans = one_turn_spans();
    let out = spans_to_ndjson(SpanEnvelope::Langfuse, &spans);
    let first: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(first["type"], serde_json::json!("span-create"));
    assert_eq!(first["body"]["trace_id"], serde_json::json!("sess-42"));
}

#[test]
fn ndjson_empty_for_empty_slice() {
    assert!(spans_to_ndjson(SpanEnvelope::Otel, &[]).is_empty());
}
