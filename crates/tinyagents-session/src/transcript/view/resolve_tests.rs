//! Cross-generation row identity includes the typed structure.

use super::*;
use crate::transcript::{TranscriptMessage, TranscriptToolCall};

fn display(message: TranscriptMessage) -> DisplayRecord {
    DisplayRecord::Message(Box::new(DisplayMessage {
        message,
        interrupted: false,
        request_id: Some("req".into()),
        iteration: None,
        ts: None,
        turn_usage: None,
        reasoning_content: None,
        failure: false,
        failure_detail: None,
        background: None,
    }))
}

fn call(id: &str) -> TranscriptToolCall {
    TranscriptToolCall {
        id: id.into(),
        name: "shell".into(),
        arguments: "{}".into(),
        extra_content: None,
    }
}

#[test]
fn a_row_with_changed_typed_structure_is_not_a_retained_row() {
    let predecessor = [display(TranscriptMessage::assistant_with_calls(
        "on it",
        vec![call("c1")],
    ))];
    let successor = [display(TranscriptMessage::assistant_with_calls(
        "on it",
        vec![call("c2")],
    ))];
    let (kept, retained) = drop_retained_rows(&successor, generation_rows(&predecessor));
    assert_eq!(kept.len(), 1, "different tool calls are a new row");
    assert!(retained.is_empty());
}

#[test]
fn an_identical_typed_row_is_still_retained() {
    let row = TranscriptMessage::tool_result("c1", "ok").with_id("c1");
    let predecessor = [display(row.clone())];
    let (kept, retained) = drop_retained_rows(&[display(row)], generation_rows(&predecessor));
    assert!(kept.is_empty());
    assert_eq!(retained.len(), 1);
}
