//! Typed row form: tool calls, tool results and inline images are stored as
//! fields, and every read rebuilds the exact legacy `content` string, so
//! nothing downstream of the reader changes. Legacy rows (literal fixtures
//! below) keep loading, and a file may hold both generations.

use super::*;
use std::fs;
use tempfile::tempdir;
use tinytools_agent::dialect::{NativeToolCall, encode_assistant_envelope, encode_tool_envelope};

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent-id".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: Some("provider".into()),
        model: Some("model".into()),
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

fn call(id: &str, extra: Option<serde_json::Value>) -> NativeToolCall {
    NativeToolCall {
        id: id.into(),
        name: "shell".into(),
        arguments: r#"{"command":"ls"}"#.into(),
        extra_content: extra,
    }
}

fn message_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value.get("role").is_some())
        .collect()
}

fn typed_turn() -> Vec<TranscriptMessage> {
    let image = "[OH_IMAGE:data:image/png;base64,AAAA]";
    vec![
        TranscriptMessage::user(format!("look {image} please")),
        TranscriptMessage::assistant(encode_assistant_envelope(
            Some("on it"),
            &[
                call(
                    "c1",
                    Some(serde_json::json!({"google": {"thought_signature": "sig"}})),
                ),
                call("c2", None),
            ],
            None,
        )),
        TranscriptMessage::tool(encode_tool_envelope("c1", "out \"1\"")).with_id("c1"),
        TranscriptMessage::tool(encode_tool_envelope("c2", "out 2")).with_id("c2"),
        TranscriptMessage::assistant("done"),
    ]
}

#[test]
fn new_rows_store_fields_and_read_back_the_exact_legacy_content() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "typed").unwrap();
    let rows = typed_turn();
    write_transcript(&path, &rows, &meta(), None).unwrap();

    let lines = message_lines(&path);
    assert_eq!(lines[0]["v"], 2);
    assert_eq!(lines[0]["shape"], "user_parts");
    assert_eq!(lines[0]["content"], "look  please");
    assert_eq!(lines[0]["parts"][1]["type"], "image");
    assert_eq!(lines[1]["shape"], "assistant_calls");
    assert_eq!(lines[1]["content"], "on it");
    assert_eq!(lines[1]["tool_calls"][1]["id"], "c2");
    assert_eq!(lines[2]["shape"], "tool_result");
    assert_eq!(lines[2]["tool_call_id"], "c1");
    assert_eq!(lines[2]["content"], "out \"1\"");
    // A plain assistant answer is not a typed row.
    assert!(lines[4].get("v").is_none() && lines[4].get("shape").is_none());
    // No envelope string is stored anywhere in a typed row.
    for line in &lines[..4] {
        assert!(
            !line["content"].as_str().unwrap().starts_with('{'),
            "{line}"
        );
    }

    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages.len(), rows.len());
    for (read, written) in read.messages.iter().zip(&rows) {
        assert_eq!(read.role, written.role);
        assert_eq!(read.content, written.content);
        assert_eq!(read.id, written.id);
    }
    let display = read_transcript_display(&path).unwrap();
    let displayed: Vec<_> = display
        .records
        .iter()
        .filter_map(|record| match record {
            DisplayRecord::Message(message) => Some(message.message.content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        displayed,
        rows.iter().map(|r| r.content.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn non_canonical_strings_stay_legacy_rows() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "opaque").unwrap();
    let rows = vec![
        TranscriptMessage::user("hi"),
        // null content, embedded reasoning, extra key, plain JSON prose
        TranscriptMessage::assistant(encode_assistant_envelope(None, &[call("c1", None)], None)),
        TranscriptMessage::assistant(encode_assistant_envelope(
            Some("x"),
            &[call("c2", None)],
            Some("why"),
        )),
        TranscriptMessage::assistant(r#"{"answer":42}"#),
        TranscriptMessage::tool(r#"{"tool_call_id":"c1","content":"a","name":"n"}"#),
        TranscriptMessage::tool("bare result"),
        TranscriptMessage::user("no image marker, [OH_IMAGE:unterminated"),
    ];
    write_transcript(&path, &rows, &meta(), None).unwrap();
    for line in message_lines(&path) {
        assert!(line.get("v").is_none(), "{line}");
    }
    let read = read_transcript(&path).unwrap();
    for (read, written) in read.messages.iter().zip(&rows) {
        assert_eq!(read.content, written.content);
    }
}

/// Rows exactly as the pre-typed writer produced them (envelope in `content`).
const LEGACY_FILE: &str = concat!(
    r#"{"_meta":{"agent":"agent","dispatcher":"native","created":"2026-01-01T00:00:00Z","updated":"2026-01-01T00:00:00Z","turn_count":1,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
    "\n",
    r#"{"role":"user","content":"run it"}"#,
    "\n",
    r#"{"role":"assistant","content":"{\"content\":\"on it\",\"tool_calls\":[{\"arguments\":\"{}\",\"id\":\"c1\",\"name\":\"shell\"}]}","tool_calls":[{"id":"c1","name":"shell","arguments":"{}"}]}"#,
    "\n",
    r#"{"id":"c1","role":"tool","content":"{\"content\":\"ok\",\"tool_call_id\":\"c1\"}"}"#,
    "\n",
);

#[test]
fn legacy_envelope_rows_still_load_and_a_new_turn_appends_typed_rows() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "mixed").unwrap();
    fs::write(&path, LEGACY_FILE).unwrap();

    let before = read_transcript(&path).unwrap();
    assert_eq!(before.messages.len(), 3);
    assert_eq!(
        before.messages[1].content,
        r#"{"content":"on it","tool_calls":[{"arguments":"{}","id":"c1","name":"shell"}]}"#
    );
    assert_eq!(
        before.messages[2].content,
        r#"{"content":"ok","tool_call_id":"c1"}"#
    );

    // Continue the old session: the existing bytes are never rewritten.
    let prior = before.messages.clone();
    let mut next = prior.clone();
    next.push(TranscriptMessage::user("again"));
    next.push(TranscriptMessage::assistant(encode_assistant_envelope(
        Some("sure"),
        &[call("c9", None)],
        None,
    )));
    next.push(TranscriptMessage::tool(encode_tool_envelope("c9", "done")).with_id("c9"));
    append_transcript_turn(&path, &prior, &next, &meta(), None, Some("req-2")).unwrap();

    let file = fs::read_to_string(&path).unwrap();
    assert!(
        file.starts_with(LEGACY_FILE),
        "existing bytes must be untouched"
    );
    let lines = message_lines(&path);
    assert_eq!(lines.len(), 6);
    assert!(lines[..3].iter().all(|l| l.get("v").is_none()));
    assert_eq!(lines[4]["shape"], "assistant_calls");
    assert_eq!(lines[5]["shape"], "tool_result");

    let after = read_transcript(&path).unwrap();
    assert_eq!(after.messages.len(), 6);
    assert_eq!(&after.messages[..3], &before.messages[..3]);
    assert_eq!(after.messages[4].content, next[4].content);
    assert_eq!(after.messages[5].content, next[5].content);
}

#[test]
fn an_unknown_shape_or_missing_fields_keep_the_stored_content() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "future").unwrap();
    let body = concat!(
        r#"{"_meta":{"agent":"a","dispatcher":"native","created":"c","updated":"u","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
        "\n",
        r#"{"v":3,"shape":"hologram","role":"assistant","content":"kept"}"#,
        "\n",
        r#"{"v":2,"shape":"assistant_calls","role":"assistant","content":"no calls"}"#,
        "\n",
    );
    fs::write(&path, body).unwrap();
    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages[0].content, "kept");
    assert_eq!(read.messages[1].content, "no calls");
}

#[test]
fn a_future_version_or_part_kind_keeps_the_stored_content() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "future-parts").unwrap();
    let body = concat!(
        r#"{"_meta":{"agent":"a","dispatcher":"native","created":"c","updated":"u","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
        "\n",
        // A later writer reusing a known shape name under a newer version.
        r#"{"v":3,"shape":"tool_result","tool_call_id":"c1","role":"tool","content":"stored-v3"}"#,
        "\n",
        // The current version carrying a part kind this reader does not know.
        r#"{"v":2,"shape":"user_parts","role":"user","content":"stored-parts","parts":[{"kind":"document","uri":"x"}]}"#,
        "\n",
    );
    fs::write(&path, body).unwrap();
    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages.len(), 2);
    assert_eq!(read.messages[0].content, "stored-v3");
    assert_eq!(read.messages[1].content, "stored-parts");
}
