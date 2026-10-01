//! Typed rows: tool calls, tool results and inline images are fields of the
//! in-memory row and of the stored line, and `content` is plain text. Legacy
//! rows (a native envelope or `[OH_IMAGE:]` marker inside `content`, literal
//! fixtures below) keep loading, normalized into the same typed fields, and a
//! file may hold both generations. [`TranscriptMessage::legacy_content`]
//! rebuilds the old string for compatibility adapters.

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

/// The same turn as typed rows.
fn typed_turn() -> Vec<TranscriptMessage> {
    let image = "data:image/png;base64,AAAA";
    vec![
        TranscriptMessage::user_with_parts(vec![
            TranscriptPart::Text {
                text: "look ".into(),
            },
            TranscriptPart::Image { url: image.into() },
            TranscriptPart::Text {
                text: " please".into(),
            },
        ]),
        TranscriptMessage::assistant_with_calls(
            "on it",
            vec![
                TranscriptToolCall::from(call(
                    "c1",
                    Some(serde_json::json!({"google": {"thought_signature": "sig"}})),
                )),
                TranscriptToolCall::from(call("c2", None)),
            ],
        ),
        TranscriptMessage::tool_result("c1", "out \"1\"").with_id("c1"),
        TranscriptMessage::tool_result("c2", "out 2").with_id("c2"),
        TranscriptMessage::assistant("done"),
    ]
}

/// The same turn as the legacy string rows an older host built.
fn legacy_turn() -> Vec<TranscriptMessage> {
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
fn typed_rows_store_fields_and_read_back_equal() {
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
        assert!(read.same_row_as(written), "{read:?} vs {written:?}");
    }
    let display = read_transcript_display(&path).unwrap();
    let displayed: Vec<_> = display
        .records
        .iter()
        .filter_map(|record| match record {
            DisplayRecord::Message(message) => Some(message.message.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(displayed.len(), rows.len());
    for (read, written) in displayed.iter().zip(&rows) {
        assert!(read.same_row_as(written));
    }
}

/// A host that still builds string rows gets the same typed lines, and the
/// rebuilt legacy string of a typed row is exactly the old string.
#[test]
fn string_rows_and_typed_rows_write_identical_lines() {
    let dir = tempdir().unwrap();
    let typed_path = resolve_keyed_transcript_path(dir.path(), "typed").unwrap();
    let legacy_path = resolve_keyed_transcript_path(dir.path(), "legacy").unwrap();
    write_transcript(&typed_path, &typed_turn(), &meta(), None).unwrap();
    write_transcript(&legacy_path, &legacy_turn(), &meta(), None).unwrap();
    assert_eq!(message_lines(&typed_path), message_lines(&legacy_path));
    for (typed, legacy) in typed_turn().iter().zip(legacy_turn()) {
        assert_eq!(typed.legacy_content(), legacy.content);
        assert!(
            TranscriptMessage::from_legacy(&legacy.role, legacy.content).same_row_as(
                &TranscriptMessage {
                    id: None,
                    ..typed.clone()
                }
            )
        );
    }
}

#[test]
fn non_canonical_strings_stay_legacy_lines_and_load_normalized() {
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
    let lines = message_lines(&path);
    for (line, row) in lines.iter().zip(&rows) {
        assert!(line.get("v").is_none(), "{line}");
        // Stored exactly as given.
        assert_eq!(line["content"], row.content.as_str());
    }
    let read = read_transcript(&path).unwrap();
    for (read, written) in read.messages.iter().zip(&rows) {
        assert!(
            read.same_row_as(&TranscriptMessage::from_legacy(
                &written.role,
                written.content.clone()
            )),
            "{read:?}"
        );
    }
    // The null-content envelope lifts into a calls row with empty text; the
    // one with reasoning keeps its calls (reasoning lives in metadata); plain
    // JSON prose and a bare result stay text.
    assert_eq!(read.messages[1].tool_calls.len(), 1);
    assert_eq!(read.messages[1].content, "");
    assert_eq!(read.messages[2].tool_calls[0].id, "c2");
    assert!(read.messages[3].tool_calls.is_empty());
    assert_eq!(read.messages[4].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(read.messages[5].tool_call_id, None);
    assert!(read.messages[6].parts.is_none());
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
    assert_eq!(before.messages[1].content, "on it");
    assert_eq!(before.messages[1].tool_calls.len(), 1);
    assert_eq!(
        before.messages[1].legacy_content(),
        r#"{"content":"on it","tool_calls":[{"arguments":"{}","id":"c1","name":"shell"}]}"#
    );
    assert_eq!(before.messages[2].content, "ok");
    assert_eq!(before.messages[2].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(
        before.messages[2].legacy_content(),
        r#"{"content":"ok","tool_call_id":"c1"}"#
    );

    // Continue the old session: the existing bytes are never rewritten.
    let prior = before.messages.clone();
    let mut next = prior.clone();
    next.push(TranscriptMessage::user("again"));
    next.push(TranscriptMessage::assistant_with_calls(
        "sure",
        vec![TranscriptToolCall::from(call("c9", None))],
    ));
    next.push(TranscriptMessage::tool_result("c9", "done").with_id("c9"));
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
    assert!(after.messages[4].same_row_as(&next[4]));
    assert!(after.messages[5].same_row_as(&next[5]));
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
