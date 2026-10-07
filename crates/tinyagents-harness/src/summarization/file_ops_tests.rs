use super::*;
use serde_json::json;
use tinyinference_llm::message::AssistantMessage;
use tinyinference_llm::tool::ToolCall;

fn calls(calls: Vec<ToolCall>) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: calls,
        usage: None,
        origin: None,
    })
}

fn extract(messages: &[Message]) -> FileOperations {
    extract_file_operations(messages, &DefaultFileOpExtractor)
}

#[test]
fn common_path_argument_names_are_recognized() {
    let ops = extract(&[calls(vec![
        ToolCall::new("1", "read_file", json!({"path": "a.rs"})),
        ToolCall::new("2", "open", json!({"file": "b.rs"})),
        ToolCall::new("3", "view", json!({"file_path": "c.rs"})),
        ToolCall::new("4", "read_many", json!({"paths": ["d.rs", "e.rs"]})),
    ])]);
    assert_eq!(
        ops.read_only(),
        vec!["a.rs", "b.rs", "c.rs", "d.rs", "e.rs"]
    );
    assert!(ops.modified().is_empty());
}

#[test]
fn mutating_tools_are_modifications_and_win_over_reads() {
    let ops = extract(&[calls(vec![
        ToolCall::new("1", "read_file", json!({"path": "a.rs"})),
        ToolCall::new("2", "edit_file", json!({"path": "a.rs"})),
        ToolCall::new("3", "write", json!({"file_path": "new.rs"})),
        ToolCall::new("4", "read_file", json!({"path": "b.rs"})),
    ])]);
    assert_eq!(ops.modified(), vec!["a.rs", "new.rs"]);
    assert_eq!(ops.read_only(), vec!["b.rs"]);
}

#[test]
fn calls_without_path_arguments_and_invalid_calls_are_ignored() {
    let mut bad = ToolCall::new("3", "read_file", json!("{not json"));
    bad.invalid = Some("parse".into());
    let ops = extract(&[
        Message::user("path: nope.rs"),
        calls(vec![
            ToolCall::new("1", "search", json!({"query": "x"})),
            ToolCall::new("2", "read_file", json!({"path": ""})),
            bad,
        ]),
    ]);
    assert!(ops.is_empty());
}

#[test]
fn an_extractor_can_be_plugged_in() {
    struct Custom;
    impl FileOpExtractor for Custom {
        fn extract(&self, call: &ToolCall, ops: &mut FileOperations) {
            if call.name == "open_doc" {
                ops.add_modified(call.arguments["doc"].as_str().unwrap_or_default());
            }
        }
    }
    let ops = extract_file_operations(
        &[calls(vec![ToolCall::new(
            "1",
            "open_doc",
            json!({"doc": "x.md"}),
        )])],
        &Custom,
    );
    assert_eq!(ops.modified(), vec!["x.md"]);
}

#[test]
fn sections_render_and_round_trip_through_summary_text() {
    let mut ops = FileOperations::default();
    ops.add_read("b.rs");
    ops.add_read("a.rs");
    ops.add_modified("m.rs");
    let text = append_file_sections("The summary.", &ops);
    assert_eq!(
        text,
        "The summary.\n\n<read-files>\na.rs\nb.rs\n</read-files>\n\n<modified-files>\nm.rs\n</modified-files>"
    );

    let (body, parsed) = split_file_sections(&text);
    assert_eq!(body, "The summary.");
    assert_eq!(parsed, ops);
}

#[test]
fn an_empty_set_adds_nothing() {
    assert_eq!(append_file_sections("S", &FileOperations::default()), "S");
}

#[test]
fn merging_unions_and_modified_still_wins() {
    let mut a = FileOperations::default();
    a.add_read("x.rs");
    let mut b = FileOperations::default();
    b.add_modified("x.rs");
    b.add_read("y.rs");
    a.merge(&b);
    assert_eq!(a.modified(), vec!["x.rs"]);
    assert_eq!(a.read_only(), vec!["y.rs"]);
}
