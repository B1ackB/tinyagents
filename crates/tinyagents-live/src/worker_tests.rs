use super::*;
use serde_json::json;
use tinyagents_harness::tinyinference_llm::message::{ContentBlock, ToolMessage};

fn tool(content: Vec<ContentBlock>, artifact: Option<Value>) -> Message {
    Message::Tool(ToolMessage {
        tool_call_id: "c".into(),
        content,
        trusted_verbatim: false,
        artifact,
    })
}

#[test]
fn renders_tool_messages_for_the_provider() {
    assert_eq!(tool_output(&[]), None);
    assert_eq!(
        tool_output(&[tool(
            vec![ContentBlock::Json(json!({"time": "noon"}))],
            None
        )]),
        Some(json!({"time": "noon"}))
    );
    assert_eq!(
        tool_output(&[tool(
            vec![
                ContentBlock::Text("a".into()),
                ContentBlock::Json(json!(1)),
                ContentBlock::Text(" ".into())
            ],
            None
        )]),
        Some(json!("a\n1"))
    );
    assert_eq!(
        tool_output(&[tool(vec![], Some(json!({"art": true})))]),
        Some(json!({"art": true}))
    );
    assert_eq!(
        tool_output(&[tool(vec![ContentBlock::Text(String::new())], None)]),
        Some(json!(EMPTY_OUTPUT))
    );
    assert_eq!(tool_output(&[Message::user("hello")]), Some(json!("hello")));
    assert_eq!(
        tool_output(&[Message::user(" ")]),
        Some(json!(EMPTY_OUTPUT))
    );
}

#[test]
fn an_unknown_call_has_no_recorded_outcome() {
    let recorder = OutcomeRecorder::default();
    assert_eq!(recorder.take("never"), None);
}
