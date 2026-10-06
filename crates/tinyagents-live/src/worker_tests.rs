use super::*;

#[test]
fn records_outcomes_from_tool_events() {
    let recorder = OutcomeRecorder::default();
    let record = |event| EventRecord {
        id: tinyagents_harness::ids::EventId::new("e".to_string()),
        offset: 0,
        event,
    };
    recorder.on_event(&record(AgentEvent::ToolFailed {
        call_id: "c1".to_string().into(),
        tool_name: "t".into(),
        error: "boom".into(),
    }));
    assert_eq!(recorder.take("c1"), Some(true));
    assert_eq!(recorder.take("c1"), None);
    assert_eq!(recorder.take("never"), None);
}

#[test]
fn reads_the_first_message_text() {
    assert_eq!(message_text(&[]), None);
    assert_eq!(
        message_text(&[Message::user("hello")]),
        Some("hello".to_string())
    );
}
