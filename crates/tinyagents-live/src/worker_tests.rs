use super::*;

#[test]
fn reads_the_first_message_text() {
    assert_eq!(message_text(&[]), None);
    assert_eq!(
        message_text(&[Message::user("hello")]),
        Some("hello".to_string())
    );
}

#[test]
fn an_unknown_call_has_no_recorded_outcome() {
    let recorder = OutcomeRecorder::default();
    assert_eq!(recorder.take("never"), None);
}
