use super::*;

#[test]
fn every_provider_spelling_of_a_length_stop_is_recognised() {
    for reason in ["length", "max_tokens", "MAX_TOKENS"] {
        assert!(is_length_stop(Some(reason)), "{reason}");
    }
    for reason in [None, Some("stop"), Some("tool_calls"), Some("end_turn")] {
        assert!(!is_length_stop(reason), "{reason:?}");
    }
}
