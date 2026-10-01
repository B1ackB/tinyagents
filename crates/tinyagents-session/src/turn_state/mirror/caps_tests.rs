use serde_json::json;

use super::*;

#[test]
fn transcript_text_is_capped_on_a_char_boundary_with_one_marker() {
    let mut text = String::new();
    // Multi-byte chars force a boundary adjustment at the cap.
    let chunk = "é".repeat(MAX_PERSISTED_TRANSCRIPT_ITEM);
    append_capped_transcript_text(&mut text, &chunk);
    assert!(text.ends_with(TRANSCRIPT_TRUNCATION_MARKER));
    assert!(text.len() <= MAX_PERSISTED_TRANSCRIPT_ITEM + TRANSCRIPT_TRUNCATION_MARKER.len());
    let before = text.clone();
    append_capped_transcript_text(&mut text, "more");
    assert_eq!(text, before, "further deltas past the cap are dropped");
    assert_eq!(text.matches(TRANSCRIPT_TRUNCATION_MARKER).count(), 1);
}

#[test]
fn transcript_text_marker_appears_when_deltas_land_exactly_on_the_cap() {
    let mut text = "a".repeat(MAX_PERSISTED_TRANSCRIPT_ITEM);
    append_capped_transcript_text(&mut text, "b");
    assert!(text.ends_with(TRANSCRIPT_TRUNCATION_MARKER));
    append_capped_transcript_text(&mut text, "");
    assert_eq!(text.matches(TRANSCRIPT_TRUNCATION_MARKER).count(), 1);
}

#[test]
fn output_cap_passes_small_drops_empty_and_truncates_large() {
    assert_eq!(cap_persisted_output(""), None);
    assert_eq!(cap_persisted_output("ok").as_deref(), Some("ok"));
    let big = "x".repeat(70 * 1024);
    let capped = cap_persisted_output(&big).unwrap();
    assert!(capped.len() <= 64 * 1024);
    assert!(capped.contains("…[truncated"));
    assert!(capped.ends_with("bytes of tool output]"));
}

#[test]
fn args_cap_keeps_small_json_drops_null_and_degrades_large_to_a_string() {
    assert_eq!(cap_persisted_args(&json!(null)), None);
    let small = json!({"q": "x"});
    assert_eq!(cap_persisted_args(&small), Some(small));
    let big = json!({"payload": "y".repeat(20 * 1024)});
    match cap_persisted_args(&big).unwrap() {
        serde_json::Value::String(s) => {
            assert!(s.ends_with("bytes of tool arguments]"));
            assert!(s.len() <= 16 * 1024 + 64);
        }
        other => panic!("expected truncated string, got {other:?}"),
    }
}
