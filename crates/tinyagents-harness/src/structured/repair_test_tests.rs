use super::*;
use serde_json::json;

#[test]
fn strict_json_needs_no_repair() {
    let (value, repair) = parse_lenient(r#"{"score":4}"#).expect("strict JSON parses");
    assert_eq!(value, json!({ "score": 4 }));
    assert_eq!(repair, JsonRepair::Strict);
    assert!(!repair.is_repaired());
}

#[test]
fn removes_a_markdown_code_fence() {
    let (value, repair) =
        parse_lenient("```json\n{\"score\": 4}\n```").expect("a fenced value parses");
    assert_eq!(value, json!({ "score": 4 }));
    assert_eq!(repair, JsonRepair::CodeFence);
}

#[test]
fn slices_a_value_out_of_prose() {
    let (value, repair) = parse_lenient("Sure! Here it is: {\"score\": 4} — hope that helps.")
        .expect("a value embedded in prose parses");
    assert_eq!(value, json!({ "score": 4 }));
    assert_eq!(repair, JsonRepair::Slice);
}

#[test]
fn repairs_relaxed_json_through_the_existing_ladder() {
    let (value, repair) = parse_lenient("{score:4}").expect("unquoted keys are repaired");
    assert_eq!(value, json!({ "score": 4 }));
    assert_eq!(repair, JsonRepair::Relaxed);
}

#[test]
fn closes_a_truncated_object() {
    let (value, repair) = parse_lenient(r#"{"summary": "the model ran out of budget mid-sent"#)
        .expect("a truncated value is closed");
    assert_eq!(repair, JsonRepair::Closed);
    assert_eq!(value["summary"], "the model ran out of budget mid-sent");
}

#[test]
fn closes_nested_containers_in_the_right_order() {
    let (value, _) = parse_lenient(r#"{"items": [{"id": 1}, {"id": 2"#).expect("nesting is closed");
    assert_eq!(value["items"][1]["id"], 2);
}

#[test]
fn drops_a_dangling_comma_before_closing() {
    let (value, repair) =
        parse_lenient(r#"{"a": 1, "b": 2,"#).expect("a dangling comma is trimmed");
    assert_eq!(repair, JsonRepair::Closed);
    assert_eq!(value, json!({ "a": 1, "b": 2 }));
}

#[test]
fn refuses_text_that_is_not_json_at_all() {
    assert!(parse_lenient("I could not answer that.").is_none());
}

#[test]
fn refuses_an_unbalanced_closer() {
    // A stray `}` is corruption, not truncation; guessing here would let
    // noise masquerade as a value.
    assert!(parse_lenient("}}}").is_none());
}

#[test]
fn does_not_confuse_brackets_inside_strings() {
    let (value, _) = parse_lenient(r#"{"text": "a { and a [ walk in"#)
        .expect("brackets inside a string are literal");
    assert_eq!(value["text"], "a { and a [ walk in");
}
