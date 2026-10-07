//! Relaxed-JSON recovery as the prompt dialect uses it.
//!
//! These cases were the unit tests of the harness's own `relaxed_json`
//! module, which duplicated a subset of `tinytools_agent::repair::json`. The
//! harness now calls [`recover_whole_object`] directly, so the same shapes are
//! pinned here against that call path: the behaviour the prompt dialect relied
//! on must not change underneath it. Cases that go through
//! [`parse_relaxed_object`] also cover its single-quote retry.

use super::*;
use serde_json::json;
use tinytools_agent::repair::json::{normalize_leaked_quote_tokens, quote_bare_keys};

#[test]
fn repairs_single_quoted_and_mismatched_keys() {
    // Captured from `llama3.2:3b` via Ollama: the model loses track of its
    // own string delimiters mid-object.
    assert_eq!(
        recover_whole_object(r#"{"name":"get_weather","parameters':{'city':"Paris"}}"#),
        Some(json!({ "name": "get_weather", "parameters": { "city": "Paris" } }))
    );
    assert_eq!(
        recover_whole_object(r#"{'city':"Paris"}"#),
        Some(json!({ "city": "Paris" }))
    );
}

/// Single-quoted *values* are not repaired by the ladder itself: an apostrophe
/// in a value is ordinary text. The prompt dialect's own single-quote retry in
/// [`parse_relaxed_object`] is what recovers a fully Python-style object.
#[test]
fn single_quoted_values_are_left_to_the_dialect_retry() {
    assert_eq!(recover_whole_object(r#"{'city':'Paris'}"#), None);
    assert_eq!(
        parse_relaxed_object(r#"{'city':'Paris'}"#),
        Some(json!({ "city": "Paris" }))
    );
}

#[test]
fn an_apostrophe_inside_a_well_formed_key_is_not_a_delimiter() {
    assert_eq!(
        recover_whole_object(r#"{"it's fine":1,bare:2}"#),
        Some(json!({ "it's fine": 1, "bare": 2 }))
    );
}

#[test]
fn quotes_unquoted_keys() {
    assert_eq!(
        recover_whole_object(r#"{toolkits:["discord"]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
    assert_eq!(
        recover_whole_object(r#"{include_unconnected:true,toolkits:["discord"]}"#),
        Some(json!({ "include_unconnected": true, "toolkits": ["discord"] }))
    );
}

#[test]
fn substitutes_leaked_quote_tokens_in_values() {
    assert_eq!(
        recover_whole_object(r#"{toolkits:[<|">discord<|">]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
    assert_eq!(
        recover_whole_object(r#"{toolkits:[<|"|>discord<|"|>]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
}

#[test]
fn peels_redundant_brace_layers() {
    assert_eq!(
        recover_whole_object(r#"{{"tool":"X","arguments":{"guild_id":"1"}}}"#),
        Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
    );
    assert_eq!(
        recover_whole_object(r#"{{{{tool:"X",arguments:{"guild_id":"1"}}}}}"#),
        Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
    );
}

#[test]
fn peels_and_quotes_together() {
    assert_eq!(
        recover_whole_object(
            r#"{{tool:"DISCORD_LIST_CHANNELS",arguments:{"guild_id":"1470856511193616498"}}}"#
        ),
        Some(json!({
            "tool": "DISCORD_LIST_CHANNELS",
            "arguments": { "guild_id": "1470856511193616498" }
        }))
    );
    assert_eq!(
        recover_whole_object(r#"{{arguments:{guild_id:"1"},tool:"X"}}"#),
        Some(json!({ "arguments": { "guild_id": "1" }, "tool": "X" }))
    );
}

#[test]
fn recovers_full_composio_execute_with_leaked_quote_tokens() {
    assert_eq!(
        recover_whole_object(
            r#"{arguments:{guild_id:<|">1470856511193616498<|">},tool:<|">DISCORD_GET_GUILD_CHANNELS<|">}"#
        ),
        Some(json!({
            "arguments": { "guild_id": "1470856511193616498" },
            "tool": "DISCORD_GET_GUILD_CHANNELS"
        }))
    );
}

#[test]
fn preserves_brace_inside_string_value() {
    assert_eq!(
        recover_whole_object(r#"{{note:"see {ref:1}"}}"#),
        Some(json!({ "note": "see {ref:1}" }))
    );
}

#[test]
fn rejects_shapes_with_no_conservative_repair() {
    // Array elements are values, never keys.
    assert_eq!(recover_whole_object(r#"{tags:[hi,bye]}"#), None);
    // A keyless nested object is not a redundant wrapper.
    assert_eq!(recover_whole_object(r#"{tool:"X",{guild_id:"Y"}}"#), None);
    assert_eq!(recover_whole_object(r#"{"a":1]"#), None);
    assert_eq!(recover_whole_object("not json at all"), None);
}

#[test]
fn rejects_non_object_values() {
    assert_eq!(recover_whole_object("42"), None);
    assert_eq!(recover_whole_object(r#""just a string""#), None);
    assert_eq!(recover_whole_object("[1,2,3]"), None);
    assert_eq!(parse_relaxed_object("[1,2,3]"), None);
}

#[test]
fn passes_valid_and_legitimate_single_objects_through() {
    assert_eq!(
        recover_whole_object(r#"{"a":1,"b":{"c":2}}"#),
        Some(json!({ "a": 1, "b": { "c": 2 } }))
    );
    assert_eq!(
        recover_whole_object(r#"{guild_id:"1",limit:50}"#),
        Some(json!({ "guild_id": "1", "limit": 50 }))
    );
}

/// The old harness ladder had no trailing-noise rung; `recover_whole_object`
/// must not have one either, or prose that merely starts with an object
/// would be dispatched as a call.
#[test]
fn rejects_an_object_followed_by_trailing_text() {
    assert_eq!(
        recover_whole_object(r#"{"name":"shell","arguments":{}} explanation follows"#),
        None
    );
    assert_eq!(
        parse_relaxed_object(r#"{name:"shell",arguments:{}} explanation follows"#),
        None
    );
    assert!(parse_bare_tool_call(r#"{"name":"shell","arguments":{}} then {more}"#).is_none());
}

#[test]
fn helpers_leave_well_formed_input_untouched() {
    assert_eq!(quote_bare_keys(r#"{"a":1,"b":2}"#), r#"{"a":1,"b":2}"#);
    assert_eq!(normalize_leaked_quote_tokens(r#"{"a":1}"#), r#"{"a":1}"#);
}
