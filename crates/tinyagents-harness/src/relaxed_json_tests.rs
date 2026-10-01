use super::*;
use serde_json::json;

#[test]
fn repairs_single_quoted_and_mismatched_keys() {
    // Captured from `llama3.2:3b` via Ollama: the model loses track of its
    // own string delimiters mid-object.
    assert_eq!(
        recover_relaxed_object(r#"{"name":"get_weather","parameters':{'city':"Paris"}}"#),
        Some(json!({ "name": "get_weather", "parameters": { "city": "Paris" } }))
    );
    // Single-quoted keys are repaired the same way, as long as the values
    // themselves are well-formed.
    assert_eq!(
        recover_relaxed_object(r#"{'city':"Paris"}"#),
        Some(json!({ "city": "Paris" }))
    );
}

/// Single-quoted *values* are deliberately **not** repaired.
///
/// A key is a short identifier, so reading `'` as a delimiter there is
/// safe. A value is free text where an apostrophe is ordinary English
/// (`"it's sunny"`), and treating those as delimiters would corrupt real
/// arguments. Such a blob stays unrecovered, the call is marked invalid,
/// and the agent loop hands the model a precise error to retry against —
/// the same path every other unrepairable blob takes.
#[test]
fn single_quoted_values_are_left_unrepaired() {
    assert_eq!(recover_relaxed_object(r#"{'city':'Paris'}"#), None);
}

#[test]
fn an_apostrophe_inside_a_well_formed_key_is_not_a_delimiter() {
    // `'` here is followed by ` fine"`, not a colon, so the key survives
    // whole rather than being truncated at the apostrophe.
    assert_eq!(
        recover_relaxed_object(r#"{"it's fine":1,bare:2}"#),
        Some(json!({ "it's fine": 1, "bare": 2 }))
    );
}

#[test]
fn quotes_unquoted_keys() {
    assert_eq!(
        recover_relaxed_object(r#"{toolkits:["discord"]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
}

#[test]
fn quotes_multiple_unquoted_keys_and_bool_value() {
    assert_eq!(
        recover_relaxed_object(r#"{include_unconnected:true,toolkits:["discord"]}"#),
        Some(json!({ "include_unconnected": true, "toolkits": ["discord"] }))
    );
}

#[test]
fn substitutes_leaked_quote_tokens_in_values() {
    assert_eq!(
        recover_relaxed_object(r#"{toolkits:[<|">discord<|">]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
}

#[test]
fn substitutes_symmetric_leaked_quote_token_variant() {
    assert_eq!(
        recover_relaxed_object(r#"{toolkits:[<|"|>discord<|"|>]}"#),
        Some(json!({ "toolkits": ["discord"] }))
    );
}

#[test]
fn peels_one_redundant_brace_layer() {
    assert_eq!(
        recover_relaxed_object(r#"{{"tool":"X","arguments":{"guild_id":"1"}}}"#),
        Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
    );
}

#[test]
fn peels_and_quotes_together() {
    assert_eq!(
        recover_relaxed_object(
            r#"{{tool:"DISCORD_LIST_CHANNELS",arguments:{"guild_id":"1470856511193616498"}}}"#
        ),
        Some(json!({
            "tool": "DISCORD_LIST_CHANNELS",
            "arguments": { "guild_id": "1470856511193616498" }
        }))
    );
}

#[test]
fn recovers_full_composio_execute_with_leaked_quote_tokens() {
    assert_eq!(
        recover_relaxed_object(
            r#"{arguments:{guild_id:<|">1470856511193616498<|">},tool:<|">DISCORD_GET_GUILD_CHANNELS<|">}"#
        ),
        Some(json!({
            "arguments": { "guild_id": "1470856511193616498" },
            "tool": "DISCORD_GET_GUILD_CHANNELS"
        }))
    );
}

#[test]
fn peels_several_redundant_layers() {
    assert_eq!(
        recover_relaxed_object(r#"{{{{tool:"X",arguments:{"guild_id":"1"}}}}}"#),
        Some(json!({ "tool": "X", "arguments": { "guild_id": "1" } }))
    );
}

#[test]
fn handles_reordered_relaxed_keys() {
    assert_eq!(
        recover_relaxed_object(r#"{{arguments:{guild_id:"1"},tool:"X"}}"#),
        Some(json!({ "arguments": { "guild_id": "1" }, "tool": "X" }))
    );
}

#[test]
fn preserves_brace_inside_string_value() {
    assert_eq!(
        recover_relaxed_object(r#"{{note:"see {ref:1}"}}"#),
        Some(json!({ "note": "see {ref:1}" }))
    );
}

#[test]
fn does_not_quote_array_elements() {
    assert_eq!(recover_relaxed_object(r#"{tags:[hi,bye]}"#), None);
}

#[test]
fn rejects_keyless_nested_object() {
    assert_eq!(recover_relaxed_object(r#"{tool:"X",{guild_id:"Y"}}"#), None);
}

#[test]
fn rejects_non_object_scalar() {
    assert_eq!(recover_relaxed_object("42"), None);
    assert_eq!(recover_relaxed_object(r#""just a string""#), None);
    assert_eq!(recover_relaxed_object("[1,2,3]"), None);
}

#[test]
fn rejects_unrecoverable_garbage() {
    assert_eq!(recover_relaxed_object(r#"{"a":1]"#), None);
    assert_eq!(recover_relaxed_object("not json at all"), None);
}

#[test]
fn already_valid_object_passes_through() {
    assert_eq!(
        recover_relaxed_object(r#"{"a":1,"b":{"c":2}}"#),
        Some(json!({ "a": 1, "b": { "c": 2 } }))
    );
}

#[test]
fn does_not_unwrap_legitimate_single_object() {
    assert_eq!(
        recover_relaxed_object(r#"{guild_id:"1",limit:50}"#),
        Some(json!({ "guild_id": "1", "limit": 50 }))
    );
}

#[test]
fn quote_bare_keys_leaves_quoted_keys_untouched() {
    assert_eq!(quote_bare_keys(r#"{"a":1,"b":2}"#), r#"{"a":1,"b":2}"#);
}

#[test]
fn normalize_leaked_quote_tokens_is_noop_without_tokens() {
    assert_eq!(normalize_leaked_quote_tokens(r#"{"a":1}"#), r#"{"a":1}"#);
}

#[test]
fn object_spans_all_respects_strings_and_trailing() {
    assert!(object_spans_all(r#"{"a":"}"}"#));
    assert!(!object_spans_all(r#"{"a":1},{"b":2}"#));
    assert!(!object_spans_all(r#"{"a":1}trailing"#));
}
