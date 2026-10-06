use std::sync::Arc;

use super::recover_text_dialect_calls;
use crate::agent_loop::dialect::TextRecovery;
use crate::context::{RunConfig, RunContext};
use crate::ids::CallId;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolSchema;

fn offered(names: &[&str]) -> TextRecovery {
    TextRecovery {
        offered: Arc::new(
            names
                .iter()
                .map(|name| ToolSchema::new(*name, "", serde_json::json!({"type": "object"})))
                .collect(),
        ),
        registry: None,
        dropped: Arc::default(),
        withhold: false,
    }
}

#[test]
fn text_dialect_markup_is_not_recovered_when_the_request_offered_no_tools() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
    let mut response = ModelResponse::assistant(
        "<tool_call><name>shell</name><arguments>{\"command\":\"id\"}</arguments></tool_call>",
    );

    recover_text_dialect_calls(
        &ctx,
        &mut response,
        &CallId::new("model-1"),
        &TextRecovery::default(),
    );

    assert!(response.message.tool_calls.is_empty());
    assert!(response.text().contains("<tool_call>"));
}

/// I-2 regression: `RunPolicy::text_dialect_recovery` resolving to off
/// (what `Auto` yields for a model whose profile reports native tool
/// calling) is represented as an empty `TextRecovery`, exactly like a
/// turn that offered no tools — so `<tool_call>` markup the model merely
/// quoted must not be executed.
#[test]
fn text_dialect_markup_is_not_recovered_when_the_policy_disables_it() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
    let mut response = ModelResponse::assistant(
        r#"<tool_call>{"name": "shell", "arguments": {"command": "id"}}</tool_call>"#,
    );

    recover_text_dialect_calls(
        &ctx,
        &mut response,
        &CallId::new("model-1"),
        &TextRecovery::default(),
    );

    assert!(response.message.tool_calls.is_empty());
    assert!(response.text().contains("<tool_call>"));
}

/// I-2 regression: a final answer that quotes `<tool_call>` markup inside
/// a language-tagged fenced code block must never be executed, even when
/// recovery is otherwise enabled and tools were offered. (A *bare* fence
/// is not protected: `tinytools-agent` treats it as a wrapped call.)
#[test]
fn text_dialect_markup_inside_a_fenced_code_block_is_never_recovered() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
    let mut response = ModelResponse::assistant(
        "Here is the format:\n```xml\n<tool_call>{\"name\": \"shell\", \"arguments\": {}}</tool_call>\n```\n",
    );

    recover_text_dialect_calls(
        &ctx,
        &mut response,
        &CallId::new("model-1"),
        &offered(&["shell"]),
    );

    assert!(
        response.message.tool_calls.is_empty(),
        "markup quoted inside a fenced code block must not become a real call"
    );
    assert!(response.text().contains("<tool_call>"));
}

/// Sanity check for the fence policy: markup outside any fence is still
/// recovered when the policy and tool offer both allow it.
#[test]
fn text_dialect_markup_outside_a_fenced_code_block_is_recovered() {
    let ctx: RunContext<()> = RunContext::new(RunConfig::new("recovery-test"), ());
    let mut response = ModelResponse::assistant(
        r#"<tool_call>{"name": "shell", "arguments": {"command": "id"}}</tool_call>"#,
    );

    recover_text_dialect_calls(
        &ctx,
        &mut response,
        &CallId::new("model-1"),
        &offered(&["shell"]),
    );

    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].name, "shell");
}

#[test]
fn every_provider_spelling_of_a_length_stop_is_recognised() {
    for reason in ["length", "max_tokens", "MAX_TOKENS"] {
        assert!(super::is_length_stop(Some(reason)), "{reason}");
    }
    for reason in [None, Some("stop"), Some("tool_calls"), Some("end_turn")] {
        assert!(!super::is_length_stop(reason), "{reason:?}");
    }
}

#[test]
fn recovered_call_ids_are_told_apart_from_native_ones() {
    use crate::agent_loop::dialect::{is_recovered_tool_call_id, recovered_tool_call_id};

    let short = CallId::new("model-3");
    let long = CallId::new(format!("openhuman-session-{}-model-3", "x".repeat(60)));
    for model_call in [&short, &long] {
        let minted = recovered_tool_call_id(model_call, 2);
        assert!(is_recovered_tool_call_id(model_call, &minted), "{minted}");
        // Another model call's id does not match, nor does a native id.
        assert!(!is_recovered_tool_call_id(&CallId::new("model-4"), &minted));
        assert!(!is_recovered_tool_call_id(model_call, "toolu_01abc"));
        assert!(!is_recovered_tool_call_id(model_call, "call_1"));
    }
    assert!(!is_recovered_tool_call_id(&short, "model-3-tool-"));
    assert!(!is_recovered_tool_call_id(&short, "model-3-tool-x"));
}

#[test]
fn only_the_last_native_and_invalid_calls_are_possibly_truncated() {
    use tinyinference_llm::tool::ToolCall;

    let model_call = CallId::new("model-1");
    let recovered = crate::agent_loop::dialect::recovered_tool_call_id(&model_call, 1);
    let calls = vec![
        ToolCall::new("n1", "a", serde_json::json!({})),
        ToolCall::invalid("n2", "a", "{", "eof"),
        ToolCall::new("n3", "a", serde_json::json!({})),
        ToolCall::new(recovered, "a", serde_json::json!({})),
    ];
    let mut ids: Vec<String> = super::truncated_call_ids(&calls, &model_call)
        .into_iter()
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["n2", "n3"]);
}
