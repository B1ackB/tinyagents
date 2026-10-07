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
fn only_the_last_and_invalid_calls_are_possibly_truncated() {
    use tinyinference_llm::tool::ToolCall;

    let calls = vec![
        ToolCall::new("n1", "a", serde_json::json!({})),
        ToolCall::invalid("n2", "a", "{", "eof"),
        ToolCall::new("n3", "a", serde_json::json!({})),
        ToolCall::new("n3", "a", serde_json::json!({})),
    ];
    let mut positions: Vec<usize> = super::truncated_call_positions(&calls)
        .into_iter()
        .collect();
    positions.sort();
    assert_eq!(
        positions,
        vec![1, 3],
        "positional: a duplicate id does not widen it"
    );
    assert!(super::truncated_call_positions(&[]).is_empty());
}
