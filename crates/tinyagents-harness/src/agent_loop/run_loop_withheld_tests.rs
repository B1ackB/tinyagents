//! A tool call written on a turn that could not take one: scrubbed from the
//! answer, never run, and re-prompted once (`TextRecovery::withholding`).
//!
//! The replies below are what DeepSeek V4 returned in bench captures when a
//! request withdrew its tools over a transcript full of tool calls.

use std::sync::Arc;

use super::*;
use crate::agent_loop::dialect::{TextRecovery, withhold_text_calls};
use crate::context::RunConfig;
use crate::limits::RunLimits;
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{ScriptedModel, StreamingMock};
use tinyinference_llm::message::{ContentBlock, MessageDelta};
use tinyinference_llm::model::{ModelResponse, ModelStreamItem};

const DSML_CALL: &str = "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name=\"shell\">\n\
<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">cd /app && grep -rn jsonpath pkg/</｜｜DSML｜｜ parameter>\n\
</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>";

fn leaked_reply() -> String {
    format!("I'll implement this from the spec. Let me first check the modules.\n\n{DSML_CALL}")
}

fn has_markup(text: &str) -> bool {
    text.contains("DSML") || text.contains("invoke name")
}

#[test]
fn a_withheld_call_is_scrubbed_counted_and_never_becomes_a_call() {
    let recovery = TextRecovery::withholding();
    let mut response = ModelResponse::assistant(leaked_reply());
    response.message.content.insert(
        0,
        ContentBlock::Thinking {
            text: "check the modules first".into(),
            signature: None,
        },
    );

    withhold_text_calls(&mut response, &CallId::new("model-1"), &recovery.dropped);

    assert!(response.message.tool_calls.is_empty(), "nothing is dispatched");
    assert_eq!(recovery.dropped.withheld(), 1);
    assert!(!has_markup(&response.text()));
    assert!(response.text().contains("implement this from the spec"));
    assert!(
        matches!(response.message.content[0], ContentBlock::Thinking { .. }),
        "reasoning survives the scrub"
    );
}

#[test]
fn withholding_leaves_a_quoted_example_and_a_json_answer_alone() {
    let recovery = TextRecovery::withholding();
    for text in [
        "The format is:\n```xml\n<invoke name=\"shell\"><parameter name=\"command\">ls</parameter></invoke>\n```",
        "{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}",
    ] {
        let mut response = ModelResponse::assistant(text);
        withhold_text_calls(&mut response, &CallId::new("model-1"), &recovery.dropped);
        assert_eq!(response.text(), text);
    }
    assert_eq!(recovery.dropped.withheld(), 0);
}

fn harness_with(model: Arc<dyn tinyinference_llm::model::ChatModel<()>>, max_calls: usize) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_model_calls(max_calls),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn a_call_on_a_turn_without_tools_is_withheld_and_reprompted() {
    let model = Arc::new(ScriptedModel::replies(vec![
        leaked_reply(),
        "The jsonpath module is not registered yet; here is the plan.".to_string(),
    ]));
    let harness = harness_with(model.clone(), 5);

    let run = harness
        .invoke_default(&(), vec![Message::user("Answer now from what you have.")])
        .await
        .expect("run succeeds");

    assert_eq!(
        run.text().as_deref(),
        Some("The jsonpath module is not registered yet; here is the plan.")
    );
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "one re-prompt");
    let retry = &requests[1].messages;
    assert_eq!(
        retry.last().map(Message::text).as_deref(),
        Some(WITHHELD_TOOL_CALL_NUDGE)
    );
    assert!(
        retry.iter().all(|m| !has_markup(&m.text())),
        "the leaked row is not replayed to the model"
    );
    assert!(run.messages.iter().all(|m| !has_markup(&m.text())));
}

#[tokio::test]
async fn with_no_call_left_the_scrubbed_reply_stands_without_markup() {
    let model = Arc::new(ScriptedModel::replies(vec![leaked_reply()]));
    let harness = harness_with(model.clone(), 1);

    let run = harness
        .invoke_default(&(), vec![Message::user("Answer now.")])
        .await
        .expect("run succeeds");

    assert_eq!(model.requests().len(), 1, "no call was left for a re-prompt");
    let text = run.text().unwrap_or_default();
    assert!(!has_markup(&text));
    assert!(text.contains("implement this from the spec"));
}

#[tokio::test]
async fn a_streamed_call_on_a_turn_without_tools_never_reaches_the_answer() {
    let chunks = ["I'll check first. ", "<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name=\"shell\">\n",
        "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">ls</｜｜DSML｜｜ parameter>\n",
        "</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>"];
    let mut items = vec![ModelStreamItem::Started];
    items.extend(chunks.iter().map(|text| ModelStreamItem::MessageDelta(MessageDelta::text(*text))));
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(chunks.concat())));
    // The mock replays the same leak on every call, so the run spends its
    // re-prompts and then keeps the scrubbed reply.
    let model = Arc::new(StreamingMock::new(items));
    let harness = harness_with(model.clone(), 10);

    let run = harness
        .invoke_streaming(&(), (), RunConfig::new("withheld-stream"), vec![Message::user("Answer now.")])
        .await
        .expect("streaming run succeeds");

    let text = run.text().unwrap_or_default();
    assert!(!has_markup(&text), "got {text:?}");
    assert!(text.contains("I'll check first."));
    assert_eq!(
        model.call_count(),
        1 + u64::from(RunPolicy::default().dropped_tool_call_nudges),
        "re-prompted up to the nudge budget, then answered"
    );
    assert!(run.messages.iter().all(|m| !has_markup(&m.text())));
}
