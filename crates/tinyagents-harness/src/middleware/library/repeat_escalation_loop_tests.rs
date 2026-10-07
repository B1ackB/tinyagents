//! The staged repeat escalation inside the real agent loop: a blocked call
//! must never reach its tool, and the run must pause on the second block.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::runtime::AgentHarness;
use crate::steering::SteeringHandle;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolPolicy, ToolResult};

struct CountingTool {
    runs: Mutex<u32>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "counting tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        *self.runs.lock().unwrap() += 1;
        Ok(ToolResult::success("same answer"))
    }
}

fn repeat_call(turn: usize) -> ModelResponse {
    let mut response = ModelResponse::assistant(format!("attempt {turn}"));
    response.message = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text(format!("attempt {turn}"))],
        tool_calls: vec![ToolCall::new(
            format!("call-{turn}"),
            "lookup",
            json!({"id": 1}),
        )],
        usage: None,
        origin: None,
    };
    response
}

#[tokio::test]
async fn blocked_calls_never_execute_and_the_second_block_pauses_the_run() {
    let steering = SteeringHandle::allow_all();
    let summary: HaltSummarySlot = Arc::new(Mutex::new(None));
    let tool = Arc::new(CountingTool {
        runs: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(
            (0..12).map(repeat_call).collect::<Vec<_>>(),
        )),
    );
    harness.register_tool(tool.clone());
    harness.push_middleware(Arc::new(RepeatProgressMiddleware::new(
        steering.clone(),
        summary.clone(),
        Arc::new(|_| false),
    )));

    let ctx = RunContext::new(RunConfig::new("repeat-loop"), ()).with_steering(steering.clone());
    let run = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a halt pauses the run rather than failing it")
        .run;

    assert_eq!(
        *tool.runs.lock().unwrap(),
        4,
        "attempts 1-4 execute; 5 and 6 are answered without running"
    );
    let texts: Vec<String> = run
        .messages
        .iter()
        .filter_map(|m| matches!(m, Message::Tool(_)).then(|| m.text()))
        .collect();
    assert!(texts[2].contains("[repeat notice]"), "{texts:?}");
    assert!(texts[4].contains("not executed"), "{texts:?}");
    assert!(
        summary.lock().unwrap().is_some(),
        "the halt names its cause"
    );
}
