//! `mode: "inline"` contracts for [`SubAgentTool`]: the child is awaited in
//! the same tool call and its final result is returned directly.

use std::sync::Arc;

use serde_json::{Value, json};

use super::test::BlockedModel;
use super::{ChildDataPolicy, SubAgent, SubAgentJobStatus, SubAgentTool};
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::ids::CallId;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::tool::ToolDispatch;
use tinyinference_llm::providers::MockModel;

fn constant_tool(answer: &str) -> SubAgentTool<(), ()> {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant(answer)));
    SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
}

async fn call(
    tool: &SubAgentTool<(), ()>,
    parent: &RunContext<()>,
    args: Value,
) -> tinytools::ToolResult {
    ToolDispatch::<(), ()>::execute(
        tool,
        &(),
        CallId::new("call-inline"),
        args,
        tinytools::ToolCallOptions::default(),
        parent,
    )
    .await
    .expect("tool call returns a result")
}

fn json_of(result: &tinytools::ToolResult) -> Value {
    serde_json::from_str(&result.output()).expect("tool result is JSON")
}

#[tokio::test]
async fn inline_mode_returns_the_final_output_in_the_same_call() {
    let tool = constant_tool("inline answer");
    let parent = RunContext::new(RunConfig::new("parent"), ());

    let result = call(&tool, &parent, json!({"input": "work", "mode": "inline"})).await;

    assert!(!result.is_error);
    let payload = json_of(&result);
    assert_eq!(payload["status"], "completed");
    assert_eq!(payload["output"], "inline answer");
    assert_eq!(payload["tool_call_id"], "call-inline");
    let job_id = payload["job_id"].as_str().expect("job id");
    assert!(
        payload["subagent_run_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("worker-d1-parent-"))
    );
    let job = tool
        .job_registry()
        .get_owned(job_id, parent.instance_id())
        .expect("inline run is registered");
    assert_eq!(job.status, SubAgentJobStatus::Completed);
}

#[tokio::test]
async fn background_remains_the_default_mode() {
    let tool = constant_tool("later");
    let parent = RunContext::new(RunConfig::new("parent"), ());

    for args in [
        json!({"input": "work"}),
        json!({"input": "work", "mode": "background"}),
    ] {
        let payload = json_of(&call(&tool, &parent, args).await);
        assert_eq!(payload["status"], "queued");
        assert!(payload.get("output").is_none());
    }
}

#[tokio::test]
async fn unknown_mode_is_a_recoverable_tool_error() {
    let tool = constant_tool("unused");
    let parent = RunContext::new(RunConfig::new("parent"), ());

    let result = call(&tool, &parent, json!({"input": "work", "mode": "sideways"})).await;

    assert!(result.is_error);
    assert!(result.output().contains("mode"));
    assert!(tool.job_registry().list().is_empty(), "nothing was spawned");
}

#[tokio::test]
async fn inline_mode_enforces_the_depth_limit_like_background() {
    let tool = constant_tool("unused");
    let parent = RunContext::new(RunConfig::new("parent").with_max_depth(0), ());

    let result = call(&tool, &parent, json!({"input": "work", "mode": "inline"})).await;

    assert!(result.is_error);
    assert!(result.output().contains("delegated-agent limit signal"));
}

#[tokio::test]
async fn inline_mode_reports_a_cancelled_child() {
    let tool = constant_tool("unused");
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let parent = RunContext::new(RunConfig::new("parent"), ()).with_cancellation(cancellation);

    let result = call(&tool, &parent, json!({"input": "work", "mode": "inline"})).await;

    assert!(result.is_error);
    assert_eq!(json_of(&result)["status"], "cancelled");
}

#[tokio::test]
async fn inline_mode_observes_parent_cancellation_mid_run() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let mut harness = AgentHarness::new();
    harness.register_model(
        "blocked",
        Arc::new(BlockedModel {
            started: started.clone(),
            release,
        }),
    );
    let tool = SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    );
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let cancellation = parent.cancellation.clone();

    let (result, ()) = tokio::join!(
        call(&tool, &parent, json!({"input": "work", "mode": "inline"})),
        async {
            started.notified().await;
            cancellation.cancel();
        }
    );

    assert!(result.is_error);
    assert_eq!(json_of(&result)["status"], "cancelled");
}

#[test]
fn schema_advertises_the_mode_argument() {
    let tool = constant_tool("unused");
    let schema = ToolDispatch::<(), ()>::tool(&tool).parameters_schema();
    assert_eq!(
        schema["properties"]["mode"]["enum"],
        json!(["background", "inline"])
    );
    assert_eq!(schema["required"], json!(["input"]));
}
