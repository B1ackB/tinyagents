//! Policy contracts for [`SubAgentTool`]: timeout, retry, result policy, role.

use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use super::test::BlockedModel;
use crate::subagent::{IncompleteKind, ResultPolicy, SubAgentPolicy, SubagentRole};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::ids::CallId;
use tinyagents_harness::retry::RetryPolicy;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::tool::ToolDispatch;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::providers::MockModel;

fn tool_over(harness: AgentHarness<(), ()>) -> SubAgentTool<(), ()> {
    SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
}

fn constant(answer: &str) -> AgentHarness<(), ()> {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant(answer)));
    harness
}

async fn call_inline(tool: &SubAgentTool<(), ()>) -> tinytools::ToolResult {
    let parent = RunContext::new(RunConfig::new("parent"), ());
    ToolDispatch::<(), ()>::execute(
        tool,
        &(),
        CallId::new("c"),
        json!({"input": "work", "mode": "inline"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .expect("tool call returns a result")
}

fn payload(result: &tinytools::ToolResult) -> Value {
    serde_json::from_str(&result.output()).expect("JSON payload")
}

#[tokio::test]
async fn timeout_cancels_the_child_and_marks_the_job_incomplete() {
    let mut harness = AgentHarness::new();
    harness.register_model(
        "child",
        Arc::new(BlockedModel {
            started: Arc::new(tokio::sync::Semaphore::new(0)),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
        }),
    );
    let tool = tool_over(harness)
        .with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(50)));
    let result = call_inline(&tool).await;
    assert!(result.is_error);
    let payload = payload(&result);
    assert_eq!(payload["status"], "incomplete");
    assert_eq!(payload["incomplete_kind"], "timeout");
    let job = &tool.job_registry().list()[0];
    assert_eq!(job.status, SubAgentJobStatus::Incomplete);
    assert!(job.status.is_terminal());
}

struct FailingModel(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(tinyinference_llm::Error::Model(
            "connection reset by peer".into(),
        ))
    }
}

async fn failing_calls(max_attempts: usize) -> usize {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(FailingModel(calls.clone())));
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        ..RunPolicy::default()
    });
    let tool = tool_over(harness).with_policy(
        SubAgentPolicy::default().with_retry(
            RetryPolicy::default()
                .with_max_attempts(max_attempts)
                .with_backoff_sleep(false),
        ),
    );
    let result = call_inline(&tool).await;
    assert!(result.is_error);
    calls.load(Ordering::SeqCst)
}

#[tokio::test]
async fn a_retryable_failure_that_ran_no_tools_is_retried_with_a_fresh_child() {
    let once = failing_calls(1).await;
    let thrice = failing_calls(3).await;
    assert!(once >= 1);
    assert_eq!(thrice, once * 3, "three attempts, each a fresh child run");
}

#[tokio::test]
async fn result_policy_trims_the_job_output_and_reports_schema_errors() {
    let tool = tool_over(constant("0123456789")).with_result_policy(
        ResultPolicy::new()
            .with_max_chars(4)
            .with_schema(json!({"type": "object"})),
    );
    let result = call_inline(&tool).await;
    assert!(
        !result.is_error,
        "a schema mismatch is reported, not failed"
    );
    let payload = payload(&result);
    assert_eq!(payload["status"], "completed");
    assert!(
        payload["output"]
            .as_str()
            .unwrap()
            .contains("6 chars omitted")
    );
    assert!(
        payload["schema_error"]
            .as_str()
            .unwrap()
            .contains("not valid JSON")
    );
}

#[tokio::test]
async fn default_policies_leave_the_output_untouched() {
    let result = call_inline(&tool_over(constant("full text"))).await;
    let payload = payload(&result);
    assert_eq!(payload["output"], "full text");
    assert!(payload.get("schema_error").is_none() && payload.get("artifacts").is_none());
}

#[tokio::test]
async fn a_leaf_refuses_to_spawn_when_its_harness_exposes_delegation_tools() {
    let mut harness = constant("x");
    harness.register_tool_dispatch(Arc::new(crate::subagent::SubAgentJobsTool::new(
        SubAgentJobRegistry::new(),
    )));
    let tool = tool_over(harness).with_role(SubagentRole::Leaf);
    let result = call_inline(&tool).await;
    assert!(result.is_error);
    assert!(result.output().contains("leaf"));
    assert!(tool.job_registry().list().is_empty(), "nothing was spawned");

    let ok = tool_over(constant("x")).with_role(SubagentRole::Leaf);
    assert!(!call_inline(&ok).await.is_error);
}

#[test]
fn incomplete_kind_serializes_as_snake_case() {
    assert_eq!(
        serde_json::to_value(IncompleteKind::BudgetExceeded).unwrap(),
        "budget_exceeded"
    );
}
