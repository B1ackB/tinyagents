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
    let tool = tool_over(constant(&"0123456789".repeat(20))).with_result_policy(
        ResultPolicy::new()
            .with_max_chars(100)
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

// ---- attempt ids, tool safety, background retry, token overrun, artifacts ----

use tinyagents_harness::cancel::CancellationToken;
use tinyinference_llm::usage::Usage;

struct ProbeTool {
    token: Arc<std::sync::Mutex<Option<CancellationToken>>>,
    hang: bool,
}

struct ProbeDeclaration;

#[async_trait::async_trait]
impl tinytools::Tool for ProbeDeclaration {
    fn name(&self) -> &str {
        "probe"
    }
    fn description(&self) -> &str {
        "records the child's cancellation token"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> anyhow::Result<tinytools::ToolResult> {
        unreachable!("dispatch only")
    }
}

#[async_trait::async_trait]
impl ToolDispatch<(), ()> for ProbeTool {
    fn tool(&self) -> Arc<dyn tinytools::Tool> {
        Arc::new(ProbeDeclaration)
    }
    async fn execute(
        &self,
        _: &(),
        _: CallId,
        _: Value,
        _: tinytools::ToolCallOptions,
        parent: &RunContext<()>,
    ) -> anyhow::Result<tinytools::ToolResult> {
        *self.token.lock().unwrap() = Some(parent.cancellation.clone());
        if self.hang {
            std::future::pending::<()>().await;
        }
        Ok(tinytools::ToolResult::json(json!({})))
    }
}

/// Calls `probe` first, then fails transiently once the tool result is in.
struct ToolThenFail(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl ChatModel<()> for ToolThenFail {
    async fn invoke(&self, state: &(), request: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        self.0.fetch_add(1, Ordering::SeqCst);
        if request.messages.len() <= 1 {
            MockModel::with_tool_call("probe", json!({})).invoke(state, request).await
        } else {
            Err(tinyinference_llm::Error::Model("connection reset by peer".into()))
        }
    }
}

fn probe_harness(model: Arc<dyn ChatModel<()>>, hang: bool) -> (AgentHarness<(), ()>, Arc<std::sync::Mutex<Option<CancellationToken>>>) {
    let token = Arc::new(std::sync::Mutex::new(None));
    let mut harness = AgentHarness::new();
    harness.register_model("child", model);
    harness.register_tool_dispatch(Arc::new(ProbeTool { token: token.clone(), hang }));
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        ..RunPolicy::default()
    });
    (harness, token)
}

fn retrying(attempts: usize) -> SubAgentPolicy {
    SubAgentPolicy::default().with_retry(
        RetryPolicy::default()
            .with_max_attempts(attempts)
            .with_backoff_sleep(false),
    )
}

#[tokio::test]
async fn no_retry_once_a_tool_ran_unless_the_policy_allows_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (harness, _) = probe_harness(Arc::new(ToolThenFail(calls.clone())), false);
    let tool = tool_over(harness).with_policy(retrying(3));
    assert!(call_inline(&tool).await.is_error);
    assert_eq!(calls.load(Ordering::SeqCst), 2, "one attempt: tool call, then the failure");

    let calls = Arc::new(AtomicUsize::new(0));
    let (harness, _) = probe_harness(Arc::new(ToolThenFail(calls.clone())), false);
    let tool = tool_over(harness).with_policy(retrying(3).with_retry_after_tool_calls(true));
    assert!(call_inline(&tool).await.is_error);
    assert_eq!(calls.load(Ordering::SeqCst), 6, "three attempts when allowed");
}

#[tokio::test]
async fn timeout_cancels_the_childs_own_token() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (harness, token) = probe_harness(Arc::new(ToolThenFail(calls)), true);
    let tool = tool_over(harness)
        .with_policy(SubAgentPolicy::default().with_timeout(Duration::from_millis(100)));
    let result = call_inline(&tool).await;
    assert_eq!(payload(&result)["incomplete_kind"], "timeout");
    let token = token.lock().unwrap().clone().expect("the child ran its tool");
    assert!(token.is_cancelled(), "the child observed cancellation");
}

async fn wait_terminal(tool: &SubAgentTool<(), ()>) -> SubAgentJob {
    for _ in 0..200 {
        if let Some(job) = tool.job_registry().list().into_iter().find(|j| j.status.is_terminal()) {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("job never settled");
}

#[tokio::test]
async fn background_mode_retries_and_the_job_link_follows_the_final_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(FailingModel(calls.clone())));
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default().with_max_attempts(1),
        ..RunPolicy::default()
    });
    let tool = tool_over(harness).with_policy(retrying(3));
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let queued = ToolDispatch::<(), ()>::execute(
        &tool,
        &(),
        CallId::new("c"),
        json!({"input": "work"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let first_run = payload(&queued)["subagent_run_id"].as_str().unwrap().to_owned();
    let job = wait_terminal(&tool).await;
    assert_eq!(job.status, SubAgentJobStatus::Failed);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(job.subagent_run_id.as_deref(), Some(format!("{first_run}-a2").as_str()));
}

async fn sibling_run_ids(tool: &SubAgentTool<(), ()>) -> Vec<String> {
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let mut ids = Vec::new();
    for _ in 0..2 {
        let queued = ToolDispatch::<(), ()>::execute(
            tool,
            &(),
            CallId::new("c"),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .unwrap();
        ids.push(payload(&queued)["subagent_run_id"].as_str().unwrap().to_owned());
    }
    ids
}

#[tokio::test]
async fn enabling_retries_does_not_change_sibling_child_ids() {
    let plain = sibling_run_ids(&tool_over(constant("ok"))).await;
    let with_retry = sibling_run_ids(&tool_over(constant("ok")).with_policy(retrying(4))).await;
    assert_eq!(plain, with_retry);
}

struct UsageModel;

#[async_trait::async_trait]
impl ChatModel<()> for UsageModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        let mut response = ModelResponse::assistant("long answer");
        response.usage = Some(Usage {
            output_tokens: 500,
            ..Usage::default()
        });
        Ok(response)
    }
}

#[tokio::test]
async fn a_token_overrun_keeps_the_output_applies_the_result_policy_and_is_incomplete() {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(UsageModel));
    let tool = tool_over(harness)
        .with_policy(
            SubAgentPolicy::default()
                .with_budget(crate::subagent::SubAgentBudget::unlimited().with_max_output_tokens(100)),
        )
        .with_result_policy(ResultPolicy::new().with_schema(json!({"type": "object"})));
    let result = call_inline(&tool).await;
    assert!(result.is_error);
    let payload = payload(&result);
    assert_eq!(payload["status"], "incomplete");
    assert_eq!(payload["incomplete_kind"], "budget_exceeded");
    assert_eq!(payload["output"], "long answer", "completed work is kept");
    assert!(payload["schema_error"].is_string(), "the result policy still ran");
}

#[tokio::test]
async fn artifact_overflow_without_a_store_surfaces_artifact_error_on_the_job() {
    let tool = tool_over(constant(&"x".repeat(500))).with_result_policy(
        ResultPolicy::new()
            .with_max_chars(100)
            .with_overflow(crate::subagent::ResultOverflow::Artifact),
    );
    let payload = payload(&call_inline(&tool).await);
    assert!(payload["artifact_error"].as_str().unwrap().contains("no artifact store"));
}

#[tokio::test]
async fn a_leaf_is_refused_with_actionable_wording_and_catches_its_own_tool_name() {
    let mut harness = constant("x");
    harness.register_tool_dispatch(Arc::new(ProbeTool {
        token: Arc::default(),
        hang: false,
    }));
    let tool = tool_over(harness)
        .with_tool_name("probe")
        .with_role(SubagentRole::Leaf);
    let result = call_inline(&tool).await;
    assert!(result.is_error);
    assert!(result.output().contains("You are a leaf agent: do this work yourself"));
}
