//! Nested tool calls (C9): a tool calls another tool through
//! `ToolExecutionContext::call_tool`, and the call goes through the same
//! admission and execution path as a model-issued one.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::limits::RunLimits;
use crate::middleware::{MiddlewareToolOutcome, ToolHandler, ToolMiddleware};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{EventRecorder, text_response, tool_call_response};
use crate::tool::ToolExecutionContext;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolPolicy, ToolResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// What a nested call returned to the calling tool, flattened for assertions.
type Outcome = std::result::Result<ToolResult, String>;

/// A leaf tool that counts executions and answers `<name>-out`.
struct Leaf {
    name: &'static str,
    policy: ToolPolicy,
    ran: Arc<Mutex<Vec<Value>>>,
}

impl Leaf {
    fn new(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            policy: ToolPolicy::read_only(),
            ran: Arc::default(),
        })
    }

    fn approval_gated(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            policy: ToolPolicy::classified().requiring_approval(),
            ran: Arc::default(),
        })
    }

    fn runs(&self) -> usize {
        self.ran.lock().unwrap().len()
    }
}

#[async_trait]
impl Tool for Leaf {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "leaf"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"n": {"type": "integer"}}})
    }
    fn policy(&self) -> ToolPolicy {
        self.policy.clone()
    }
    async fn execute(&self, arguments: Value) -> anyhow::Result<ToolResult> {
        self.ran.lock().unwrap().push(arguments);
        Ok(ToolResult::success(format!("{}-out", self.name)))
    }
}

/// A tool that makes the scripted nested calls in order and records what each
/// returned. With `propagate` it forwards the first nested error with `?`.
struct Caller {
    name: &'static str,
    script: Vec<(&'static str, Value)>,
    outcomes: Arc<Mutex<Vec<Outcome>>>,
    cancel_first: bool,
    propagate: bool,
}

impl Caller {
    fn new(name: &'static str, script: Vec<(&'static str, Value)>) -> Self {
        Self {
            name,
            script,
            outcomes: Arc::default(),
            cancel_first: false,
            propagate: false,
        }
    }

    fn outcomes(&self) -> Arc<Mutex<Vec<Outcome>>> {
        Arc::clone(&self.outcomes)
    }
}

fn harness_extension(context: Option<&dyn tinytools::ToolRunContext>) -> ToolExecutionContext {
    context
        .and_then(tinytools::ToolRunContext::host_extension)
        .and_then(|any| any.downcast_ref::<ToolExecutionContext>())
        .expect("the harness always installs its context")
        .clone()
}

#[async_trait]
impl Tool for Caller {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "calls other tools"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn is_concurrency_safe(&self, _arguments: &Value) -> bool {
        true
    }
    async fn execute(&self, _arguments: Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn tinytools::ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let harness = harness_extension(context);
        if self.cancel_first {
            harness.cancellation.cancel();
        }
        for (name, args) in &self.script {
            let result = harness.call_tool(name, args.clone()).await;
            if self.propagate {
                self.outcomes
                    .lock()
                    .unwrap()
                    .push(result.as_ref().map(Clone::clone).map_err(|e| e.to_string()));
                result?;
            } else {
                self.outcomes
                    .lock()
                    .unwrap()
                    .push(result.map_err(|e| e.to_string()));
            }
        }
        Ok(ToolResult::success("caller-out"))
    }
}

/// A tool that calls itself until a nested call is refused, recording how deep
/// it got.
struct Relay {
    reached: Arc<Mutex<Vec<String>>>,
    refusal: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl Tool for Relay {
    fn name(&self) -> &str {
        "relay"
    }
    fn description(&self) -> &str {
        "recursive"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn tinytools::ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let harness = harness_extension(context);
        self.reached.lock().unwrap().push(harness.call_id.to_string());
        match harness.call_tool("relay", json!({})).await {
            Ok(_) => {}
            Err(error) => *self.refusal.lock().unwrap() = Some(error.to_string()),
        }
        Ok(ToolResult::success("relayed"))
    }
}

/// Refuses every call to `secret`, like a policy middleware denying a tool.
struct DenySecret;

#[async_trait]
impl ToolMiddleware<()> for DenySecret {
    fn name(&self) -> &str {
        "deny_secret"
    }
    async fn wrap_tool(
        &self,
        ctx: &RunContext<()>,
        state: &(),
        call: ToolCall,
        next: ToolHandler<'_, (), ()>,
    ) -> Result<MiddlewareToolOutcome> {
        if call.name == "secret" {
            return Ok(ToolResult::error("denied by policy").into());
        }
        next.run(ctx, state, call).await
    }
}

fn response(calls: Vec<ToolCall>) -> ModelResponse {
    let mut calls = calls.into_iter();
    let mut response = tool_call_response(calls.next().expect("at least one call"));
    response.message.tool_calls.extend(calls);
    response
}

fn parent_call(id: &str, tool: &str) -> ToolCall {
    ToolCall::new(id, tool, json!({}))
}

/// A harness whose model makes `calls` in one turn, then says "done".
fn harness_with(calls: Vec<ToolCall>, limits: RunLimits) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            response(calls),
            text_response("done"),
        ])),
    );
    harness.with_policy(RunPolicy {
        limits,
        ..RunPolicy::default()
    });
    harness
}

async fn run(
    harness: &AgentHarness<()>,
    recorder: &EventRecorder,
) -> Result<crate::middleware::AgentRun> {
    let ctx = RunContext::new(RunConfig::new("nested"), ()).with_events(recorder.sink());
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
}

fn started(recorder: &EventRecorder) -> Vec<(String, Option<String>)> {
    recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolStarted {
                call_id,
                parent_call_id,
                ..
            } => Some((
                call_id.to_string(),
                parent_call_id.as_ref().map(ToString::to_string),
            )),
            _ => None,
        })
        .collect()
}

fn nested_started(recorder: &EventRecorder) -> usize {
    started(recorder)
        .iter()
        .filter(|(_, parent)| parent.is_some())
        .count()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn call_tool_without_a_runner_is_a_clear_error() {
    let ctx = RunContext::new(RunConfig::new("outside-loop"), ());
    let context = ToolExecutionContext::from_run_context(&ctx, CallId::new("c1"));

    let error = context
        .call_tool("anything", json!({}))
        .await
        .expect_err("no runner outside the agent loop");

    let message = error.to_string();
    assert!(message.contains("anything"), "{message}");
    assert!(
        message.contains("only available while the agent loop"),
        "{message}"
    );
}

#[tokio::test]
async fn a_nested_call_runs_the_tool_and_returns_its_result() {
    let leaf = Leaf::new("leaf");
    let caller = Caller::new("caller", vec![("leaf", json!({"n": 1}))]);
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(leaf.clone());
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    let outcomes = outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].as_ref().unwrap().output(), "leaf-out");
    assert_eq!(leaf.ran.lock().unwrap().as_slice(), [json!({"n": 1})]);
}

#[tokio::test]
async fn a_nested_call_is_denied_by_policy_middleware() {
    let secret = Leaf::new("secret");
    let caller = Caller::new("caller", vec![("secret", json!({}))]);
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(secret.clone());
    harness.register_tool(Arc::new(caller));
    harness.push_tool_middleware(Arc::new(DenySecret));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    let outcomes = outcomes.lock().unwrap();
    let result = outcomes[0].as_ref().expect("a denial is a tool result");
    assert!(result.is_error);
    assert_eq!(result.output(), "denied by policy");
    assert_eq!(secret.runs(), 0, "the denied tool must not run");
}

#[tokio::test]
async fn a_nested_call_that_needs_approval_fails_without_deferring_the_parent() {
    let gated = Leaf::approval_gated("delete");
    let caller = Caller::new("caller", vec![("delete", json!({}))]);
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(gated.clone());
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    let run = run(&harness, &recorder).await.expect("run succeeds");

    let outcomes = outcomes.lock().unwrap();
    assert_eq!(
        outcomes[0].as_ref().expect_err("approval-needed call fails"),
        "permanent tool failure: nested call 'delete' requires approval; nested calls cannot be deferred"
    );
    assert_eq!(gated.runs(), 0);
    assert!(
        !recorder
            .events()
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolDeferred { .. })),
        "the parent must not be deferred"
    );
    assert!(run.deferred.is_none() || run.deferred.as_ref().is_some_and(|d| d.is_empty()));
}

#[tokio::test]
async fn nested_calls_count_against_max_tool_calls_and_trip_it() {
    let leaf = Leaf::new("leaf");
    let caller = Caller::new(
        "caller",
        vec![("leaf", json!({})), ("leaf", json!({})), ("leaf", json!({}))],
    );
    let outcomes = caller.outcomes();
    // One parent call + two nested calls fit; the third nested call trips.
    let mut harness = harness_with(
        vec![parent_call("p1", "caller")],
        RunLimits::default().with_max_tool_calls(3),
    );
    harness.register_tool(leaf.clone());
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    let outcomes = outcomes.lock().unwrap();
    assert!(outcomes[0].is_ok() && outcomes[1].is_ok());
    let tripped = outcomes[2].as_ref().expect_err("third nested call trips");
    assert!(tripped.contains("max tool calls (3)"), "{tripped}");
    assert_eq!(leaf.runs(), 2);
    assert!(recorder.events().iter().any(|event| matches!(
        event,
        AgentEvent::LimitReached {
            kind: crate::events::LimitKind::ToolCalls
        }
    )));
}

#[tokio::test]
async fn concurrent_parents_share_one_nested_budget() {
    let leaf = Leaf::new("leaf");
    let script = || vec![("leaf", json!({})), ("leaf", json!({}))];
    let a = Caller::new("alpha", script());
    let b = Caller::new("beta", script());
    let (outcomes_a, outcomes_b) = (a.outcomes(), b.outcomes());
    // Two parents + four nested wants 6 slots; the cap of 5 lets exactly three
    // of the four nested calls through, however the two parents interleave.
    let mut harness = harness_with(
        vec![parent_call("pa", "alpha"), parent_call("pb", "beta")],
        RunLimits::default().with_max_tool_calls(5),
    );
    harness.register_tool(leaf.clone());
    harness.register_tool(Arc::new(a));
    harness.register_tool(Arc::new(b));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    let admitted = outcomes_a
        .lock()
        .unwrap()
        .iter()
        .chain(outcomes_b.lock().unwrap().iter())
        .filter(|outcome| outcome.is_ok())
        .count();
    assert_eq!(admitted, 3, "total nested calls admitted across parents");
    assert_eq!(leaf.runs(), 3);
    assert_eq!(nested_started(&recorder), 3);
}

#[tokio::test]
async fn nesting_deeper_than_max_nested_depth_is_refused() {
    let reached = Arc::new(Mutex::new(Vec::new()));
    let refusal = Arc::new(Mutex::new(None));
    let mut harness = harness_with(
        vec![parent_call("p1", "relay")],
        RunLimits::default().with_max_nested_depth(2),
    );
    harness.register_tool(Arc::new(Relay {
        reached: Arc::clone(&reached),
        refusal: Arc::clone(&refusal),
    }));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    // Level 0 (model-issued), then levels 1 and 2; level 3 is refused.
    assert_eq!(reached.lock().unwrap().as_slice(), ["p1", "p1/1", "p1/1/1"]);
    let refusal = refusal.lock().unwrap().clone().expect("deepest call refused");
    assert!(refusal.contains("max_nested_depth"), "{refusal}");
}

#[tokio::test]
async fn default_depth_cap_allows_three_levels() {
    let reached = Arc::new(Mutex::new(Vec::new()));
    let mut harness = harness_with(vec![parent_call("p1", "relay")], RunLimits::default());
    harness.register_tool(Arc::new(Relay {
        reached: Arc::clone(&reached),
        refusal: Arc::default(),
    }));

    run(&harness, &EventRecorder::new()).await.expect("run succeeds");

    assert_eq!(reached.lock().unwrap().len(), 4, "levels 0 through 3");
}

#[tokio::test]
async fn nested_events_carry_the_parent_call_id() {
    let caller = Caller::new("caller", vec![("leaf", json!({})), ("leaf", json!({}))]);
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(Leaf::new("leaf"));
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    run(&harness, &recorder).await.expect("run succeeds");

    assert_eq!(
        started(&recorder),
        [
            ("p1".to_string(), None),
            ("p1/1".to_string(), Some("p1".to_string())),
            ("p1/2".to_string(), Some("p1".to_string())),
        ]
    );
    let completed: Vec<(String, Option<String>)> = recorder
        .events()
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCompleted {
                call_id,
                parent_call_id,
                ..
            } => Some((
                call_id.to_string(),
                parent_call_id.as_ref().map(ToString::to_string),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        completed,
        [
            ("p1/1".to_string(), Some("p1".to_string())),
            ("p1/2".to_string(), Some("p1".to_string())),
            ("p1".to_string(), None),
        ]
    );
}

#[tokio::test]
async fn nested_calls_are_summarised_in_parent_metadata_not_the_transcript() {
    let caller = Caller::new(
        "caller",
        vec![("leaf", json!({"n": 7})), ("missing", json!({}))],
    );
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(Leaf::new("leaf"));
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    let run = run(&harness, &recorder).await.expect("run succeeds");

    // Transcript: exactly one tool row, for the model-issued call.
    let tool_rows: Vec<String> = run
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some(t.tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_rows, ["p1"]);
    assert_eq!(run.tool_calls, 1, "nested calls are not model-issued calls");

    let metadata = run
        .tool_metadata
        .iter()
        .find(|entry| entry.call_id == CallId::new("p1"))
        .expect("parent metadata recorded")
        .metadata
        .clone();
    let nested = metadata["nested_calls"].as_array().expect("nested summary");
    assert_eq!(nested.len(), 2);
    assert_eq!(nested[0]["id"], "p1/1");
    assert_eq!(nested[0]["name"], "leaf");
    assert_eq!(nested[0]["status"], "ok");
    assert_eq!(nested[0]["args"], r#"{"n":7}"#);
    assert!(nested[0]["duration_ms"].is_u64());
    assert_eq!(nested[1]["id"], "p1/2");
    assert_eq!(nested[1]["status"], "failed");
    assert!(
        nested[1]["error"].as_str().unwrap().contains("missing"),
        "{nested:?}"
    );
}

#[tokio::test]
async fn the_nested_summary_is_capped_and_truncates_long_arguments() {
    let script: Vec<(&'static str, Value)> = (0..40)
        .map(|_| ("leaf", json!({"n": "x".repeat(4000)})))
        .collect();
    let caller = Caller::new("caller", script);
    let mut harness = harness_with(
        vec![parent_call("p1", "caller")],
        RunLimits::default().with_max_tool_calls(100),
    );
    harness.register_tool(Leaf::new("leaf"));
    harness.register_tool(Arc::new(caller));

    let run = run(&harness, &EventRecorder::new())
        .await
        .expect("run succeeds");

    let metadata = &run.tool_metadata[0].metadata;
    let nested = metadata["nested_calls"].as_array().unwrap();
    assert_eq!(nested.len(), 32);
    assert_eq!(metadata["nested_calls_truncated"], 8);
    assert!(nested[0]["args"].as_str().unwrap().len() <= 1024 + 3);
}

#[tokio::test]
async fn a_cancelled_run_refuses_nested_calls() {
    let leaf = Leaf::new("leaf");
    let mut caller = Caller::new("caller", vec![("leaf", json!({}))]);
    caller.cancel_first = true;
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(leaf.clone());
    harness.register_tool(Arc::new(caller));

    let recorder = EventRecorder::new();
    let _ = run(&harness, &recorder).await;

    let outcomes = outcomes.lock().unwrap();
    assert!(outcomes[0].as_ref().unwrap_err().contains("cancel"));
    assert_eq!(leaf.runs(), 0);
}

#[tokio::test]
async fn an_unknown_nested_tool_names_the_tool() {
    let caller = Caller::new("caller", vec![("ghost", json!({}))]);
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(Arc::new(caller));

    run(&harness, &EventRecorder::new())
        .await
        .expect("run succeeds");

    assert!(
        outcomes.lock().unwrap()[0]
            .as_ref()
            .unwrap_err()
            .contains("ghost")
    );
}

#[tokio::test]
async fn invalid_nested_arguments_are_refused_before_the_tool_runs() {
    let leaf = Leaf::new("leaf");
    let caller = Caller::new("caller", vec![("leaf", json!({"n": "not-an-integer"}))]);
    let outcomes = caller.outcomes();
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(leaf.clone());
    harness.register_tool(Arc::new(caller));

    run(&harness, &EventRecorder::new())
        .await
        .expect("run succeeds");

    assert!(
        outcomes.lock().unwrap()[0]
            .as_ref()
            .unwrap_err()
            .contains("invalid arguments")
    );
    assert_eq!(leaf.runs(), 0);
}

#[tokio::test]
async fn a_refusal_propagated_with_question_mark_reaches_the_model_as_a_tool_failure() {
    let gated = Leaf::approval_gated("delete");
    let mut caller = Caller::new("caller", vec![("delete", json!({}))]);
    caller.propagate = true;
    let mut harness = harness_with(vec![parent_call("p1", "caller")], RunLimits::default());
    harness.register_tool(gated);
    harness.register_tool(Arc::new(caller));

    let run = run(&harness, &EventRecorder::new())
        .await
        .expect("run succeeds");

    let text = run
        .messages
        .iter()
        .find_map(|m| match m {
            Message::Tool(t) if t.tool_call_id == "p1" => Some(m.text()),
            _ => None,
        })
        .expect("parent answered");
    assert!(text.contains("requires approval"), "{text}");
}
