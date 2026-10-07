//! Live model switch (`SteeringCommand::SwitchModel`) consumed at the model-call
//! boundary.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::runtime::AgentHarness;
use crate::steering::{SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringPolicy};
use crate::testkit::{EventRecorder, FakeTool};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;

/// Answers with fixed text and counts calls.
struct Counting {
    text: &'static str,
    calls: Mutex<usize>,
}

impl Counting {
    fn new(text: &'static str) -> Arc<Self> {
        Arc::new(Self {
            text,
            calls: Mutex::new(0),
        })
    }
    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

#[async_trait]
impl ChatModel<()> for Counting {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.calls.lock().unwrap() += 1;
        Ok(ModelResponse::assistant(self.text))
    }
}

/// First call: asks the orchestrator-side handle to switch to `target` and
/// requests a tool, so a second model call follows. Later calls answer.
struct SwitchesMidRun {
    handle: SteeringHandle,
    target: &'static str,
    calls: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for SwitchesMidRun {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        if *calls == 1 {
            self.handle.send(SteeringCommand::SwitchModel {
                model: self.target.into(),
            });
            return Ok(crate::testkit::tool_call_response(ToolCall::new(
                "c1",
                "lookup",
                json!({}),
            )));
        }
        Ok(ModelResponse::assistant("primary answered again"))
    }
}

fn allow_switch() -> SteeringHandle {
    SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::SwitchModel))
}

#[tokio::test]
async fn a_queued_switch_redirects_the_next_model_call() {
    let primary = Counting::new("primary");
    let backup = Counting::new("backup");
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    let handle = allow_switch();
    handle.send(SteeringCommand::SwitchModel {
        model: "backup".into(),
    });
    let ctx = RunContext::new(RunConfig::new("switch"), ()).with_steering(handle);

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("backup".to_string()));
    assert_eq!(primary.calls(), 0);
    assert_eq!(backup.calls(), 1);
}

#[tokio::test]
async fn a_switch_sent_mid_run_applies_to_the_following_calls() {
    let backup = Counting::new("backup answered");
    let handle = allow_switch();
    let primary = Arc::new(SwitchesMidRun {
        handle: handle.clone(),
        target: "backup",
        calls: Mutex::new(0),
    });
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    let ctx = RunContext::new(RunConfig::new("switch-mid"), ()).with_steering(handle);

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("backup answered".to_string()));
    assert_eq!(*primary.calls.lock().unwrap(), 1);
    assert_eq!(backup.calls(), 1);
}

#[tokio::test]
async fn an_unknown_model_is_rejected_without_failing_the_run() {
    let primary = Counting::new("primary");
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    let handle = allow_switch();
    handle.send(SteeringCommand::SwitchModel {
        model: "ghost".into(),
    });
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("switch-unknown"), ())
        .with_steering(handle.clone())
        .with_events(recorder.sink());

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("an unknown model must not crash the run");

    assert_eq!(run.text(), Some("primary".to_string()));
    assert_eq!(
        handle.model_override(),
        None,
        "a rejected switch is dropped"
    );
    let events = recorder.events();
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::Steered { command_kind, accepted: false } if command_kind == "switch_model"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ModelOverrideSkipped { requested, resolved }
            if requested == "ghost" && resolved == "primary"
    )));
}

#[tokio::test]
async fn a_disallowed_switch_leaves_the_model_alone() {
    let primary = Counting::new("primary");
    let backup = Counting::new("backup");
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    let handle = SteeringHandle::new(SteeringPolicy::new());
    handle.send(SteeringCommand::SwitchModel {
        model: "backup".into(),
    });
    let ctx = RunContext::new(RunConfig::new("switch-denied"), ()).with_steering(handle);

    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .unwrap();

    assert_eq!(run.text(), Some("primary".to_string()));
    assert_eq!(backup.calls(), 0);
}
