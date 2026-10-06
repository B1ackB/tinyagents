//! Explicit parent-call -> child-run link carried by sub-agent results.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::test::wait_for_terminal;
use super::{ChildDataPolicy, SubAgent, SubAgentTool};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::Result;
use tinyagents_harness::events::{AgentEvent, EventSink, RecordingListener};
use tinyagents_harness::ids::CallId;
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::tool::ToolDispatch;
use tinyinference_llm::providers::MockModel;

fn tool() -> SubAgentTool<(), ()> {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant("done")));
    SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
}

fn json_of(result: &tinytools::ToolResult) -> Value {
    serde_json::from_str(&result.output()).expect("tool result is JSON")
}

#[tokio::test]
async fn queued_result_and_completed_job_carry_the_explicit_link() {
    let tool = tool();
    let jobs = tool.job_registry().clone();
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let parent = RunContext::new(RunConfig::new("parent"), ()).with_events(events);

    let result = ToolDispatch::<(), ()>::execute(
        &tool,
        &(),
        CallId::new("call-42"),
        json!({"input": "work"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let queued = json_of(&result);
    let job_id = queued["job_id"].as_str().unwrap().to_owned();
    let run_id = queued["subagent_run_id"]
        .as_str()
        .expect("run id in queued result");
    assert!(run_id.starts_with("worker-d1-parent-"), "run id: {run_id}");
    assert_eq!(queued["tool_call_id"], "call-42");

    let job = wait_for_terminal(&jobs, &job_id, parent.instance_id()).await;
    assert_eq!(job.subagent_run_id.as_deref(), Some(run_id));
    assert_eq!(job.tool_call_id.as_deref(), Some("call-42"));
    let completed = serde_json::to_value(&job).unwrap();
    assert_eq!(completed["subagent_run_id"], run_id);
    assert_eq!(completed["tool_call_id"], "call-42");

    // The child run itself is stamped with the same ids.
    let started = recorder
        .events()
        .iter()
        .any(|record| matches!(&record.event, AgentEvent::RunStarted { run_id: id, .. } if id.as_str() == run_id));
    assert!(
        started,
        "child run id matches the advertised subagent_run_id"
    );
}

struct MetadataProbe(Arc<Mutex<Option<Value>>>);

#[async_trait::async_trait]
impl Middleware<(), ()> for MetadataProbe {
    fn name(&self) -> &str {
        "metadata-probe"
    }

    async fn before_agent(&self, ctx: &mut RunContext<()>, _state: &()) -> Result<()> {
        *self.0.lock().unwrap() = Some(ctx.config.metadata.clone());
        Ok(())
    }
}

#[tokio::test]
async fn child_run_metadata_names_the_parent_call_and_job() {
    let seen = Arc::new(Mutex::new(None));
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant("done")));
    harness.push_middleware(Arc::new(MetadataProbe(seen.clone())));
    let tool = SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    );
    let jobs = tool.job_registry().clone();
    let parent = RunContext::new(
        RunConfig::new("parent").with_metadata(json!({"keep": 1})),
        (),
    );
    let result = ToolDispatch::<(), ()>::execute(
        &tool,
        &(),
        CallId::new("call-7"),
        json!({"input": "work"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let queued = json_of(&result);
    wait_for_terminal(
        &jobs,
        queued["job_id"].as_str().unwrap(),
        parent.instance_id(),
    )
    .await;

    let metadata = seen.lock().unwrap().clone().expect("child ran");
    assert_eq!(metadata["keep"], 1, "parent metadata is preserved");
    assert_eq!(metadata["parent_tool_call_id"], "call-7");
    assert_eq!(metadata["subagent_job_id"], queued["job_id"]);
    assert_eq!(metadata["subagent_run_id"], queued["subagent_run_id"]);
}

#[tokio::test]
async fn tool_call_id_is_omitted_when_the_caller_has_none() {
    let tool = tool();
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let result = tool
        .invoke_in_parent_context(
            &(),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .unwrap();
    let queued = json_of(&result);
    assert!(queued.get("tool_call_id").is_none());
    assert!(queued.get("subagent_run_id").is_some());
}
