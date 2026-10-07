//! Spawn-admission contracts for [`SubAgentTool`]: caps are enforced for both
//! modes, concurrent spawns cannot race past a cap, and slots are returned
//! when a child finishes or its spawn fails.

use super::*;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::jobs_tests::PanickingModel;
use super::test::BlockedModel;
use crate::subagent::{SpawnAdmission, SpawnPolicy};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::providers::MockModel;

type Gate = Arc<tokio::sync::Semaphore>;

fn policy(parent: Option<usize>, root: Option<usize>, targets: Option<&[&str]>) -> SpawnPolicy {
    SpawnPolicy {
        max_children_per_parent: parent,
        max_total_per_root: root,
        allowed_targets: targets.map(|t| t.iter().map(|s| (*s).to_owned()).collect()),
    }
}

fn blocked_tool(policy: SpawnPolicy) -> (Arc<SubAgentTool<(), ()>>, Gate, Gate) {
    let started: Gate = Arc::new(tokio::sync::Semaphore::new(0));
    let release: Gate = Arc::new(tokio::sync::Semaphore::new(0));
    let mut harness = AgentHarness::new();
    harness.register_model(
        "worker",
        Arc::new(BlockedModel {
            started: started.clone(),
            release: release.clone(),
        }),
    );
    let tool = SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
    .with_spawn_admission(SpawnAdmission::new(policy));
    (Arc::new(tool), started, release)
}

fn constant_tool(policy: SpawnPolicy) -> SubAgentTool<(), ()> {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant("done")));
    SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
    .with_spawn_admission(SpawnAdmission::new(policy))
}

async fn call(
    tool: &SubAgentTool<(), ()>,
    parent: &RunContext<()>,
    args: Value,
) -> tinytools::ToolResult {
    tool.invoke_in_parent_context(&(), args, tinytools::ToolCallOptions::default(), parent)
        .await
        .expect("tool call returns a result")
}

fn parent() -> RunContext<()> {
    RunContext::new(RunConfig::new("parent"), ())
}

fn assert_limit_signal(result: &tinytools::ToolResult) {
    assert!(result.is_error, "{}", result.output());
    assert!(
        result
            .output()
            .contains("limit signal, not a completed answer"),
        "{}",
        result.output()
    );
}

async fn slots_settle_to(tool: &SubAgentTool<(), ()>, parent: &str, expected: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while tool.spawn_admission().active_children(parent) != expected {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("slots settle");
}

#[tokio::test]
async fn background_spawn_over_the_parent_cap_is_a_limit_signal() {
    let (tool, started, release) = blocked_tool(policy(Some(1), None, None));
    let parent = parent();

    let first = call(&tool, &parent, json!({"input": "a"})).await;
    assert!(!first.is_error);
    let second = call(&tool, &parent, json!({"input": "b"})).await;

    assert_limit_signal(&second);
    assert!(second.output().contains("1/1"), "{}", second.output());
    assert_eq!(tool.job_registry().list().len(), 1, "no job was created");

    // The slot returns when the child reaches a terminal state.
    let _started = started.acquire().await.unwrap();
    release.add_permits(1);
    slots_settle_to(&tool, "parent", 0).await;
    let third = call(&tool, &parent, json!({"input": "c"})).await;
    assert!(!third.is_error, "{}", third.output());
    release.add_permits(1);
}

#[tokio::test]
async fn inline_spawn_over_the_parent_cap_is_a_limit_signal() {
    let (tool, _started, release) = blocked_tool(policy(Some(1), None, None));
    let parent = parent();
    let held = call(&tool, &parent, json!({"input": "a"})).await;
    assert!(!held.is_error);

    let inline = call(&tool, &parent, json!({"input": "b", "mode": "inline"})).await;

    assert_limit_signal(&inline);
    assert_eq!(tool.job_registry().list().len(), 1);
    release.add_permits(1);
}

#[tokio::test]
async fn inline_slot_is_released_when_the_call_returns() {
    let tool = constant_tool(policy(Some(1), None, None));
    let parent = parent();
    for _ in 0..3 {
        let result = call(&tool, &parent, json!({"input": "x", "mode": "inline"})).await;
        assert!(!result.is_error, "{}", result.output());
    }
    assert_eq!(tool.spawn_admission().active_children("run:parent"), 0);
}

#[tokio::test]
async fn total_budget_per_root_is_spent_even_by_finished_children() {
    let tool = constant_tool(policy(None, Some(2), None));
    let parent = parent();
    for _ in 0..2 {
        let result = call(&tool, &parent, json!({"input": "x", "mode": "inline"})).await;
        assert!(!result.is_error);
    }
    let over = call(&tool, &parent, json!({"input": "x", "mode": "inline"})).await;
    assert_limit_signal(&over);
    assert!(over.output().contains("2/2"), "{}", over.output());
}

#[tokio::test]
async fn disallowed_target_is_refused_before_spawning() {
    let tool = constant_tool(policy(None, None, Some(&["researcher"])));
    let result = call(&tool, &parent(), json!({"input": "x"})).await;
    assert_limit_signal(&result);
    assert!(result.output().contains("`worker`"), "{}", result.output());
    assert!(tool.job_registry().list().is_empty());
}

#[tokio::test]
async fn allowed_target_spawns() {
    let tool = constant_tool(policy(None, None, Some(&["worker"])));
    let result = call(&tool, &parent(), json!({"input": "x", "mode": "inline"})).await;
    assert!(!result.is_error, "{}", result.output());
}

#[tokio::test]
async fn failed_spawn_refunds_the_reservation() {
    let tool = constant_tool(policy(Some(1), Some(1), None));
    // Depth limit hit after admission: nothing spawns, nothing stays reserved.
    let too_deep = RunContext::new(RunConfig::new("parent").with_max_depth(0), ());
    let result = call(&tool, &too_deep, json!({"input": "x"})).await;
    assert_limit_signal(&result);
    assert_eq!(tool.spawn_admission().active_children("run:parent"), 0);
    assert_eq!(tool.spawn_admission().spawned_in_scope("run:parent"), 0);

    let ok = call(&tool, &parent(), json!({"input": "x", "mode": "inline"})).await;
    assert!(!ok.is_error, "{}", ok.output());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_background_spawns_cannot_exceed_the_cap() {
    let (tool, _started, release) = blocked_tool(policy(Some(3), None, None));
    let parent = Arc::new(parent());
    let tasks: Vec<_> = (0..12)
        .map(|_| {
            let tool = tool.clone();
            let parent = parent.clone();
            tokio::spawn(async move { call(&tool, &parent, json!({"input": "x"})).await })
        })
        .collect();
    let mut admitted = 0;
    for task in tasks {
        if !task.await.unwrap().is_error {
            admitted += 1;
        }
    }
    assert_eq!(admitted, 3);
    assert_eq!(tool.job_registry().list().len(), 3);
    release.add_permits(3);
}

#[tokio::test]
async fn shared_admission_counts_across_tools() {
    let shared = SpawnAdmission::new(policy(Some(1), None, None));
    let (tool_a, _sa, release_a) = blocked_tool(SpawnPolicy::default());
    let (tool_b, _sb, release_b) = blocked_tool(SpawnPolicy::default());
    let tool_a = Arc::new(
        Arc::into_inner(tool_a)
            .unwrap()
            .with_spawn_admission(shared.clone()),
    );
    let tool_b = Arc::new(
        Arc::into_inner(tool_b)
            .unwrap()
            .with_spawn_admission(shared),
    );
    let parent = parent();

    assert!(!call(&tool_a, &parent, json!({"input": "a"})).await.is_error);
    assert_limit_signal(&call(&tool_b, &parent, json!({"input": "b"})).await);
    release_a.add_permits(1);
    release_b.add_permits(1);
}

fn parent_on_thread(run_id: &str, thread: &str) -> RunContext<()> {
    RunContext::new(RunConfig::new(run_id).with_thread(thread), ())
}

#[tokio::test]
async fn a_later_turn_shares_the_cap_with_a_background_child_still_alive() {
    let (tool, started, release) = blocked_tool(policy(Some(1), None, None));

    // Turn 1 spawns a background child and ends; turn 2 has a fresh run id on
    // the same conversation thread.
    let turn_one = parent_on_thread("run-turn-1", "thread-A");
    assert!(!call(&tool, &turn_one, json!({"input": "a"})).await.is_error);
    let _started = started.acquire().await.unwrap();
    drop(turn_one);

    let turn_two = parent_on_thread("run-turn-2", "thread-A");
    let blocked = call(&tool, &turn_two, json!({"input": "b"})).await;
    assert_limit_signal(&blocked);
    assert_eq!(tool.spawn_admission().active_children("thread:thread-A"), 1);

    // A different conversation is unaffected.
    let other = parent_on_thread("run-turn-3", "thread-B");
    assert!(!call(&tool, &other, json!({"input": "c"})).await.is_error);

    release.add_permits(2);
}

#[tokio::test]
async fn dropping_an_inline_call_releases_its_slot() {
    let (tool, started, _release) = blocked_tool(policy(Some(1), Some(1), None));
    let parent = parent();

    let abandoned = tokio::time::timeout(
        Duration::from_millis(50),
        call(&tool, &parent, json!({"input": "a", "mode": "inline"})),
    )
    .await;
    assert!(abandoned.is_err(), "the blocked inline call timed out");
    let _started = started.acquire().await.unwrap();

    // The dropped future released the live slot, and refunded nothing it
    // already spent: the child had started, so the budget stays consumed.
    assert_eq!(tool.spawn_admission().active_children("run:parent"), 0);
    assert_eq!(tool.spawn_admission().spawned_in_scope("run:parent"), 1);
}

#[tokio::test]
async fn a_panicking_background_child_releases_its_slot() {
    let mut harness = AgentHarness::new();
    harness.register_model("worker", Arc::new(PanickingModel));
    let tool = SubAgentTool::new(
        Arc::new(SubAgent::new("worker", "works", Arc::new(harness))),
        ChildDataPolicy::new(|_: &()| ()),
    )
    .with_spawn_admission(SpawnAdmission::new(policy(Some(1), None, None)));
    let parent = parent();

    assert!(!call(&tool, &parent, json!({"input": "a"})).await.is_error);
    slots_settle_to(&tool, "parent", 0).await;

    let again = call(&tool, &parent, json!({"input": "b"})).await;
    assert!(!again.is_error, "{}", again.output());
}
