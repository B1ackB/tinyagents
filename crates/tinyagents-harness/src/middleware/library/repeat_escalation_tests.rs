//! Tests for the staged (warn, block, halt) repeat escalation, the pattern
//! warnings and the post-compaction guard as the middleware drives them.

use std::sync::Arc;

use serde_json::json;

use super::wrap_up::DEFAULT_CLEARED_PLACEHOLDER;
use super::*;
use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::middleware::{Middleware, ToolInvocationIdentity};
use crate::no_progress::RepeatProgressConfig;
use crate::steering::{SteeringCommand, SteeringHandle};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message as TaMessage};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall as TaToolCall;
use tinytools::ToolResult as TaToolResult;

fn ctx() -> RunContext {
    let mut ctx = RunContext::new(RunConfig::new("mw-test"), ());
    ctx.instance_id = 1;
    ctx
}

fn mw(handle: &SteeringHandle, summary: &HaltSummarySlot) -> RepeatProgressMiddleware {
    RepeatProgressMiddleware::new(
        handle.clone(),
        summary.clone(),
        Arc::new(|tool| tool == "wait_subagent"),
    )
}

fn pauses(handle: &SteeringHandle) -> usize {
    handle
        .drain()
        .into_iter()
        .filter(|c| matches!(c, SteeringCommand::Pause))
        .count()
}

fn response(tool: &str, args: serde_json::Value, narration: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant(narration);
    response.message = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text(narration.to_string())],
        tool_calls: vec![TaToolCall::new("repeat-1", tool, args)],
        usage: None,
        origin: None,
    };
    response.finish_reason = Some("tool_calls".to_string());
    response
}

/// One model turn calling `tool`, driven the way the loop does: admission
/// (`before_tool`; a refusal answers the call without running it), then
/// `after_tool`. Returns the text the model would read.
async fn turn(
    mw: &RepeatProgressMiddleware,
    tool: &str,
    args: serde_json::Value,
    narration: &str,
    output: &str,
) -> String {
    let mut resp = response(tool, args.clone(), narration);
    mw.after_model(&mut ctx(), &(), &mut resp).await.unwrap();
    let mut call = TaToolCall::new("repeat-1", tool, args);
    let mut result = match mw.before_tool(&mut ctx(), &(), &mut call).await {
        Ok(()) => TaToolResult::success(output),
        Err(TinyAgentsError::ToolFailed(message)) => TaToolResult::failed(message),
        Err(other) => panic!("unexpected admission error: {other}"),
    };
    let invocation = ToolInvocationIdentity::new("repeat-1", tool);
    mw.after_tool(&mut ctx(), &(), &invocation, &mut result)
        .await
        .unwrap();
    result.output()
}

/// The same call every time. The narration varies so that only the call
/// recurrence (not the identical-output streak) is in play.
async fn same_turn(mw: &RepeatProgressMiddleware, output: &str) -> String {
    static TURN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let narration = format!(
        "working {}",
        TURN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    turn(mw, "lookup", json!({"id": 1}), &narration, output).await
}

#[tokio::test]
async fn an_identical_call_is_warned_then_blocked_then_halts() {
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = mw(&handle, &summary);

    for n in 1..=2 {
        let text = same_turn(&mw, "ok").await;
        assert_eq!(text, "ok", "attempt {n} is quiet");
    }
    let warned = same_turn(&mw, "ok").await;
    assert!(
        warned.starts_with("ok") && warned.contains("[repeat notice]"),
        "the third identical result carries a warning: {warned}"
    );
    assert_eq!(pauses(&handle), 0, "a warning does not stop the run");

    assert_eq!(
        same_turn(&mw, "ok").await,
        "ok",
        "the warning is shown once per signature"
    );

    let blocked = same_turn(&mw, "ok").await;
    assert!(
        blocked.contains("not executed"),
        "the fifth identical call is answered without running: {blocked}"
    );
    assert_eq!(pauses(&handle), 0, "the first block does not halt");

    let halted = same_turn(&mw, "ok").await;
    assert!(halted.contains("blocked"), "{halted}");
    assert_eq!(pauses(&handle), 1, "the second block halts the run");
    assert!(
        summary
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|text| text.starts_with("Stopping:")),
        "the halt summary reaches the host"
    );
}

#[tokio::test]
async fn immediate_halt_config_halts_at_the_first_threshold_with_no_note() {
    let handle = SteeringHandle::allow_all();
    let summary = Arc::new(std::sync::Mutex::new(None));
    let mw = mw(&handle, &summary).with_config(RepeatProgressConfig::immediate_halt());
    for _ in 0..3 {
        assert_eq!(same_turn(&mw, "ok").await, "ok");
    }
    assert_eq!(pauses(&handle), 1);
}

#[tokio::test]
async fn a_changed_result_is_never_blocked() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    for i in 0..4 {
        let text = same_turn(&mw, &format!("result-{i}")).await;
        assert!(!text.contains("not executed"), "{text}");
    }
    assert_eq!(pauses(&handle), 0);
}

#[tokio::test]
async fn thresholds_come_from_the_config() {
    let handle = SteeringHandle::allow_all();
    let mw =
        mw(&handle, &Arc::new(std::sync::Mutex::new(None))).with_config(RepeatProgressConfig {
            call_threshold: 2,
            ..RepeatProgressConfig::default()
        });
    same_turn(&mw, "ok").await;
    assert!(same_turn(&mw, "ok").await.contains("[repeat notice]"));
}

#[tokio::test]
async fn alternating_calls_get_a_ping_pong_warning() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    let mut texts = Vec::new();
    // Distinct narration keeps the output streak out of it.
    for i in 0..3 {
        texts.push(turn(&mw, "read", json!({"p": "a"}), &format!("a{i}"), "doc").await);
        texts.push(turn(&mw, "search", json!({"q": "b"}), &format!("b{i}"), "hits").await);
    }
    assert!(
        texts.iter().any(|text| text.contains("alternating")),
        "{texts:?}"
    );
    assert_eq!(pauses(&handle), 0, "ping-pong only warns");
}

#[tokio::test]
async fn argument_churn_gets_a_warning() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    let mut texts = Vec::new();
    for variant in 0..3 {
        for n in 0..3 {
            texts.push(
                turn(
                    &mw,
                    "search",
                    json!({"q": variant}),
                    &format!("v{variant}n{n}"),
                    "no results",
                )
                .await,
            );
        }
    }
    assert!(
        texts
            .iter()
            .any(|text| text.contains("different sets of arguments")),
        "{texts:?}"
    );
    assert_eq!(pauses(&handle), 0, "churn only warns");
}

#[tokio::test]
async fn exempt_polling_calls_are_never_blocked() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    for _ in 0..10 {
        let text = turn(&mw, "wait_subagent", json!({"t": 1}), "waiting", "running").await;
        assert_eq!(text, "running");
    }
    assert_eq!(pauses(&handle), 0);
}

/// The next model call after the recorded result `repeat-1` left the context.
async fn compact(mw: &RepeatProgressMiddleware) {
    let observer = mw.eviction_observer();
    let mut request = ModelRequest::new(vec![TaMessage::tool("repeat-1", "ok")]);
    mw.before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
    let mut request = ModelRequest::new(vec![TaMessage::tool(
        "repeat-1",
        DEFAULT_CLEARED_PLACEHOLDER,
    )]);
    observer
        .before_model(&mut ctx(), &(), &mut request)
        .await
        .unwrap();
}

#[tokio::test]
async fn repeating_the_pre_compaction_tail_blocks_after_one_repeat() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    assert_eq!(same_turn(&mw, "ok").await, "ok");
    compact(&mw).await;

    let after = same_turn(&mw, "ok").await;
    assert!(after.contains("compacted"), "{after}");
    let blocked = same_turn(&mw, "ok").await;
    assert!(
        blocked.contains("not executed"),
        "the repeat after compaction skips the warn-block gap: {blocked}"
    );
}

#[tokio::test]
async fn a_different_call_after_compaction_is_left_alone() {
    let handle = SteeringHandle::allow_all();
    let mw = mw(&handle, &Arc::new(std::sync::Mutex::new(None)));
    same_turn(&mw, "ok").await;
    compact(&mw).await;
    for _ in 0..2 {
        let text = turn(&mw, "lookup", json!({"id": 2}), "next", "fresh").await;
        assert_eq!(text, "fresh");
    }
}

#[tokio::test]
async fn compaction_does_not_forgive_an_earlier_block() {
    let handle = SteeringHandle::allow_all();
    // The guard would escalate sooner; isolate the run-wide block count.
    let mw =
        mw(&handle, &Arc::new(std::sync::Mutex::new(None))).with_config(RepeatProgressConfig {
            post_compaction_window: 0,
            ..RepeatProgressConfig::default()
        });
    for _ in 0..4 {
        same_turn(&mw, "ok").await;
    }
    assert!(same_turn(&mw, "ok").await.contains("not executed"));
    compact(&mw).await;
    for _ in 0..4 {
        same_turn(&mw, "ok").await;
    }
    same_turn(&mw, "ok").await;
    assert_eq!(
        pauses(&handle),
        1,
        "the second block halts even after compaction"
    );
}
