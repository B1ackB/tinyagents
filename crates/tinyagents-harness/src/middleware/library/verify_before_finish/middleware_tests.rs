//! Tests for [`VerifyBeforeFinishMiddleware`].

use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::limits::RunLimits;
use crate::middleware::{AgentRun, Middleware};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{FakeTool, ScriptedModel};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;

const CHECK: &str = "CHECK: re-read the request";

#[test]
fn retention_prunes_dropped_contexts_without_evicting_active_runs() {
    let mut runs = HashMap::new();
    let first = RunContext::new(RunConfig::new("first"), ());
    state_for(&mut runs, &first).fired = true;
    let others: Vec<_> = (0..MAX_RETAINED_RUNS)
        .map(|_| RunContext::new(RunConfig::new("other"), ()))
        .collect();
    for ctx in &others {
        state_for(&mut runs, ctx);
    }
    assert!(state_for(&mut runs, &first).fired);
    assert_eq!(runs.len(), MAX_RETAINED_RUNS + 1);

    drop(others);
    let next = RunContext::new(RunConfig::new("next"), ());
    state_for(&mut runs, &next);
    assert_eq!(runs.len(), 2);
    assert!(state_for(&mut runs, &first).fired);
}

/// An assistant turn that requests one call of `name`.
fn tool_round(id: &str, name: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant(String::new());
    response.message.content = Vec::new();
    response.message.id = Some(format!("msg-{id}"));
    response.message.tool_calls = vec![ToolCall::new(id, name, serde_json::json!({}))];
    response.finish_reason = Some("tool_calls".to_string());
    response
}

fn answer(text: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant(text.to_string());
    response.finish_reason = Some("stop".to_string());
    response
}

/// Drive one run with `responses` scripted, the middleware installed and
/// `limits` applied. Returns the run and every request the model received.
async fn drive(
    responses: Vec<ModelResponse>,
    mw: VerifyBeforeFinishMiddleware,
    limits: RunLimits,
) -> (AgentRun, Vec<tinyinference_llm::model::ModelRequest>) {
    let model = Arc::new(ScriptedModel::new(responses));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone());
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "found")));
    harness.register_tool(Arc::new(FakeTool::returning("todo", "listed")));
    harness.push_middleware(Arc::new(mw));
    harness.with_policy(RunPolicy {
        limits,
        ..RunPolicy::default()
    });
    let run = harness
        .invoke_default(&(), vec![Message::user("do the task")])
        .await
        .expect("run succeeds");
    (run, model.requests())
}

fn check_count(run: &AgentRun) -> usize {
    run.messages
        .iter()
        .filter(|m| matches!(m, Message::User(_)) && m.text() == CHECK)
        .count()
}

/// A tool-using turn that answers is asked to check once, and the answer
/// after the check is the turn's answer.
#[tokio::test]
async fn asks_for_the_check_before_accepting_the_first_answer() {
    let (run, requests) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("checked"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK),
        RunLimits::default().with_max_model_calls(10),
    )
    .await;

    assert_eq!(run.text(), Some("checked".to_string()));
    assert_eq!(run.model_calls, 3);
    assert_eq!(check_count(&run), 1);
    let last = requests.last().expect("three requests");
    assert_eq!(
        last.messages.last().map(Message::text),
        Some(CHECK.to_string()),
        "the check is the tail of the request, as a user turn"
    );
    assert!(
        matches!(last.messages.last(), Some(Message::User(_))),
        "the check must be a user turn, not a mid-conversation system message"
    );
}

/// The check fires at most once per run: the answer that follows it ends the
/// turn even though it, too, is a tool-less answer.
#[tokio::test]
async fn fires_at_most_once_per_run() {
    let (run, _) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            tool_round("c2", "lookup"),
            answer("fixed"),
            answer("never reached"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK),
        RunLimits::default().with_max_model_calls(10),
    )
    .await;

    assert_eq!(run.text(), Some("fixed".to_string()));
    assert_eq!(run.model_calls, 4);
    assert_eq!(check_count(&run), 1);
}

/// With two or fewer model calls left the check would collide with the
/// final-call wrap-up, so the first answer stands.
#[tokio::test]
async fn skips_when_few_model_calls_remain() {
    // Call 1 is the tool round, call 2 the answer: 4 - 2 = 2 remain.
    let (run, _) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("never reached"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK),
        RunLimits::default().with_max_model_calls(4),
    )
    .await;

    assert_eq!(run.text(), Some("draft".to_string()));
    assert_eq!(check_count(&run), 0);
}

/// Three remaining calls are enough room for the check.
#[tokio::test]
async fn fires_with_three_model_calls_remaining() {
    let (run, _) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("checked"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK),
        RunLimits::default().with_max_model_calls(5),
    )
    .await;

    assert_eq!(run.text(), Some("checked".to_string()));
    assert_eq!(check_count(&run), 1);
}

/// A run close to the policy wall-clock cap the host declared keeps its first
/// answer. The policy cap is not on the run context, so the host passes it in.
#[tokio::test]
async fn skips_when_the_policy_wall_clock_is_short() {
    let (run, _) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("never reached"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK)
            .with_wall_clock_limit(Duration::from_secs(60))
            .with_min_remaining_wall_clock(Duration::from_secs(600)),
        RunLimits::default()
            .with_max_model_calls(10)
            .with_max_wall_clock_ms(Some(60_000)),
    )
    .await;

    assert_eq!(run.text(), Some("draft".to_string()));
    assert_eq!(check_count(&run), 0);
}

/// The run config's own deadline (`RunConfig::timeout_ms`) is honoured too.
#[tokio::test]
async fn skips_when_the_run_deadline_is_short() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK)
        .with_min_remaining_wall_clock(Duration::from_secs(600));
    let mut ctx = RunContext::new(
        RunConfig::new("vbf")
            .with_max_model_calls(10)
            .with_timeout_ms(60_000),
        (),
    );
    ctx.limits.record_model_call().unwrap();
    mw.after_model(&mut ctx, &(), &mut tool_round("c1", "lookup"))
        .await
        .unwrap();
    ctx.limits.record_model_call().unwrap();
    let mut draft = answer("draft");
    mw.after_model(&mut ctx, &(), &mut draft).await.unwrap();
    assert!(draft.continue_turn.is_none());
}

/// A deadline with room to spare does not block the check.
#[tokio::test]
async fn fires_when_the_wall_clock_has_room() {
    let (run, _) = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("checked"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK)
            .with_wall_clock_limit(Duration::from_secs(600))
            .with_min_remaining_wall_clock(Duration::from_secs(1)),
        RunLimits::default()
            .with_max_model_calls(10)
            .with_max_wall_clock_ms(Some(600_000)),
    )
    .await;

    assert_eq!(check_count(&run), 1);
}

/// A plain chat answer (no tool rounds) is not a multi-step task.
#[tokio::test]
async fn default_trigger_needs_a_tool_round() {
    let (run, _) = drive(
        vec![answer("hello"), answer("never reached")],
        VerifyBeforeFinishMiddleware::new(CHECK),
        RunLimits::default().with_max_model_calls(10),
    )
    .await;

    assert_eq!(run.text(), Some("hello".to_string()));
    assert_eq!(check_count(&run), 0);
}

/// `with_min_tool_rounds` raises the bar.
#[tokio::test]
async fn min_tool_rounds_gates_the_check() {
    let short = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("never reached"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK).with_min_tool_rounds(2),
        RunLimits::default().with_max_model_calls(10),
    )
    .await
    .0;
    assert_eq!(check_count(&short), 0);

    let long = drive(
        vec![
            tool_round("c1", "lookup"),
            tool_round("c2", "lookup"),
            answer("draft"),
            answer("checked"),
        ],
        VerifyBeforeFinishMiddleware::new(CHECK).with_min_tool_rounds(2),
        RunLimits::default().with_max_model_calls(10),
    )
    .await
    .0;
    assert_eq!(check_count(&long), 1);
    assert_eq!(long.text(), Some("checked".to_string()));
}

/// A custom trigger sees which tools the run called.
#[tokio::test]
async fn custom_trigger_sees_the_tools_called() {
    let only_with_todo =
        || VerifyBeforeFinishMiddleware::new(CHECK).with_trigger(|a| a.called("todo"));

    let without = drive(
        vec![
            tool_round("c1", "lookup"),
            answer("draft"),
            answer("never reached"),
        ],
        only_with_todo(),
        RunLimits::default().with_max_model_calls(10),
    )
    .await
    .0;
    assert_eq!(check_count(&without), 0);

    let with = drive(
        vec![tool_round("c1", "todo"), answer("draft"), answer("checked")],
        only_with_todo(),
        RunLimits::default().with_max_model_calls(10),
    )
    .await
    .0;
    assert_eq!(check_count(&with), 1);
}

/// An answer that still carries tool calls is not a final answer: the check
/// is not requested, and the round is counted instead.
#[tokio::test]
async fn does_not_fire_on_a_response_with_tool_calls() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK);
    let mut ctx = RunContext::new(RunConfig::new("vbf").with_max_model_calls(10), ());
    ctx.limits.record_model_call().unwrap();
    let mut first = tool_round("c1", "lookup");
    mw.after_model(&mut ctx, &(), &mut first).await.unwrap();
    assert!(first.continue_turn.is_none());

    // The same response again, now with a round already behind it.
    ctx.limits.record_model_call().unwrap();
    let mut second = tool_round("c2", "lookup");
    second.message.content = vec![tinyinference_llm::message::ContentBlock::Text(
        "and one more lookup".to_string(),
    )];
    mw.after_model(&mut ctx, &(), &mut second).await.unwrap();
    assert!(
        second.continue_turn.is_none(),
        "text alongside tool calls is a progress note, not a final answer"
    );
}

/// An empty or truncated answer belongs to the loop's own retries, and an
/// answer some other layer already continued is left alone.
#[tokio::test]
async fn leaves_empty_truncated_and_continued_answers_alone() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK);
    let mut ctx = RunContext::new(RunConfig::new("vbf").with_max_model_calls(10), ());
    ctx.limits.record_model_call().unwrap();
    mw.after_model(&mut ctx, &(), &mut tool_round("c1", "lookup"))
        .await
        .unwrap();
    ctx.limits.record_model_call().unwrap();

    let mut empty = answer("   ");
    mw.after_model(&mut ctx, &(), &mut empty).await.unwrap();
    assert!(empty.continue_turn.is_none());

    let mut truncated = answer("half an ans");
    truncated.finish_reason = Some("length".to_string());
    mw.after_model(&mut ctx, &(), &mut truncated).await.unwrap();
    assert!(truncated.continue_turn.is_none());

    let mut continued = answer("let me look");
    continued.continue_turn = Some("(next)".to_string());
    mw.after_model(&mut ctx, &(), &mut continued).await.unwrap();
    assert_eq!(continued.continue_turn.as_deref(), Some("(next)"));

    // None of those spent the one check.
    let mut real = answer("done");
    mw.after_model(&mut ctx, &(), &mut real).await.unwrap();
    assert_eq!(real.continue_turn.as_deref(), Some(CHECK));
}

#[tokio::test]
async fn failed_run_releases_its_activity() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK);
    let mut ctx = RunContext::new(RunConfig::new("vbf"), ());
    mw.after_model(&mut ctx, &(), &mut tool_round("c1", "lookup"))
        .await
        .unwrap();
    assert!(mw.runs.lock().unwrap().contains_key(&ctx.instance_id()));
    <VerifyBeforeFinishMiddleware as Middleware<(), ()>>::on_error(
        &mw,
        &mut ctx,
        &crate::error::TinyAgentsError::Model("failed".into()),
    )
    .await
    .unwrap();
    assert!(!mw.runs.lock().unwrap().contains_key(&ctx.instance_id()));
}

#[tokio::test]
async fn interrupted_runs_cannot_grow_activity_map_without_bound() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK);
    for _ in 0..1_025 {
        let mut ctx = RunContext::new(RunConfig::new("vbf"), ());
        mw.after_model(&mut ctx, &(), &mut tool_round("c1", "lookup"))
            .await
            .unwrap();
    }
    assert_eq!(mw.runs.lock().unwrap().len(), 1_024);
}

#[tokio::test]
async fn deferred_resume_restores_tool_activity_from_transcript() {
    let mw =
        VerifyBeforeFinishMiddleware::new(CHECK).with_trigger(|activity| activity.called("lookup"));
    let mut ctx = RunContext::new(RunConfig::new("resumed").with_max_model_calls(10), ())
        .with_deferred_results(crate::tool::DeferredToolResults::new().approve("c1"));
    mw.before_agent(&mut ctx, &()).await.unwrap();
    let mut request = tinyinference_llm::model::ModelRequest {
        messages: vec![
            Message::user("do the task"),
            Message::Assistant(tool_round("c1", "lookup").message),
            Message::tool("c1", "approved"),
        ],
        ..Default::default()
    };
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    let mut draft = answer("done");
    mw.after_model(&mut ctx, &(), &mut draft).await.unwrap();
    assert_eq!(draft.continue_turn.as_deref(), Some(CHECK));
}

#[tokio::test]
async fn deferred_resume_does_not_repeat_an_existing_check() {
    let mw = VerifyBeforeFinishMiddleware::new(CHECK);
    let mut ctx = RunContext::new(RunConfig::new("resumed").with_max_model_calls(10), ())
        .with_deferred_results(crate::tool::DeferredToolResults::new().approve("c1"));
    mw.before_agent(&mut ctx, &()).await.unwrap();
    let mut request = tinyinference_llm::model::ModelRequest {
        messages: vec![Message::user("do the task"), Message::user(CHECK)],
        ..Default::default()
    };
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    let mut draft = answer("done");
    mw.after_model(&mut ctx, &(), &mut draft).await.unwrap();
    assert!(draft.continue_turn.is_none());
}
