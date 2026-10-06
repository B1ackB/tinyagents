//! Tests for the streaming model idle timeout and its consecutive-timeout
//! breaker.
//!
//! Every test runs on tokio's paused clock: a silent stream parks on a timer
//! the runtime auto-advances, so there are no real sleeps and the elapsed
//! virtual time is exact.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::Instant;

use crate::context::RunConfig;
use crate::error::TinyAgentsError;
use crate::limits::RunLimits;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{ScriptedModel, SlowModel};
use tinyinference_llm::message::{Message, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
};

/// One step of a scripted stream.
#[derive(Clone)]
enum Step {
    /// Yield this item.
    Item(ModelStreamItem),
    /// Wait this long (on the tokio clock) before the next step.
    Sleep(Duration),
    /// Never produce anything again.
    Hang,
}

fn delta(text: &str) -> Step {
    Step::Item(ModelStreamItem::MessageDelta(MessageDelta {
        text: text.to_string(),
        reasoning: String::new(),
        tool_call: None,
    }))
}

fn started() -> Step {
    Step::Item(ModelStreamItem::Started)
}

fn completed(text: &str) -> Step {
    Step::Item(ModelStreamItem::Completed(ModelResponse::assistant(text)))
}

/// A streaming model that plays one script per call; the last script repeats.
struct ScriptedStreams {
    scripts: Mutex<VecDeque<Vec<Step>>>,
    calls: Mutex<usize>,
}

impl ScriptedStreams {
    fn new(scripts: Vec<Vec<Step>>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            calls: Mutex::new(0),
        })
    }

    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ScriptedStreams {
    async fn invoke(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        unreachable!("these tests only drive the streaming path")
    }

    async fn stream(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        *self.calls.lock().unwrap() += 1;
        let script = {
            let mut scripts = self.scripts.lock().unwrap();
            if scripts.len() > 1 {
                scripts.pop_front().unwrap()
            } else {
                scripts.front().cloned().unwrap_or_default()
            }
        };
        let stream = futures::stream::unfold(VecDeque::from(script), |mut steps| async move {
            loop {
                match steps.pop_front()? {
                    Step::Item(item) => return Some((item, steps)),
                    Step::Sleep(delay) => tokio::time::sleep(delay).await,
                    Step::Hang => futures::future::pending::<()>().await,
                }
            }
        });
        Ok(ModelStream::new(Box::pin(stream)))
    }
}

fn harness_with(
    primary: Arc<dyn ChatModel<()>>,
    limits: RunLimits,
    attempts: usize,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary);
    harness.with_policy(RunPolicy {
        limits,
        retry: RetryPolicy::default()
            .with_max_attempts(attempts)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness
}

/// Runs a streaming turn, failing the test (instead of hanging) when the run
/// does not finish within a virtual hour.
async fn run(
    harness: &AgentHarness<()>,
    config: RunConfig,
) -> crate::error::Result<crate::middleware::AgentRun> {
    tokio::time::timeout(
        Duration::from_secs(3600),
        harness.invoke_streaming(&(), (), config, vec![Message::user("hi")]),
    )
    .await
    .expect("the run hung: no timeout ever fired on a silent stream")
}

#[tokio::test(start_paused = true)]
async fn silent_stream_fails_with_a_retryable_call_timeout() {
    let model = ScriptedStreams::new(vec![vec![started(), delta("par"), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(None),
        1,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("idle-run"))
        .await
        .expect_err("a stream that goes silent must fail");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("idle"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    assert!(crate::retry::is_retryable(&err));
    // Two windows of silence at most: the first attempt and (when retries are
    // enabled) one retry. Never the hour the run would otherwise hang for.
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "{:?}",
        began.elapsed()
    );
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_is_rearmed_by_every_event() {
    // 30 events 800ms apart: 24s in total, far past the 1s idle window, but no
    // single gap is. A total-duration timer would kill this call.
    let mut script = vec![started()];
    for _ in 0..30 {
        script.push(Step::Sleep(Duration::from_millis(800)));
        script.push(delta("x"));
    }
    script.push(completed(&"x".repeat(30)));
    let model = ScriptedStreams::new(vec![script]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        1,
    );

    let run = run(&harness, RunConfig::new("steady-run"))
        .await
        .expect("a steadily streaming call must not trip the idle timeout");

    assert_eq!(run.text().as_deref(), Some("x".repeat(30).as_str()));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn first_event_timeout_is_separate_from_the_idle_timeout() {
    // The provider takes 3s to emit anything: past the 1s idle window but
    // inside the 5s first-event window.
    let model = ScriptedStreams::new(vec![vec![
        Step::Sleep(Duration::from_secs(3)),
        started(),
        delta("ok"),
        completed("ok"),
    ]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(5_000)),
        1,
    );

    let run = run(&harness, RunConfig::new("slow-start-run"))
        .await
        .expect("a slow first event inside the first-event window must succeed");
    assert_eq!(run.text().as_deref(), Some("ok"));
    assert_eq!(model.calls(), 1);
}

#[tokio::test(start_paused = true)]
async fn first_event_timeout_fires_when_nothing_ever_arrives() {
    let model = ScriptedStreams::new(vec![vec![Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_stream_first_event_timeout_ms(Some(5_000))
            .with_max_consecutive_stream_idle_timeouts(None),
        1,
    );

    let began = Instant::now();
    let err = run(&harness, RunConfig::new("no-first-event"))
        .await
        .expect_err("a stream that never starts must fail");

    match &err {
        TinyAgentsError::CallTimeout(message) => {
            assert!(message.contains("first event"), "{message}");
        }
        other => panic!("expected CallTimeout, got {other:?}"),
    }
    // The first attempt waited the long first-event window, not the idle one.
    assert!(
        began.elapsed() >= Duration::from_secs(5),
        "{:?}",
        began.elapsed()
    );
}

#[tokio::test(start_paused = true)]
async fn first_event_window_defaults_to_the_idle_timeout() {
    let model = ScriptedStreams::new(vec![vec![Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(None),
        1,
    );

    let began = Instant::now();
    run(&harness, RunConfig::new("default-first"))
        .await
        .expect_err("a stream that never starts must fail");

    assert!(
        began.elapsed() < Duration::from_secs(5),
        "{:?}",
        began.elapsed()
    );
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_is_retried_and_the_retry_can_succeed() {
    let model = ScriptedStreams::new(vec![
        vec![started(), Step::Hang],
        vec![started(), delta("recovered"), completed("recovered")],
    ]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        3,
    );

    let run = run(&harness, RunConfig::new("retry-run"))
        .await
        .expect("the retry after an idle timeout must succeed");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert_eq!(model.calls(), 2);
}

#[tokio::test(start_paused = true)]
async fn idle_timeout_falls_back_to_the_next_model() {
    let primary = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let fallback = Arc::new(ScriptedModel::replies(vec!["fallback answer"]));
    let mut harness = harness_with(
        primary.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(None),
        1,
    );
    harness.register_model("fallback", fallback.clone());
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(None),
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy {
            models: vec!["primary".to_string(), "fallback".to_string()],
        }),
        ..RunPolicy::default()
    });

    let run = run(&harness, RunConfig::new("fallback-run"))
        .await
        .expect("an idle timeout must fall through to the fallback chain");

    assert_eq!(run.text().as_deref(), Some("fallback answer"));
    assert_eq!(fallback.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn breaker_stops_retrying_after_consecutive_idle_timeouts() {
    let primary = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let fallback = Arc::new(ScriptedModel::replies(vec!["never reached"]));
    let mut harness = harness_with(primary.clone(), RunLimits::default(), 10);
    harness.register_model("fallback", fallback.clone());
    harness.with_policy(RunPolicy {
        limits: RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(3))
            .with_max_retries_per_call(10),
        retry: RetryPolicy::default()
            .with_max_attempts(10)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy {
            models: vec!["primary".to_string(), "fallback".to_string()],
        }),
        ..RunPolicy::default()
    });

    let res = run(&harness, RunConfig::new("breaker-run")).await;
    eprintln!("PRIMARY CALLS {}", primary.calls());
    let err = run(&harness, RunConfig::new("breaker-run"))
        .await
        .expect_err("the breaker must fail the run");

    match &err {
        TinyAgentsError::LimitExceeded(message) => {
            assert!(message.contains("3 consecutive"), "{message}");
        }
        other => panic!("expected LimitExceeded, got {other:?}"),
    }
    assert!(!crate::retry::is_retryable(&err));
    assert_eq!(primary.calls(), 3, "the breaker must stop at its threshold");
    assert!(
        fallback.requests().is_empty(),
        "a tripped breaker must not fan out to the fallback chain"
    );
}

#[tokio::test(start_paused = true)]
async fn any_stream_event_resets_the_breaker() {
    // Threshold 2. Attempt 1 is silent (count 1); attempt 2 delivers an event
    // (count back to 0) before going silent (count 1); attempt 3 is silent
    // (count 2, tripped). Without the reset the breaker would trip on attempt 2.
    let model = ScriptedStreams::new(vec![
        vec![Step::Hang],
        vec![started(), delta("x"), Step::Hang],
        vec![Step::Hang],
    ]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default()
            .with_stream_idle_timeout_ms(Some(1_000))
            .with_max_consecutive_stream_idle_timeouts(Some(2))
            .with_max_retries_per_call(10),
        10,
    );

    let err = run(&harness, RunConfig::new("reset-run"))
        .await
        .expect_err("the breaker must eventually trip");

    assert!(matches!(err, TinyAgentsError::LimitExceeded(_)), "{err:?}");
    assert_eq!(model.calls(), 3);
}

#[tokio::test(start_paused = true)]
async fn disabled_idle_timeout_waits_for_a_slow_stream() {
    let model = ScriptedStreams::new(vec![vec![
        started(),
        Step::Sleep(Duration::from_secs(900)),
        delta("late"),
        completed("late"),
    ]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(None),
        1,
    );

    let run = run(&harness, RunConfig::new("disabled-run"))
        .await
        .expect("with the idle timeout disabled a slow stream must complete");
    assert_eq!(run.text().as_deref(), Some("late"));
}

#[tokio::test(start_paused = true)]
async fn run_deadline_still_wins_as_a_terminal_timeout() {
    let model = ScriptedStreams::new(vec![vec![started(), Step::Hang]]);
    let harness = harness_with(
        model.clone(),
        RunLimits::default().with_stream_idle_timeout_ms(Some(60_000)),
        3,
    );

    let err = run(
        &harness,
        RunConfig::new("deadline-run").with_timeout_ms(500),
    )
    .await
    .expect_err("the run deadline must fire first");

    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert_eq!(model.calls(), 1, "a run-deadline timeout is not retried");
}

#[tokio::test(start_paused = true)]
async fn non_streaming_calls_ignore_the_idle_timeout() {
    // A buffered call has no inter-event gaps to measure: a 300s call under a
    // 1s idle timeout must still complete.
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "slow",
        Arc::new(SlowModel::new(Duration::from_secs(300), "done")),
    );
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_stream_idle_timeout_ms(Some(1_000)),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke(
            &(),
            (),
            RunConfig::new("unary-run"),
            vec![Message::user("hi")],
        )
        .await
        .expect("the idle timeout must not apply to non-streaming calls");
    assert_eq!(run.text().as_deref(), Some("done"));
}
