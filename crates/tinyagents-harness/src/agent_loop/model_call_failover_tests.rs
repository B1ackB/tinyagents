//! Reason-aware failover wired into the model call: the retry/fallback path
//! must follow the [`crate::retry::decide`] table, not a single retryable flag.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::AgentEvent;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, RunPolicy};
use crate::testkit::{EventRecorder, FakeTool};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ProviderError,
};
use tinyinference_llm::tool::ToolCall;

/// Plays a script of outcomes (one per call); once exhausted it repeats the
/// last. `Err` entries are provider errors.
struct ScriptedOutcomes {
    script: Mutex<VecDeque<Result<ModelResponse, ProviderError>>>,
    attempts: Mutex<usize>,
}

impl ScriptedOutcomes {
    fn new(script: Vec<Result<ModelResponse, ProviderError>>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into()),
            attempts: Mutex::new(0),
        })
    }

    fn failing(status: u16, message: &str, retryable: bool) -> Arc<Self> {
        Self::new(vec![Err(provider_error(status, message, retryable))])
    }

    fn answering(text: &str) -> Arc<Self> {
        Self::new(vec![Ok(ModelResponse::assistant(text))])
    }

    fn attempts(&self) -> usize {
        *self.attempts.lock().unwrap()
    }
}

fn provider_error(status: u16, message: &str, retryable: bool) -> ProviderError {
    ProviderError {
        provider: "test".into(),
        status: Some(status),
        message: message.into(),
        retryable,
        ..ProviderError::default()
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedOutcomes {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        let mut script = self.script.lock().unwrap();
        let next = if script.len() > 1 {
            script.pop_front().unwrap()
        } else {
            script.front().cloned().expect("non-empty script")
        };
        next.map_err(|error| tinyinference_llm::Error::Provider(Box::new(error)))
    }
}

/// Fails every call with a non-provider error built by `make`.
struct ScriptedErrors {
    make: fn() -> tinyinference_llm::Error,
    attempts: Mutex<usize>,
}

impl ScriptedErrors {
    fn new(make: fn() -> tinyinference_llm::Error) -> Arc<Self> {
        Arc::new(Self {
            make,
            attempts: Mutex::new(0),
        })
    }
    fn attempts(&self) -> usize {
        *self.attempts.lock().unwrap()
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedErrors {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        Err((self.make)())
    }
}

fn harness_with_primary(
    primary: Arc<ScriptedErrors>,
    backup: &Arc<ScriptedOutcomes>,
    max_attempts: usize,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary);
    harness.register_model("backup", backup.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(max_attempts)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy::new(["primary", "backup"])),
        ..RunPolicy::default()
    });
    harness
}

fn harness(
    primary: &Arc<ScriptedOutcomes>,
    backup: &Arc<ScriptedOutcomes>,
    max_attempts: usize,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("backup", backup.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(max_attempts)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy::new(["primary", "backup"])),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn auth_failure_skips_retries_and_falls_back() {
    // `retryable: true` and three attempts allowed: only the reason can stop
    // the loop re-sending a rejected credential.
    let primary = ScriptedOutcomes::failing(401, "invalid api key", true);
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("fallback answers");
    assert_eq!(run.text(), Some("from backup".to_string()));
    assert_eq!(
        primary.attempts(),
        1,
        "auth is never retried on the same model"
    );
}

#[tokio::test]
async fn rate_limit_retries_the_same_model_then_falls_back() {
    let primary = ScriptedOutcomes::failing(429, "too many requests", true);
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("fallback answers");
    assert_eq!(run.text(), Some("from backup".to_string()));
    assert_eq!(primary.attempts(), 3);
}

#[tokio::test]
async fn provider_specific_format_error_falls_back() {
    // An OpenAI strict-schema rejection: another provider may accept it.
    let primary = ScriptedOutcomes::failing(
        400,
        "Invalid schema for function 'lookup': 'additionalProperties' is required",
        false,
    );
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a 4xx can be provider-specific, so the chain is walked");
    assert_eq!(run.text(), Some("from backup".to_string()));
    assert_eq!(primary.attempts(), 1, "format errors are not retried");
}

#[tokio::test]
async fn adapter_validation_error_falls_back() {
    let primary = ScriptedErrors::new(|| {
        tinyinference_llm::Error::Unsupported("tool choice not supported by adapter".into())
    });
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness_with_primary(primary.clone(), &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("adapter-level rejection falls back");
    assert_eq!(run.text(), Some("from backup".to_string()));
    assert_eq!(primary.attempts(), 1);
}

#[tokio::test]
async fn gateway_404_without_a_model_name_falls_back() {
    let primary = ScriptedOutcomes::failing(404, "route not found", false);
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a wrong route is not a model verdict");
    assert_eq!(run.text(), Some("from backup".to_string()));
}

fn windowed(max_input_tokens: u64) -> ModelProfile {
    ModelProfile {
        max_input_tokens: Some(max_input_tokens),
        ..ModelProfile::default()
    }
}

/// A model with a declared context window that either fails with a context
/// overflow or answers.
struct Windowed {
    profile: ModelProfile,
    overflow: bool,
    attempts: Mutex<usize>,
}

impl Windowed {
    fn new(window: u64, overflow: bool) -> Arc<Self> {
        Arc::new(Self {
            profile: windowed(window),
            overflow,
            attempts: Mutex::new(0),
        })
    }
    fn attempts(&self) -> usize {
        *self.attempts.lock().unwrap()
    }
}

#[async_trait]
impl ChatModel<()> for Windowed {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        *self.attempts.lock().unwrap() += 1;
        if self.overflow {
            return Err(tinyinference_llm::Error::Provider(Box::new(
                provider_error(
                    400,
                    "This model's maximum context length is 8192 tokens",
                    false,
                ),
            )));
        }
        Ok(ModelResponse::assistant("from sibling"))
    }
}

fn windowed_harness(primary: &Arc<Windowed>, sibling: &Arc<Windowed>) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", primary.clone());
    harness.register_model("sibling", sibling.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(2)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy::new(["primary", "sibling"])),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn context_overflow_falls_back_to_a_strictly_larger_window() {
    let primary = Windowed::new(8_000, true);
    let sibling = Windowed::new(128_000, false);
    let run = windowed_harness(&primary, &sibling)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("a larger-window sibling can take the request");
    assert_eq!(run.text(), Some("from sibling".to_string()));
    assert_eq!(primary.attempts(), 1, "overflow is not retried");
}

#[tokio::test]
async fn context_overflow_surfaces_when_no_sibling_has_a_larger_window() {
    let primary = Windowed::new(8_000, true);
    let sibling = Windowed::new(8_000, false);
    harness_err(&windowed_harness(&primary, &sibling)).await;
    assert_eq!(sibling.attempts(), 0, "an equal window cannot help");
}

#[tokio::test]
async fn context_overflow_surfaces_when_windows_are_unknown() {
    let primary = ScriptedOutcomes::failing(
        400,
        "This model's maximum context length is 8192 tokens",
        false,
    );
    let backup = ScriptedOutcomes::answering("must not be reached");
    harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("no evidence the sibling is larger");
    assert_eq!(backup.attempts(), 0);
}

async fn harness_err(harness: &AgentHarness<()>) {
    let error = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("overflow with no larger sibling surfaces");
    assert!(
        matches!(error, TinyAgentsError::Provider(_)),
        "got {error:?}"
    );
}

#[tokio::test]
async fn permanent_auth_failure_is_skipped_for_the_rest_of_the_run() {
    // Call 1: primary is revoked -> backup requests a tool. Call 2: the run
    // must go straight to backup without re-sending the revoked key.
    let primary = ScriptedOutcomes::failing(401, "API key has been revoked", false);
    let backup = ScriptedOutcomes::new(vec![
        Ok(crate::testkit::tool_call_response(ToolCall::new(
            "c1",
            "lookup",
            json!({}),
        ))),
        Ok(ModelResponse::assistant("done")),
    ]);
    let mut harness = harness(&primary, &backup, 3);
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("skip-hint"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("backup finishes the run");
    assert_eq!(run.text(), Some("done".to_string()));
    assert_eq!(primary.attempts(), 1, "revoked model tried once per run");
    assert_eq!(backup.attempts(), 2);
    assert!(
        recorder.events().iter().any(|event| matches!(
            event,
            AgentEvent::FallbackSkipped { model } if model == "primary"
        )),
        "the skip must be observable; got {:?}",
        recorder.kinds()
    );
}

#[tokio::test]
async fn plain_auth_failure_is_retried_on_the_next_call() {
    // Unlike a revoked key, a rejected-but-refreshable credential is not
    // remembered: the next call tries the primary again.
    let primary = ScriptedOutcomes::failing(401, "invalid api key", false);
    let backup = ScriptedOutcomes::new(vec![
        Ok(crate::testkit::tool_call_response(ToolCall::new(
            "c1",
            "lookup",
            json!({}),
        ))),
        Ok(ModelResponse::assistant("done")),
    ]);
    let mut harness = harness(&primary, &backup, 3);
    harness.register_tool(Arc::new(FakeTool::returning("lookup", "ok")));
    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("backup finishes the run");
    assert_eq!(primary.attempts(), 2);
}

fn single_model_harness(model: &Arc<ScriptedOutcomes>) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("primary", model.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        ..RunPolicy::default()
    });
    harness
}

#[tokio::test]
async fn a_written_off_model_is_still_tried_when_no_fallback_exists() {
    let primary = ScriptedOutcomes::answering("still here");
    let harness = single_model_harness(&primary);
    let mut ctx = RunContext::new(RunConfig::new("advisory-skip"), ());
    ctx.limits.skip_model_for_run("primary");
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("the skip hint is advisory");
    assert_eq!(run.text(), Some("still here".to_string()));
    assert_eq!(primary.attempts(), 1);
}

#[tokio::test]
async fn a_written_off_model_is_still_tried_when_the_only_fallback_is_written_off() {
    let primary = ScriptedOutcomes::answering("primary answers");
    let backup = ScriptedOutcomes::answering("backup answers");
    let harness = harness(&primary, &backup, 1);
    let mut ctx = RunContext::new(RunConfig::new("advisory-skip-2"), ());
    ctx.limits.skip_model_for_run("primary");
    ctx.limits.skip_model_for_run("backup");
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect("falls back to trying the resolved model");
    assert_eq!(run.text(), Some("primary answers".to_string()));
    assert_eq!(primary.attempts(), 1);
    assert_eq!(backup.attempts(), 0);
}

#[tokio::test]
async fn a_fallback_chosen_by_the_skip_hint_is_not_tried_twice() {
    // Chain [a, b, c, b]: `a` is written off, `b` is substituted and fails, `c`
    // fails; `b` must not be selected a second time.
    let a = ScriptedOutcomes::answering("a");
    let b = ScriptedOutcomes::failing(400, "bad request", false);
    let c = ScriptedOutcomes::failing(400, "bad request", false);
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("a", a.clone());
    harness.register_model("b", b.clone());
    harness.register_model("c", c.clone());
    harness.with_policy(RunPolicy {
        retry: RetryPolicy::default()
            .with_max_attempts(1)
            .with_backoff_sleep(false),
        fallback: Some(FallbackPolicy::new(["a", "b", "c", "b"])),
        ..RunPolicy::default()
    });
    let mut ctx = RunContext::new(RunConfig::new("visited"), ());
    ctx.limits.skip_model_for_run("a");
    harness
        .invoke_in_context(&(), ctx, vec![Message::user("hi")])
        .await
        .expect_err("every candidate fails");
    assert_eq!(a.attempts(), 0);
    assert_eq!(b.attempts(), 1, "b must be tried once");
    assert_eq!(c.attempts(), 1);
}
