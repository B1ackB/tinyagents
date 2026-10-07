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
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse, ProviderError};
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
    assert_eq!(primary.attempts(), 1, "auth is never retried on the same model");
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
async fn format_error_surfaces_without_trying_the_fallback() {
    let primary = ScriptedOutcomes::failing(400, "messages: field required", false);
    let backup = ScriptedOutcomes::answering("must not be reached");
    let error = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("a malformed request is not fixed by another model");
    assert!(matches!(error, TinyAgentsError::Provider(_)), "got {error:?}");
    assert_eq!(primary.attempts(), 1);
    assert_eq!(backup.attempts(), 0);
}

#[tokio::test]
async fn model_specific_format_error_falls_back() {
    let primary = ScriptedOutcomes::failing(400, "this model does not support tools", false);
    let backup = ScriptedOutcomes::answering("from backup");
    let run = harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("capability gap falls back");
    assert_eq!(run.text(), Some("from backup".to_string()));
    assert_eq!(primary.attempts(), 1);
}

#[tokio::test]
async fn context_overflow_surfaces_without_trying_the_fallback() {
    let primary = ScriptedOutcomes::failing(
        400,
        "This model's maximum context length is 8192 tokens",
        false,
    );
    let backup = ScriptedOutcomes::answering("must not be reached");
    harness(&primary, &backup, 3)
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect_err("compaction, not failover, handles overflow");
    assert_eq!(backup.attempts(), 0);
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
