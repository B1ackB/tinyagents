//! Tests for the prompt-cache miss accounting in
//! [`PromptCacheGuardMiddleware`].

#[allow(unused_imports)]
use super::*;

use std::sync::Arc;

use crate::context::{RunConfig, RunContext};
use crate::events::{AgentEvent, RecordingListener};
use crate::middleware::{Middleware, PromptCacheGuardMiddleware};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::usage::Usage;

fn response(input: u64, cached: u64) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        },
        usage: Some(Usage {
            input_tokens: input,
            cache_read_tokens: cached,
            ..Usage::default()
        }),
        finish_reason: None,
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn ctx(run: &str, thread: Option<&str>) -> (RunContext, Arc<RecordingListener>) {
    let mut config = RunConfig::new(run);
    if let Some(thread) = thread {
        config = config.with_thread(thread);
    }
    let c = RunContext::new(config, ());
    let recorder = Arc::new(RecordingListener::new());
    c.events.subscribe(recorder.clone());
    (c, recorder)
}

/// One model call: the guard's `before_model`, then `after_model` with `usage`.
async fn call(
    guard: &PromptCacheGuardMiddleware,
    c: &mut RunContext,
    system: &str,
    input: u64,
    cached: u64,
) {
    call_on(guard, c, system, "model-a", input, cached).await;
}

async fn call_on(
    guard: &PromptCacheGuardMiddleware,
    c: &mut RunContext,
    system: &str,
    model: &str,
    input: u64,
    cached: u64,
) {
    let mut request = ModelRequest {
        messages: vec![Message::system(system), Message::user("hello")],
        ..Default::default()
    };
    Middleware::<(), ()>::before_model(guard, c, &(), &mut request)
        .await
        .unwrap();
    let mut resp = response(input, cached);
    resp.resolved_model = Some(tinyinference_llm::model::ResolvedModel {
        name: model.to_string(),
        requested: None,
        source: tinyinference_llm::model::ModelResolutionSource::RegistryDefault,
    });
    Middleware::<(), ()>::after_model(guard, c, &(), &mut resp)
        .await
        .unwrap();
}

fn misses(recorder: &RecordingListener) -> Vec<AgentEvent> {
    recorder
        .events()
        .into_iter()
        .map(|r| r.event)
        .filter(|e| matches!(e, AgentEvent::PromptCacheMiss { .. }))
        .collect()
}

#[tokio::test]
async fn a_sharp_cache_drop_emits_one_prompt_cache_miss() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys", 10_000, 9_000).await;
    call(&guard, &mut c, "sys", 12_000, 2_000).await;

    let found = misses(&recorder);
    assert_eq!(found.len(), 1);
    let AgentEvent::PromptCacheMiss {
        expected_cached_tokens,
        cached_tokens,
        wasted_input_tokens,
        ..
    } = found[0].clone()
    else {
        unreachable!()
    };
    assert_eq!(
        (expected_cached_tokens, cached_tokens, wasted_input_tokens),
        (10_000, 2_000, 8_000)
    );
}

#[tokio::test]
async fn a_warm_cache_emits_nothing() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys", 10_000, 0).await;
    call(&guard, &mut c, "sys", 12_000, 10_000).await;
    assert!(misses(&recorder).is_empty());
}

#[tokio::test]
async fn a_prefix_the_guard_saw_change_is_not_reported_as_a_silent_miss() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys v1", 10_000, 9_000).await;
    call(&guard, &mut c, "sys v2", 10_500, 0).await;
    assert!(misses(&recorder).is_empty());
    assert_eq!(
        guard.layout_events().len(),
        1,
        "the layout change is recorded"
    );
}

#[tokio::test]
async fn a_thread_shares_its_baseline_across_runs() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut first, _) = ctx("run-1", Some("thread-a"));
    call(&guard, &mut first, "sys", 10_000, 9_000).await;
    let (mut second, recorder) = ctx("run-2", Some("thread-a"));
    call(&guard, &mut second, "sys", 11_000, 0).await;
    assert_eq!(misses(&recorder).len(), 1);
}

#[tokio::test]
async fn runs_without_a_thread_do_not_share_a_baseline() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut first, _) = ctx("run-1", None);
    call(&guard, &mut first, "sys", 10_000, 9_000).await;
    let (mut second, recorder) = ctx("run-2", None);
    call(&guard, &mut second, "sys", 11_000, 0).await;
    assert!(misses(&recorder).is_empty());
}

#[tokio::test]
async fn the_noise_floor_is_configurable() {
    let guard = PromptCacheGuardMiddleware::new().with_cache_miss_noise_floor(50_000);
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys", 10_000, 9_000).await;
    call(&guard, &mut c, "sys", 12_000, 0).await;
    assert!(misses(&recorder).is_empty());
}

#[tokio::test]
async fn a_compaction_between_calls_is_not_a_cache_miss() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys", 10_000, 9_000).await;
    // What ContextCompressionMiddleware does when it rewrites the prefix.
    c.mark_prompt_prefix_changed();
    call(&guard, &mut c, "sys", 3_000, 100).await;
    assert!(misses(&recorder).is_empty());
    // The new prefix then gets its own baseline.
    call(&guard, &mut c, "sys", 4_000, 0).await;
    assert_eq!(misses(&recorder).len(), 1);
}

#[tokio::test]
async fn a_compacted_thread_keeps_its_epoch_in_a_fresh_run() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut first, _) = ctx("run-1", Some("thread-a"));
    call(&guard, &mut first, "sys", 10_000, 9_000).await;
    first.mark_prompt_prefix_changed();
    call(&guard, &mut first, "sys", 3_000, 100).await;

    let (mut second, recorder) = ctx("run-2", Some("thread-a"));
    call(&guard, &mut second, "sys", 4_000, 3_000).await;
    assert!(misses(&recorder).is_empty());
}

#[tokio::test]
async fn a_cache_miss_uses_the_active_model_call_id() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call(&guard, &mut c, "sys", 10_000, 9_000).await;
    c.active_model_call = Some(crate::ids::CallId::new("r-model-2"));
    call(&guard, &mut c, "sys", 12_000, 0).await;

    let event = misses(&recorder).pop().expect("cache miss event");
    let AgentEvent::PromptCacheMiss { call_id, .. } = event else {
        unreachable!()
    };
    assert_eq!(call_id, crate::ids::CallId::new("r-model-2"));
}

#[tokio::test]
async fn a_model_switch_is_not_a_cache_miss() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut c, recorder) = ctx("r", None);
    call_on(&guard, &mut c, "sys", "model-a", 10_000, 9_000).await;
    call_on(&guard, &mut c, "sys", "model-b", 11_000, 0).await;
    assert!(misses(&recorder).is_empty());
}

#[tokio::test]
async fn two_runs_sharing_a_run_id_but_not_a_thread_do_not_share_a_baseline() {
    let guard = PromptCacheGuardMiddleware::new();
    let (mut first, _) = ctx("same-id", None);
    call(&guard, &mut first, "sys", 10_000, 9_000).await;
    let (mut second, recorder) = ctx("same-id", None);
    call(&guard, &mut second, "sys", 11_000, 0).await;
    assert!(misses(&recorder).is_empty());
}
