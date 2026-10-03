//! Tests for the image-aware token estimate and trim.

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::Middleware;
use tinyinference_llm::message::Message as TaMessage;
use tinyinference_llm::model::ModelProfile;
use tinyinference_llm::model::ModelRequest;

// Image-aware token estimation. A base64 image marker must be priced at the
// flat IMAGE_MARKER_TOKEN_COST, not chars/4 of its payload — otherwise one
// image reads as millions of tokens and the trimmer evicts everything,
// including the system prompt.

#[test]
fn estimate_text_tokens_markerless_is_chars_over_four() {
    assert_eq!(estimate_text_tokens(&"a".repeat(40)), 40_u64.div_ceil(4));
    assert_eq!(estimate_text_tokens(""), 0);
}

#[test]
fn estimate_text_tokens_prices_image_marker_flat_not_by_length() {
    let huge = "x".repeat(40_000);
    let text = format!("[IMAGE:{huge}]");
    let tokens = estimate_text_tokens(&text);
    // chars/4 of the payload would be ~10_000; the flat price is 1_200.
    assert!(
        tokens >= IMAGE_MARKER_TOKEN_COST,
        "at least the flat image cost: {tokens}"
    );
    assert!(
        tokens < 2_000,
        "image priced flat, not by base64 length: {tokens}"
    );
}

#[test]
fn estimate_text_tokens_charges_each_image_marker_once() {
    let tokens = estimate_text_tokens("[IMAGE:aaaa] and [IMAGE:bbbb]");
    assert!(
        tokens >= 2 * IMAGE_MARKER_TOKEN_COST,
        "two images each priced: {tokens}"
    );
    assert!(
        tokens < 2 * IMAGE_MARKER_TOKEN_COST + 100,
        "no runaway from the surrounding text: {tokens}"
    );
}

#[test]
fn an_unterminated_marker_is_counted_as_text() {
    let text = format!("[IMAGE:{}", "y".repeat(400));
    assert_eq!(estimate_text_tokens(&text), (text.len() as u64).div_ceil(4));
}

#[test]
fn the_input_budget_reserves_a_proportional_reply() {
    // window/10, floored at 512 and capped at max(8192, window/4).
    assert_eq!(legacy_max_input_tokens(8_192), 8_192 - 819);
    assert_eq!(legacy_max_input_tokens(1_000), 1_000 - 512);
    assert_eq!(legacy_max_input_tokens(200_000), 200_000 - 20_000);
    assert_eq!(legacy_max_input_tokens(2_000_000), 2_000_000 - 200_000);
}

async fn trim(window: u64, messages: Vec<TaMessage>) -> Vec<TaMessage> {
    let mw = ImageAwareMessageTrimMiddleware::for_context_window(window);
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    let mut request = ModelRequest {
        messages,
        ..Default::default()
    };
    mw.before_model(&mut ctx, &(), &mut request).await.unwrap();
    request.messages
}

#[tokio::test]
async fn a_request_inside_the_budget_is_untouched() {
    let messages = vec![TaMessage::system("sys"), TaMessage::user("hi")];
    let kept = trim(8_192, messages).await;
    assert_eq!(kept.len(), 2);
}

#[tokio::test]
async fn one_large_image_does_not_evict_the_transcript() {
    let image = format!("[IMAGE:{}]", "x".repeat(400_000));
    let messages = vec![
        TaMessage::system("sys"),
        TaMessage::user("earlier"),
        TaMessage::user(image),
    ];
    let kept = trim(32_000, messages).await;
    assert_eq!(
        kept.len(),
        3,
        "the image is priced flat, so nothing is evicted"
    );
}

#[tokio::test]
async fn eviction_keeps_system_messages_and_order_and_drops_oldest_first() {
    let big = "z".repeat(4_000);
    let messages = vec![
        TaMessage::system("sys-1"),
        TaMessage::user(format!("old {big}")),
        TaMessage::system("sys-2"),
        TaMessage::user(format!("mid {big}")),
        TaMessage::user("newest"),
    ];
    // Budget ~1_000 input tokens: two big bodies cannot both stay.
    let kept = trim(1_600, messages).await;
    let texts: Vec<String> = kept.iter().map(|m| m.text()).collect();
    assert!(texts.contains(&"sys-1".to_string()), "{texts:?}");
    assert!(texts.contains(&"sys-2".to_string()), "{texts:?}");
    assert!(texts.contains(&"newest".to_string()), "{texts:?}");
    assert!(!texts.iter().any(|t| t.starts_with("old ")), "{texts:?}");
    let sys1 = texts.iter().position(|t| t == "sys-1").unwrap();
    let sys2 = texts.iter().position(|t| t == "sys-2").unwrap();
    assert!(sys1 < sys2, "relative order is preserved: {texts:?}");
}

#[tokio::test]
async fn eviction_never_leaves_an_orphaned_leading_tool_result() {
    let big = "z".repeat(4_000);
    let messages = vec![
        TaMessage::system("sys"),
        TaMessage::user(format!("ask {big}")),
        TaMessage::tool("call-1", "answer"),
        TaMessage::user("newest"),
    ];
    let kept = trim(1_100, messages).await;
    let first_non_system = kept
        .iter()
        .find(|m| !matches!(m, TaMessage::System(_)))
        .expect("something remains");
    assert!(
        !matches!(first_non_system, TaMessage::Tool(_)),
        "a leading tool result without its call is a provider 400: {kept:?}"
    );
}

#[tokio::test]
async fn eviction_preserves_ephemeral_artifact_index_on_hoisting_models() {
    let mut request = ModelRequest::new(vec![TaMessage::user("x".repeat(8_000))]);
    let profile = ModelProfile {
        hoists_system_messages: true,
        ..ModelProfile::default()
    };
    push_ephemeral_instruction(
        &mut request,
        "stored artifact: outputs/result.json",
        Some(&profile),
    );
    request.messages.push(TaMessage::user("latest question"));
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    ctx.model_profile = Some(profile);

    ImageAwareMessageTrimMiddleware::for_context_window(1_600)
        .before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();

    let text = request
        .messages
        .iter()
        .map(TaMessage::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains(&"x".repeat(8_000)));
    assert!(text.contains("stored artifact: outputs/result.json"));
    assert!(text.contains("latest question"));
}

#[tokio::test]
async fn rescued_note_is_charged_before_trim_stops() {
    let profile = ModelProfile {
        hoists_system_messages: true,
        ..ModelProfile::default()
    };
    let mut request = ModelRequest::new(vec![TaMessage::user("x".repeat(8_000))]);
    push_ephemeral_instruction(&mut request, "artifact".repeat(60), Some(&profile));
    request.messages.push(TaMessage::user("m".repeat(4_000)));
    request.messages.push(TaMessage::user("latest"));
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    ctx.model_profile = Some(profile);
    let trim = ImageAwareMessageTrimMiddleware::for_context_window(1_600);

    trim.before_model(&mut ctx, &(), &mut request)
        .await
        .unwrap();

    let text = request
        .messages
        .iter()
        .map(TaMessage::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains(&"m".repeat(4_000)));
    assert!(text.contains("artifact"));
    assert!(text.contains("latest"));
    let tokens: u64 = request.messages.iter().map(estimate_message_tokens).sum();
    assert!(
        tokens <= legacy_max_input_tokens(1_600),
        "rescued note exceeded the input budget: {tokens}"
    );
}
