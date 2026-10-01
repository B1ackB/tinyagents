use super::*;

fn request() -> TurnContextRequest {
    TurnContextRequest::new("planner", "thread-1", "what changed today?")
}

#[test]
fn new_populates_every_field() {
    let req = request();
    assert_eq!(req.agent_id, "planner");
    assert_eq!(req.thread_id.as_str(), "thread-1");
    assert_eq!(req.user_text, "what changed today?");
}

#[test]
fn has_user_text_ignores_surrounding_whitespace() {
    assert!(request().has_user_text());
    assert!(!TurnContextRequest::new("planner", "t", "   \n\t ").has_user_text());
    assert!(!TurnContextRequest::new("planner", "t", "").has_user_text());
    // A single visible character still counts — the check is "blank", not
    // "meaningful".
    assert!(TurnContextRequest::new("planner", "t", "?").has_user_text());
}

#[test]
fn request_round_trips_through_serde() {
    let req = request();
    let json = serde_json::to_string(&req).expect("serialize");
    let back: TurnContextRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(req, back);
}

#[test]
fn a_blank_user_text_reports_no_user_text() {
    // `TurnContextRequest` has no `Default` on purpose (`ThreadId` has
    // none, and a blank agent id would compose the wrong identity), so
    // construct the empty-text case explicitly.
    let req = TurnContextRequest::new("agent", ThreadId::from("thread-1"), "   ");
    assert!(!req.has_user_text());
}

#[tokio::test]
async fn static_composer_returns_its_fixed_prompt() {
    let composer = StaticContextComposer::new("you are a careful assistant");
    assert_eq!(composer.system_prompt(), "you are a careful assistant");
    assert_eq!(
        composer
            .compose_system_prompt(&request())
            .await
            .expect("prompt"),
        "you are a careful assistant"
    );
}

#[tokio::test]
async fn static_composer_prompt_is_independent_of_the_request() {
    let composer = StaticContextComposer::new("fixed");
    let a = TurnContextRequest::new("planner", "thread-1", "first");
    let b = TurnContextRequest::new("researcher", "thread-2", "second");
    assert_eq!(
        composer.compose_system_prompt(&a).await.expect("prompt a"),
        composer.compose_system_prompt(&b).await.expect("prompt b"),
    );
}

#[tokio::test]
async fn static_composer_preamble_is_empty_and_not_an_error() {
    let composer = StaticContextComposer::new("fixed");
    let preamble = composer.preamble(&request()).await.expect("preamble");
    assert!(preamble.is_empty());
}

#[tokio::test]
async fn empty_composer_contributes_nothing() {
    let composer = StaticContextComposer::empty();
    assert_eq!(
        composer
            .compose_system_prompt(&request())
            .await
            .expect("prompt"),
        ""
    );
    assert!(
        composer
            .preamble(&request())
            .await
            .expect("preamble")
            .is_empty()
    );
}

#[tokio::test]
async fn usable_as_a_trait_object() {
    // Pins object safety: the harness stores this capability as
    // `Arc<dyn ContextComposer>`, so a non-dyn-safe signature would only
    // fail at the wiring site, not here.
    let composer: std::sync::Arc<dyn ContextComposer> =
        std::sync::Arc::new(StaticContextComposer::new("dyn"));
    assert_eq!(
        composer
            .compose_system_prompt(&request())
            .await
            .expect("prompt"),
        "dyn"
    );
}
