use super::*;

use tinyinference_llm::model::{ModelRequest, ModelResponse};

/// Minimal model double: replies with a fixed string so a resolved model can
/// be identified by the text it produces as well as by pointer identity.
struct EchoModel(&'static str);

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for EchoModel {
    async fn invoke(
        &self,
        _state: &State,
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant(self.0))
    }
}

fn fixed(reply: &'static str) -> FixedModelResolver<()> {
    FixedModelResolver::new(Arc::new(EchoModel(reply)))
}

// ── ModelResolveRequest invariants ───────────────────────────────────────

#[test]
fn new_request_has_no_role_and_is_not_lead() {
    let req = ModelResolveRequest::new("planner");
    assert_eq!(req.agent_id, "planner");
    assert_eq!(req.role(), None);
    assert!(!req.is_team_lead);
    assert_eq!(req.model_pin(), None);
}

#[test]
fn builders_set_role_and_lead() {
    let req = ModelResolveRequest::new("planner")
        .with_role("researcher")
        .as_team_lead();
    assert_eq!(req.role(), Some("researcher"));
    assert!(req.is_team_lead);
}

#[test]
fn model_pin_is_a_separate_channel_from_role() {
    // The bug #89 exists to prevent: with no pin field, a wiring author
    // puts the model id in `role`, the host reads it as an unknown role and
    // silently falls back to a default. The two must never collapse into
    // one string.
    let req = ModelResolveRequest::new("planner")
        .with_role("researcher")
        .with_model_pin("claude-3-5-sonnet");
    assert_eq!(req.role(), Some("researcher"));
    assert_eq!(req.model_pin(), Some("claude-3-5-sonnet"));
}

#[test]
fn a_pin_can_be_set_without_a_role() {
    // The common case for a user-authored agent: it pins a model and
    // declares no role at all.
    let req = ModelResolveRequest::new("planner").with_model_pin("local-llama");
    assert_eq!(req.role(), None);
    assert_eq!(req.model_pin(), Some("local-llama"));
}

#[test]
fn blank_model_pin_reads_as_absent() {
    // `model = ""` in a definition means "no pin", not a model named "".
    // Reaching a resolver as Some("") would fail the lookup and turn a
    // cosmetic config blank into an unroutable turn.
    for blank in ["", "   ", "\t\n"] {
        let req = ModelResolveRequest::new("planner").with_model_pin(blank);
        assert_eq!(
            req.model_pin(),
            None,
            "blank pin {blank:?} must read as absent"
        );
    }
}

#[test]
fn model_pin_accessor_trims_surrounding_whitespace() {
    let req = ModelResolveRequest::new("planner").with_model_pin("  gpt-4o  ");
    assert_eq!(req.model_pin(), Some("gpt-4o"));
}

#[test]
fn blank_role_reads_as_absent() {
    // A host mapping a missing field to `Some("")` must not select a
    // role-specific routing branch keyed on the empty string.
    assert_eq!(ModelResolveRequest::new("a").with_role("").role(), None);
    assert_eq!(
        ModelResolveRequest::new("a").with_role("  \t ").role(),
        None
    );
}

#[test]
fn role_accessor_trims_surrounding_whitespace() {
    let req = ModelResolveRequest::new("a").with_role("  lead  ");
    assert_eq!(req.role(), Some("lead"));
    // The stored value is untouched — trimming is a read-side courtesy, not
    // normalization the host's own round-trip has to anticipate.
    assert_eq!(req.role.as_deref(), Some("  lead  "));
}

#[test]
fn default_request_is_empty_and_not_lead() {
    let req = ModelResolveRequest::default();
    assert!(req.agent_id.is_empty());
    assert_eq!(req.role(), None);
    assert!(!req.is_team_lead);
    assert_eq!(req.model_pin(), None);
}

#[test]
fn request_round_trips_through_serde() {
    let req = ModelResolveRequest::new("planner")
        .with_role("researcher")
        .with_model_pin("claude-3-5-sonnet")
        .as_team_lead();
    let json = serde_json::to_string(&req).expect("serialize");
    let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, req);
}

#[test]
fn absent_role_is_omitted_and_restored() {
    let req = ModelResolveRequest::new("planner");
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(
        !json.contains("role"),
        "unset role must not serialize: {json}"
    );
    let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, req);
}

#[test]
fn absent_model_pin_is_omitted_and_restored() {
    let req = ModelResolveRequest::new("planner");
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(
        !json.contains("model_pin"),
        "unset model pin must not serialize: {json}"
    );
    let back: ModelResolveRequest = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, req);
    assert_eq!(back.model_pin(), None);
}

#[test]
fn request_deserializes_from_agent_id_alone() {
    // Hosts hand-writing this in config should not have to spell out
    // defaults for the two optional facts.
    let back: ModelResolveRequest =
        serde_json::from_str(r#"{"agent_id":"planner"}"#).expect("deserialize");
    assert_eq!(back, ModelResolveRequest::new("planner"));
}

// ── FixedModelResolver behaviour ─────────────────────────────────────────

#[tokio::test]
async fn fixed_resolver_returns_the_wrapped_model() {
    let resolver = fixed("hello");
    let model = resolver
        .resolve(&ModelResolveRequest::new("planner"))
        .await
        .expect("resolve");
    let response = model
        .invoke(&(), ModelRequest::default())
        .await
        .expect("invoke");
    assert_eq!(response.text(), "hello");
}

#[tokio::test]
async fn fixed_resolver_ignores_every_routing_field() {
    let resolver = fixed("hello");
    let lead = resolver
        .resolve(
            &ModelResolveRequest::new("lead")
                .with_role("planner")
                .as_team_lead(),
        )
        .await
        .expect("resolve lead");
    let worker = resolver
        .resolve(&ModelResolveRequest::new("worker").with_role("scribe"))
        .await
        .expect("resolve worker");
    // A pin is advisory: a host with exactly one model is entitled to
    // ignore it, and must not be expected to look the id up.
    let pinned = resolver
        .resolve(&ModelResolveRequest::new("pinned").with_model_pin("gpt-4o"))
        .await
        .expect("resolve pinned");
    assert!(
        Arc::ptr_eq(&lead, &worker) && Arc::ptr_eq(&lead, &pinned),
        "the fixed resolver must not route on agent id, role, lead status, or model pin"
    );
}

#[tokio::test]
async fn resolved_model_shares_the_wrapped_allocation() {
    // Resolution must hand back a handle to the same long-lived client, not
    // a clone — a per-turn copy would discard its connection pool.
    let resolver = fixed("hello");
    let resolved = resolver
        .resolve(&ModelResolveRequest::new("planner"))
        .await
        .expect("resolve");
    assert!(Arc::ptr_eq(resolver.model(), &resolved));
}

#[tokio::test]
async fn clone_resolves_to_the_same_model() {
    let resolver = fixed("hello");
    let clone = resolver.clone();
    let a = resolver
        .resolve(&ModelResolveRequest::new("a"))
        .await
        .expect("resolve a");
    let b = clone
        .resolve(&ModelResolveRequest::new("b"))
        .await
        .expect("resolve b");
    assert!(Arc::ptr_eq(&a, &b));
}

#[tokio::test]
async fn usable_behind_a_trait_object() {
    // The runtime holds `Arc<dyn ModelResolver<State>>`, so object safety is
    // part of the contract, not an implementation detail.
    let resolver: Arc<dyn ModelResolver<()>> = Arc::new(fixed("dyn"));
    let model = resolver
        .resolve(&ModelResolveRequest::new("planner"))
        .await
        .expect("resolve");
    let response = model
        .invoke(&(), ModelRequest::default())
        .await
        .expect("invoke");
    assert_eq!(response.text(), "dyn");
}
