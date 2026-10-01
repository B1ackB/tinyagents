use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn call_estimate_total_saturates_instead_of_wrapping() {
    let est = CallEstimate::new("m", u64::MAX, 10);
    assert_eq!(est.estimated_total_tokens(), u64::MAX);
}

#[test]
fn call_estimate_builders_attribute_agent_and_thread() {
    let est = CallEstimate::new("m", 100, 20)
        .with_agent("lead")
        .with_thread("t-1");
    assert_eq!(est.agent_id.as_deref(), Some("lead"));
    assert_eq!(est.thread_id, Some(ThreadId::from("t-1")));
    assert_eq!(est.estimated_total_tokens(), 120);
}

#[test]
fn call_estimate_round_trips_through_serde() {
    let est = CallEstimate::new("m", 1, 2).with_agent("a");
    let json = serde_json::to_string(&est).expect("serializes");
    let back: CallEstimate = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(back, est);
}

#[test]
fn permit_runs_release_hook_exactly_once_on_drop() {
    let released = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&released);
    {
        let _permit = Permit::with_release(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(released.load(Ordering::SeqCst), 0, "not released early");
    }
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn permit_release_returns_capacity_immediately() {
    let released = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&released);
    let permit = Permit::with_release(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    permit.release();
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[test]
fn unlimited_permit_carries_no_reservation() {
    let permit = Permit::unlimited();
    assert!(permit.id().is_none());
    assert!(permit.reserved_tokens().is_none());
}

#[test]
fn permit_metadata_builders_are_readable() {
    let permit = Permit::unlimited()
        .with_id("grant-7")
        .with_reserved_tokens(512);
    assert_eq!(permit.id(), Some("grant-7"));
    assert_eq!(permit.reserved_tokens(), Some(512));
    assert!(format!("{permit:?}").contains("grant-7"));
}

#[test]
fn context_state_utilization_is_none_without_a_known_window() {
    let state = ContextState {
        prompt_tokens: 1_000,
        ..ContextState::default()
    };
    assert_eq!(state.utilization(), None);

    let zero_window = ContextState {
        prompt_tokens: 1_000,
        context_window_tokens: Some(0),
        ..ContextState::default()
    };
    assert_eq!(zero_window.utilization(), None);
}

#[test]
fn context_state_utilization_is_the_occupied_fraction() {
    let state = ContextState {
        prompt_tokens: 500,
        context_window_tokens: Some(2_000),
        ..ContextState::default()
    };
    let utilization = state.utilization().expect("window is known");
    assert!((utilization - 0.25).abs() < f64::EPSILON);
}

#[test]
fn compression_hint_defaults_to_none_and_classifies_severity() {
    assert_eq!(CompressionHint::default(), CompressionHint::None);
    assert!(!CompressionHint::None.is_advised());
    assert!(CompressionHint::Soft.is_advised());
    assert!(!CompressionHint::Soft.is_required());
    assert!(CompressionHint::Hard.is_advised());
    assert!(CompressionHint::Hard.is_required());
}

#[tokio::test]
async fn unlimited_gate_grants_a_costless_permit() {
    let gate = UnlimitedBudgetGate::new();
    let permit = gate
        .acquire(&CallEstimate::new("m", u64::MAX, u64::MAX))
        .await
        .expect("unlimited gate never refuses");
    assert!(permit.reserved_tokens().is_none());
}

#[tokio::test]
async fn unlimited_gate_records_usage_without_accumulating() {
    let gate = UnlimitedBudgetGate;
    gate.record(&Usage::new(1_000, 1_000))
        .await
        .expect("recording is infallible");
    // Recording cannot make a later acquire refuse.
    gate.acquire(&CallEstimate::new("m", 1, 1))
        .await
        .expect("still grants after recording");
}

#[test]
fn unlimited_gate_never_advises_compression_even_when_full() {
    let gate = UnlimitedBudgetGate::new();
    let full = ContextState {
        message_count: 10_000,
        prompt_tokens: 999_999,
        context_window_tokens: Some(1_000),
        iterations: 500,
    };
    assert_eq!(gate.compression_hint(&full), CompressionHint::None);
}

#[test]
fn unlimited_gate_is_usable_as_a_trait_object() {
    // Pins object safety: the harness stores this as `Arc<dyn BudgetGate>`.
    let gate: Arc<dyn BudgetGate> = Arc::new(UnlimitedBudgetGate::new());
    assert_eq!(
        gate.compression_hint(&ContextState::default()),
        CompressionHint::None
    );
}
