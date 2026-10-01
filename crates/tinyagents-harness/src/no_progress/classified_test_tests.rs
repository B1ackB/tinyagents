use super::*;

#[test]
fn equivalent_failures_survive_intervening_calls_and_clear_by_scope() {
    let tracker = ClassifiedFailureTracker::default();
    let a = ClassifiedFailure::new("permission", "search", "account-a");
    let b = ClassifiedFailure::new("permission", "search", "account-b");
    assert_eq!(tracker.record(&a, 1), NoProgress::Continue);
    assert_eq!(tracker.record(&b, 1), NoProgress::Continue);
    assert!(
        matches!(tracker.record(&a, 1), NoProgress::Halt(message) if message.contains("2 attempt(s)"))
    );
    tracker.clear(&a);
    assert_eq!(tracker.record(&a, 1), NoProgress::Continue);
    assert!(matches!(tracker.record(&b, 1), NoProgress::Halt(_)));
}

#[test]
fn zero_budget_stops_immediately() {
    let tracker = ClassifiedFailureTracker::default();
    let key = ClassifiedFailure::new("policy", "write", "project");
    assert!(matches!(tracker.record(&key, 0), NoProgress::Halt(_)));
}
