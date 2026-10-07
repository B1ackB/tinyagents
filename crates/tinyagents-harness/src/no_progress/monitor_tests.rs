//! Tests for the composed repeat monitor.

use super::*;

fn args(n: u32) -> String {
    format!("args-{n}")
}

#[test]
fn the_default_config_stages_escalation_and_enables_every_detector() {
    let config = RepeatProgressConfig::default();
    assert_eq!(config.escalation, Some(RepeatEscalation::default()));
    assert_eq!(config.output_threshold, DEFAULT_REPEAT_OUTPUT_THRESHOLD);
    assert_eq!(config.call_threshold, DEFAULT_REPEAT_CALL_THRESHOLD);
    assert_eq!(
        config.ping_pong_alternations,
        DEFAULT_PING_PONG_ALTERNATIONS
    );
    assert_eq!(
        config.post_compaction_window,
        DEFAULT_POST_COMPACTION_WINDOW
    );
}

#[test]
fn immediate_halt_config_is_the_historical_behaviour() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::immediate_halt());
    monitor.record_call("lookup", "a", "r");
    monitor.record_call("lookup", "a", "r");
    let observed = monitor.record_call("lookup", "a", "r");
    assert!(matches!(observed.verdict, SuccessfulRepeat::Halt(_)));
    assert_eq!(monitor.pre_call("lookup", "a"), CallGate::Allow);
}

#[test]
fn ping_pong_notes_come_back_with_the_observation() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    let mut notes = Vec::new();
    for _ in 0..3 {
        notes.extend(monitor.record_call("read", "a", "doc").notes);
        notes.extend(monitor.record_call("search", "b", "hits").notes);
    }
    assert!(notes.iter().any(|note| note.contains("alternating")));
}

#[test]
fn argument_churn_notes_come_back_with_the_observation() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig {
        // Out of the way so only the churn detector can speak.
        call_threshold: 99,
        ..RepeatProgressConfig::default()
    });
    let mut notes = Vec::new();
    for variant in 0..3 {
        for _ in 0..3 {
            notes.extend(monitor.record_call("search", &args(variant), "none").notes);
        }
    }
    assert!(notes.iter().any(|note| note.contains("search")));
}

#[test]
fn a_repeat_of_the_pre_compaction_tail_escalates_straight_to_a_block() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    monitor.record_call("read", "a", "doc");
    monitor.on_context_evicted();

    let observed = monitor.record_call("read", "a", "doc");
    assert!(
        observed.notes.iter().any(|note| note.contains("compact")),
        "the model is told why: {:?}",
        observed.notes
    );
    assert!(
        matches!(monitor.pre_call("read", "a"), CallGate::Block(_)),
        "one repeat after compaction is enough to block the next, not three"
    );
}

#[test]
fn a_fresh_call_after_compaction_is_not_blocked() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    monitor.record_call("read", "a", "doc");
    monitor.on_context_evicted();
    monitor.record_call("read", "other", "doc-2");
    assert_eq!(monitor.pre_call("read", "other"), CallGate::Allow);
    assert_eq!(monitor.pre_call("read", "a"), CallGate::Allow);
}

#[test]
fn eviction_keeps_the_block_count_but_clears_the_ledger() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc");
    }
    assert!(matches!(monitor.pre_call("read", "a"), CallGate::Block(_)));
    monitor.on_context_evicted();
    assert_eq!(
        monitor.pre_call("read", "a"),
        CallGate::Allow,
        "the evicted result no longer counts"
    );
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc");
    }
    assert!(
        matches!(monitor.pre_call("read", "a"), CallGate::Halt(_)),
        "but the earlier block still counts toward the halt"
    );
}
