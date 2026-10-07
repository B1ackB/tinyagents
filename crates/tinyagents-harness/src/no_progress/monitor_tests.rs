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
    monitor.record_call("lookup", "a", "r", true);
    monitor.record_call("lookup", "a", "r", true);
    let observed = monitor.record_call("lookup", "a", "r", true);
    assert!(matches!(observed.verdict, SuccessfulRepeat::Halt(_)));
    assert_eq!(monitor.pre_call("lookup", "a"), CallGate::Allow);
}

#[test]
fn ping_pong_notes_come_back_with_the_observation() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    let mut notes = Vec::new();
    for _ in 0..3 {
        notes.extend(monitor.record_call("read", "a", "doc", true).notes);
        notes.extend(monitor.record_call("search", "b", "hits", true).notes);
    }
    assert!(notes.iter().any(|note| note.contains("alternating")));
}

#[test]
fn argument_churn_notes_come_back_with_the_observation() {
    // The call threshold is out of the way so only the churn detector speaks.
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default().with_call_threshold(99));
    let mut notes = Vec::new();
    for variant in 0..3 {
        for _ in 0..3 {
            notes.extend(
                monitor
                    .record_call("search", &args(variant), "none", true)
                    .notes,
            );
        }
    }
    assert!(notes.iter().any(|note| note.contains("search")));
}

#[test]
fn a_repeat_of_an_already_repeating_pre_compaction_tail_only_warns() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    monitor.record_call("read", "a", "doc", true);
    monitor.record_call("read", "a", "doc", true);
    monitor.on_context_evicted();

    let observed = monitor.record_call("read", "a", "doc", true);
    assert!(
        observed.notes.iter().any(|note| note.contains("compact")),
        "the model is told why: {:?}",
        observed.notes
    );
    assert_eq!(
        monitor.pre_call("read", "a"),
        CallGate::Allow,
        "the guard never blocks"
    );
}

#[test]
fn one_re_read_after_compaction_is_neither_warned_nor_blocked() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    monitor.record_call("read", "a", "doc", true);
    monitor.on_context_evicted();

    let observed = monitor.record_call("read", "a", "doc", true);
    assert!(observed.notes.is_empty(), "{:?}", observed.notes);
    assert_eq!(monitor.pre_call("read", "a"), CallGate::Allow);
}

#[test]
fn a_state_changing_call_discards_the_prediction_for_other_calls() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc", true);
    }
    monitor.record_call("edit", "b", "ok", false);
    assert_eq!(
        monitor.pre_call("read", "a"),
        CallGate::Allow,
        "the read may return something new after an edit"
    );
}

#[test]
fn a_read_only_call_keeps_the_prediction() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc", true);
    }
    monitor.record_call("grep", "b", "hits", true);
    assert!(matches!(monitor.pre_call("read", "a"), CallGate::Block(_)));
}

#[test]
fn a_repeating_state_changing_call_is_still_predicted_for_itself() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("write", "a", "ok", false);
    }
    assert!(matches!(monitor.pre_call("write", "a"), CallGate::Block(_)));
}

#[test]
fn eviction_keeps_the_block_count_but_clears_the_ledger() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc", true);
    }
    assert!(matches!(monitor.pre_call("read", "a"), CallGate::Block(_)));
    monitor.on_context_evicted();
    assert_eq!(
        monitor.pre_call("read", "a"),
        CallGate::Allow,
        "the evicted result no longer counts"
    );
    for _ in 0..4 {
        monitor.record_call("read", "a", "doc", true);
    }
    assert!(
        matches!(monitor.pre_call("read", "a"), CallGate::Halt(_)),
        "but the earlier block of that call still counts toward the halt"
    );
}

#[test]
fn signatures_with_the_separator_inside_do_not_collide() {
    assert_ne!(
        call_signature("a\u{1}b", "c"),
        call_signature("a", "b\u{1}c")
    );
}

#[test]
fn an_untracked_state_changing_success_stops_block_predictions() {
    let monitor = RepeatMonitor::new(&RepeatProgressConfig::default());
    for _ in 0..4 {
        monitor.record_call("read", "x", "same", true);
    }
    // A read-only untracked success keeps the prediction.
    monitor.note_untracked_success(true);
    assert!(matches!(monitor.pre_call("read", "x"), CallGate::Block(_)));
    // A state-changing one discards it.
    monitor.note_untracked_success(false);
    assert!(matches!(monitor.pre_call("read", "x"), CallGate::Allow));
}
