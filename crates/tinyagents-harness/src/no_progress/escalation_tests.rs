//! Tests for staged (warn, block, halt) escalation of successful repeats.

use super::*;
use crate::no_progress::{CallGate, SuccessfulRepeat, SuccessfulRepeatTracker};

const CALL: &str = "lookup\u{1}abc";

fn staged() -> SuccessfulRepeatTracker {
    SuccessfulRepeatTracker::new(4, 3).with_escalation(RepeatEscalation::default())
}

fn record(tracker: &SuccessfulRepeatTracker) -> SuccessfulRepeat {
    tracker.record_call_outcome(CALL, "same result")
}

#[test]
fn recurrence_warns_at_the_first_threshold_instead_of_halting() {
    let tracker = staged();
    assert_eq!(record(&tracker), SuccessfulRepeat::Continue);
    assert_eq!(record(&tracker), SuccessfulRepeat::Continue);
    assert!(matches!(
        record(&tracker),
        SuccessfulRepeat::Warn(note) if note.contains("identical result")
    ));
    assert_eq!(
        record(&tracker),
        SuccessfulRepeat::Continue,
        "the warning is reported once per signature"
    );
}

#[test]
fn a_call_is_blocked_two_repeats_after_the_warning() {
    let tracker = staged();
    for _ in 0..3 {
        record(&tracker);
    }
    assert_eq!(
        tracker.pre_call(CALL),
        CallGate::Allow,
        "the call after the warning still runs"
    );
    record(&tracker);
    assert!(matches!(
        tracker.pre_call(CALL),
        CallGate::Block(text) if text.contains("not executed")
    ));
}

#[test]
fn a_second_block_in_the_run_halts() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    assert!(matches!(
        tracker.pre_call(CALL),
        CallGate::Halt(text) if text.contains("blocked")
    ));
}

#[test]
fn an_unseen_call_or_a_changed_result_is_never_blocked() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert_eq!(tracker.pre_call("other\u{1}xyz"), CallGate::Allow);
    // The same call now returns something new: the next attempt is not a repeat.
    tracker.record_call_outcome(CALL, "different result");
    assert_eq!(tracker.pre_call(CALL), CallGate::Allow);
}

#[test]
fn a_call_that_ran_through_to_the_block_count_halts() {
    // A host that never consults `pre_call` still gets bounded.
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(record(&tracker), SuccessfulRepeat::Halt(_)));
}

#[test]
fn without_escalation_the_first_threshold_halts_and_nothing_blocks() {
    let tracker = SuccessfulRepeatTracker::new(4, 3);
    record(&tracker);
    record(&tracker);
    assert!(matches!(record(&tracker), SuccessfulRepeat::Halt(_)));
    assert_eq!(tracker.pre_call(CALL), CallGate::Allow);
    assert_eq!(tracker.pre_call(CALL), CallGate::Allow);
}

#[test]
fn custom_escalation_moves_the_block_and_halt_points() {
    let tracker = SuccessfulRepeatTracker::new(4, 3).with_escalation(RepeatEscalation {
        block_after_warn: 1,
        blocks_before_halt: 1,
    });
    for _ in 0..3 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Halt(_)));
}

#[test]
fn identical_batches_warn_then_halt_two_repeats_later() {
    let tracker = staged();
    assert_eq!(
        tracker.record_call_batch("b", true, false),
        SuccessfulRepeat::Continue
    );
    tracker.record_call_batch("b", true, false);
    assert!(matches!(
        tracker.record_call_batch("b", true, false),
        SuccessfulRepeat::Warn(note) if note.contains("tool-call batch")
    ));
    assert_eq!(
        tracker.record_call_batch("b", true, false),
        SuccessfulRepeat::Continue
    );
    assert!(matches!(
        tracker.record_call_batch("b", true, false),
        SuccessfulRepeat::Halt(_)
    ));
}

#[test]
fn identical_output_warns_then_halts_two_repeats_later() {
    let tracker = staged();
    let mut verdicts = Vec::new();
    for i in 0..6 {
        tracker.record_output("same narration", false);
        // Distinct call signatures keep the batch streak out of this test.
        verdicts.push(tracker.record_call_batch(&format!("b{i}"), true, false));
    }
    assert!(matches!(verdicts[3], SuccessfulRepeat::Warn(_)));
    assert_eq!(verdicts[4], SuccessfulRepeat::Continue);
    assert!(matches!(verdicts[5], SuccessfulRepeat::Halt(_)));
}

#[test]
fn blocks_survive_a_ledger_reset_but_not_a_full_reset() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    tracker.reset_ledger();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(
        matches!(tracker.pre_call(CALL), CallGate::Halt(_)),
        "the block count is run-wide and outlives a context eviction"
    );
    tracker.reset();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
}
