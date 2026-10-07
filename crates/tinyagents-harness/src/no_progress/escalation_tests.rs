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
fn a_call_that_ran_through_the_block_stage_still_halts() {
    // A host that never consults `pre_call` (or whose predictions were
    // invalidated every time) is still bounded, at the count where a second
    // block would have happened.
    let tracker = staged();
    for _ in 0..5 {
        assert!(!matches!(record(&tracker), SuccessfulRepeat::Halt(_)));
    }
    assert!(matches!(record(&tracker), SuccessfulRepeat::Halt(_)));
}

#[test]
fn blocks_are_counted_per_call_signature() {
    let tracker = staged();
    let other = "read\u{1}other";
    for _ in 0..4 {
        record(&tracker);
        tracker.record_call_outcome(other, "other result");
    }
    // A parallel batch [lookup, read]: each is blocked once, neither halts.
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    assert!(matches!(tracker.pre_call(other), CallGate::Block(_)));
    // Each repeating once more halts.
    assert!(matches!(tracker.pre_call(CALL), CallGate::Halt(_)));
}

#[test]
fn a_new_result_clears_the_blocks_of_that_call() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    tracker.record_call_outcome(CALL, "something new");
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(
        matches!(tracker.pre_call(CALL), CallGate::Block(_)),
        "progress in between: this is a first block again, not a halt"
    );
}

#[test]
fn invalidating_predictions_keeps_the_ledger_counts() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    tracker.invalidate_predictions_except("edit\u{1}x");
    assert_eq!(tracker.pre_call(CALL), CallGate::Allow);
    assert_eq!(tracker.recurrence_count(CALL, "same result"), 4);
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
    let tracker = SuccessfulRepeatTracker::new(4, 3).with_escalation(RepeatEscalation::new(1, 1));
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

#[test]
fn progress_after_an_invalidation_still_clears_the_blocks() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    tracker.invalidate_predictions_except("edit\u{1}x");
    tracker.record_call_outcome(CALL, "a new result");
    for _ in 0..3 {
        tracker.record_call_outcome(CALL, "a new result");
    }
    assert!(
        matches!(tracker.pre_call(CALL), CallGate::Block(_)),
        "the new result was progress, so this is a first block, not a halt"
    );
}

#[test]
fn progress_after_a_ledger_reset_clears_the_blocks_too() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
    tracker.reset_ledger();
    for _ in 0..4 {
        tracker.record_call_outcome(CALL, "a new result");
    }
    assert!(matches!(tracker.pre_call(CALL), CallGate::Block(_)));
}

#[test]
fn extreme_thresholds_saturate_instead_of_overflowing() {
    let tracker = SuccessfulRepeatTracker::new(u32::MAX, u32::MAX)
        .with_escalation(RepeatEscalation::new(u32::MAX, u32::MAX));
    for _ in 0..3 {
        assert_eq!(record(&tracker), SuccessfulRepeat::Continue);
        assert_eq!(
            tracker.record_call_batch("sig", true, false),
            SuccessfulRepeat::Continue
        );
        assert!(matches!(tracker.pre_call(CALL), CallGate::Allow));
    }
}

#[test]
fn prediction_state_is_not_retained_without_escalation() {
    let tracker = SuccessfulRepeatTracker::new(4, 3);
    for n in 0..50 {
        tracker.record_call_outcome(&format!("read\u{1}{n}"), "result");
    }
    assert!(crate::no_progress::util::lock(&tracker.last_outcome).is_empty());
    assert!(crate::no_progress::util::lock(&tracker.predictable).is_empty());
}

#[test]
fn invalidating_all_predictions_stops_blocks_until_the_call_returns_again() {
    let tracker = staged();
    for _ in 0..4 {
        record(&tracker);
    }
    tracker.invalidate_all_predictions();
    assert!(matches!(tracker.pre_call(CALL), CallGate::Allow));
}
