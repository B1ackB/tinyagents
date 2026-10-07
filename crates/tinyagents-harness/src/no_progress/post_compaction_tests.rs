//! Tests for the post-compaction loop guard.

use super::*;

fn guard_with_tail(window: u32) -> PostCompactionGuard {
    let guard = PostCompactionGuard::new(window);
    assert!(!guard.record("read\u{1}a", "doc-a", true));
    assert!(!guard.record("search\u{1}b", "hits-b", true));
    assert!(!guard.record("read\u{1}c", "doc-c", true));
    guard.arm();
    guard
}

#[test]
fn a_call_repeating_the_pre_compaction_tail_is_flagged() {
    let guard = guard_with_tail(3);
    assert!(
        guard.record("search\u{1}b", "hits-b", true),
        "same call, same result as right before compaction"
    );
}

#[test]
fn the_same_call_with_a_new_result_is_progress() {
    let guard = guard_with_tail(3);
    assert!(!guard.record("search\u{1}b", "hits-b-after-the-index-changed", true));
}

#[test]
fn a_call_outside_the_tail_is_not_flagged() {
    let guard = guard_with_tail(3);
    assert!(!guard.record("write\u{1}z", "ok", true));
}

#[test]
fn only_the_window_after_compaction_is_watched() {
    let guard = guard_with_tail(3);
    for i in 0..3 {
        assert!(!guard.record("other\u{1}x", &format!("r{i}"), false));
    }
    assert!(
        !guard.record("search\u{1}b", "hits-b", true),
        "a repeat after the window is the ordinary ledger's business"
    );
}

#[test]
fn a_flagged_repeat_disarms_the_guard() {
    let guard = guard_with_tail(3);
    assert!(guard.record("read\u{1}c", "doc-c", true));
    assert!(!guard.record("search\u{1}b", "hits-b", true));
}

#[test]
fn only_the_last_window_calls_before_compaction_are_the_tail() {
    let guard = PostCompactionGuard::new(2);
    guard.record("old\u{1}1", "r1", true);
    guard.record("new\u{1}2", "r2", true);
    guard.record("new\u{1}3", "r3", true);
    guard.arm();
    assert!(
        !guard.record("old\u{1}1", "r1", true),
        "fell out of the tail"
    );
}

#[test]
fn a_zero_window_disables_the_guard() {
    let guard = PostCompactionGuard::new(0);
    guard.record("read\u{1}a", "doc", true);
    guard.arm();
    assert!(!guard.record("read\u{1}a", "doc", true));
}

#[test]
fn arming_with_no_recorded_tail_leaves_an_armed_guard_alone() {
    let guard = guard_with_tail(3);
    // A second compaction before any new call must not erase the tail.
    guard.arm();
    assert!(guard.record("search\u{1}b", "hits-b", true));
}

#[test]
fn reset_forgets_everything() {
    let guard = guard_with_tail(3);
    guard.reset();
    assert!(!guard.record("search\u{1}b", "hits-b", true));
}

#[test]
fn a_tail_that_was_not_repeating_is_not_remembered() {
    let guard = PostCompactionGuard::new(3);
    assert!(!guard.record("read\u{1}a", "doc-a", false));
    guard.arm();
    assert!(
        !guard.record("read\u{1}a", "doc-a", false),
        "one re-read of evicted content is correct behaviour"
    );
}

#[test]
fn a_repeating_pair_ages_out_while_unrelated_calls_run() {
    let guard = PostCompactionGuard::new(3);
    assert!(!guard.record("read\u{1}a", "doc-a", true));
    for n in 0..3 {
        assert!(!guard.record(&format!("other\u{1}{n}"), "x", false));
    }
    assert!(!guard.record("read\u{1}z", "doc-z", true));
    guard.arm();
    assert!(
        !guard.record("read\u{1}a", "doc-a", true),
        "the stale pair is outside the last window of calls"
    );
}
