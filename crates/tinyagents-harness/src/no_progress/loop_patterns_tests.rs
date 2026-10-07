//! Tests for the warning-only ping-pong and argument-churn detectors.

use super::*;

fn feed_alternation(detector: &PingPongDetector, rounds: usize) -> Vec<Option<String>> {
    (0..rounds)
        .flat_map(|_| {
            [
                detector.record("read\u{1}a", "doc-a"),
                detector.record("search\u{1}b", "hits-b"),
            ]
        })
        .collect()
}

#[test]
fn ping_pong_warns_once_after_the_configured_alternations() {
    let detector = PingPongDetector::new(6);
    let verdicts = feed_alternation(&detector, 5);
    // Calls 1-5 are quiet; the 6th alternation (A,B,A,B,A,B) warns.
    assert!(verdicts[..5].iter().all(Option::is_none));
    assert!(
        verdicts[5]
            .as_deref()
            .is_some_and(|note| note.contains("alternating")),
        "the sixth alternation warns: {:?}",
        verdicts[5]
    );
    assert!(
        verdicts[6..].iter().all(Option::is_none),
        "the same pair is warned about once"
    );
}

#[test]
fn ping_pong_ignores_alternation_whose_outcome_changes() {
    let detector = PingPongDetector::new(6);
    for i in 0..10 {
        assert!(detector.record("read\u{1}a", &format!("doc-{i}")).is_none());
        assert!(detector.record("search\u{1}b", "hits").is_none());
    }
}

#[test]
fn ping_pong_ignores_a_single_call_repeated_and_a_broken_rhythm() {
    let detector = PingPongDetector::new(6);
    for _ in 0..10 {
        assert!(detector.record("read\u{1}a", "doc").is_none());
    }
    let detector = PingPongDetector::new(6);
    for _ in 0..4 {
        detector.record("read\u{1}a", "doc");
        detector.record("search\u{1}b", "hits");
    }
    // A third call interrupts the pair; the count starts over.
    assert!(detector.record("write\u{1}c", "ok").is_none());
    assert!(feed_alternation(&detector, 2).iter().all(Option::is_none));
}

#[test]
fn ping_pong_reset_forgets_the_tail_and_the_warned_pair() {
    let detector = PingPongDetector::new(4);
    assert!(feed_alternation(&detector, 2).iter().any(Option::is_some));
    detector.reset();
    assert!(feed_alternation(&detector, 2).iter().any(Option::is_some));
}

#[test]
fn argument_churn_warns_when_many_variants_each_repeat_one_outcome() {
    let detector = ArgumentChurnDetector::new(3, 3);
    let mut warnings = Vec::new();
    for variant in ["a", "b", "c"] {
        for _ in 0..3 {
            warnings.push(detector.record("search", variant, "no results"));
        }
    }
    assert!(warnings[..8].iter().all(Option::is_none));
    assert!(
        warnings[8]
            .as_deref()
            .is_some_and(|note| note.contains("search")),
        "the third variant's third call completes the pattern: {:?}",
        warnings[8]
    );
    assert!(
        detector.record("search", "c", "no results").is_none(),
        "warned once per tool and outcome"
    );
}

#[test]
fn argument_churn_needs_the_same_outcome_and_enough_calls_per_variant() {
    let detector = ArgumentChurnDetector::new(3, 3);
    // Three variants but only two calls each.
    for variant in ["a", "b", "c"] {
        for _ in 0..2 {
            assert!(detector.record("search", variant, "none").is_none());
        }
    }
    // Plenty of calls, but each variant gets its own outcome.
    for variant in ["a", "b", "c"] {
        for _ in 0..5 {
            assert!(
                detector
                    .record("fetch", variant, &format!("body-{variant}"))
                    .is_none()
            );
        }
    }
}

#[test]
fn argument_churn_is_scoped_to_one_tool() {
    let detector = ArgumentChurnDetector::new(3, 3);
    for (tool, variant) in [("t1", "a"), ("t2", "b"), ("t3", "c")] {
        for _ in 0..3 {
            assert!(detector.record(tool, variant, "same").is_none());
        }
    }
}
