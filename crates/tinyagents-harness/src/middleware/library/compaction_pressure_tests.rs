use super::*;

fn policy(trigger: u64) -> SummarizationPolicy {
    SummarizationPolicy::default().with_trigger_override(trigger)
}

fn usage(input_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        ..Usage::default()
    }
}

#[test]
fn falls_back_to_the_estimate_without_usage() {
    let pressure = CompactionPressure::default();
    let messages = vec![Message::user("a".repeat(400))];
    let (tokens, source) = pressure.prompt_tokens(&messages, 7);
    assert_eq!(source, PromptSource::Estimated);
    assert_eq!(
        tokens,
        crate::token_estimation::estimate_slice_tokens(&messages) + 7
    );
}

#[test]
fn measured_prompt_adds_only_the_appended_messages_and_schema_growth() {
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    let sent = vec![Message::user("x"), Message::assistant("y")];
    pressure.note_request(&sent, 10);
    pressure.observe(Some(&usage(5_000)), &policy(100_000), 2, 10);

    let appended = Message::user("b".repeat(400));
    let messages = vec![sent[0].clone(), sent[1].clone(), appended.clone()];
    let (tokens, source) = pressure.prompt_tokens(&messages, 25);
    assert_eq!(source, PromptSource::Measured);
    assert_eq!(
        tokens,
        5_000 + crate::token_estimation::estimate_slice_tokens(&[appended]) + 15
    );
}

#[test]
fn a_shorter_request_than_the_measured_one_falls_back() {
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    pressure.note_request(&vec![Message::user("x"); 5], 0);
    pressure.observe(Some(&usage(5_000)), &policy(100_000), 2, 10);
    let (_, source) = pressure.prompt_tokens(&[Message::user("x")], 0);
    assert_eq!(source, PromptSource::Estimated);
}

#[test]
fn a_longer_request_with_a_different_prefix_falls_back() {
    // The measured request was front-trimmed (anti-thrash); the next request
    // is the untrimmed transcript, longer but not an extension of it.
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    pressure.note_request(&[Message::user("tail-1"), Message::user("tail-2")], 0);
    pressure.observe(Some(&usage(5_000)), &policy(100_000), 2, 10);
    let untrimmed = vec![
        Message::user("head"),
        Message::user("tail-1"),
        Message::user("tail-2"),
    ];
    let (tokens, source) = pressure.prompt_tokens(&untrimmed, 0);
    assert_eq!(source, PromptSource::Estimated);
    assert_eq!(
        tokens,
        crate::token_estimation::estimate_slice_tokens(&untrimmed)
    );
}

#[test]
fn two_ineffective_compactions_suppress_for_the_cooldown() {
    let policy = policy(1_000);
    let mut pressure = CompactionPressure::default();
    for strike in 1..=2 {
        assert!(!pressure.begin_call());
        pressure.note_compaction();
        pressure.note_request(&vec![Message::user("a"); 3], 0);
        let engaged = pressure.observe(Some(&usage(2_000)), &policy, 2, 3);
        assert_eq!(engaged, strike == 2);
    }
    assert!(pressure.begin_call());
    assert!(pressure.begin_call());
    assert!(pressure.begin_call());
    assert!(!pressure.begin_call(), "the cooldown is three calls");
}

#[test]
fn an_effective_compaction_resets_the_strikes() {
    let policy = policy(1_000);
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    pressure.note_compaction();
    pressure.note_request(&vec![Message::user("a"); 3], 0);
    pressure.observe(Some(&usage(2_000)), &policy, 2, 10);
    assert_eq!(pressure.strikes, 1);

    pressure.begin_call();
    pressure.note_compaction();
    pressure.note_request(&vec![Message::user("a"); 3], 0);
    pressure.observe(Some(&usage(500)), &policy, 2, 10);
    assert_eq!(pressure.strikes, 0);
    assert!(!pressure.begin_call());
}

#[test]
fn usage_without_a_pending_request_is_ignored() {
    let policy = policy(1_000);
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    assert!(!pressure.observe(Some(&usage(2_000)), &policy, 2, 10));
    assert!(pressure.measured.is_none());
    pressure.note_request(&[Message::user("a")], 0);
    assert!(!pressure.observe(None, &policy, 2, 10));
    assert!(pressure.measured.is_none());
}

#[test]
fn shrunken_tool_schemas_fall_back_to_the_estimate() {
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    let sent = vec![Message::user("x")];
    pressure.note_request(&sent, 100);
    pressure.observe(Some(&usage(5_000)), &policy(100_000), 2, 10);
    let (tokens, source) = pressure.prompt_tokens(&sent, 10);
    assert_eq!(source, PromptSource::Estimated);
    assert_eq!(
        tokens,
        crate::token_estimation::estimate_slice_tokens(&sent) + 10
    );
}

#[test]
fn changed_prefix_uses_full_request_estimate() {
    let mut pressure = CompactionPressure::default();
    pressure.begin_call();
    pressure.note_request(&[Message::user("small")], 0);
    pressure.observe(Some(&usage(2)), &policy(100_000), 2, 10);
    let messages = vec![Message::user("large".repeat(1_000))];
    let (tokens, source) = pressure.prompt_tokens(&messages, 0);
    assert_eq!(source, PromptSource::Estimated);
    assert_eq!(
        tokens,
        crate::token_estimation::estimate_slice_tokens(&messages)
    );
}

#[test]
fn matching_measured_usage_uses_provider_count() {
    let mut pressure = CompactionPressure::default();
    let messages = vec![Message::user("large".repeat(1_000))];
    pressure.begin_call();
    pressure.note_request(&messages, 0);
    pressure.observe(Some(&usage(2)), &policy(100_000), 2, 10);
    let (tokens, source) = pressure.prompt_tokens(&messages, 0);
    assert_eq!(source, PromptSource::Measured);
    assert_eq!(tokens, 2);
}

// ── preemptive route decision ─────────────────────────────────────────────────

#[test]
fn a_prompt_inside_the_budget_fits() {
    assert_eq!(
        CompactionPressure::route(900, 1_000, 5_000),
        CompactionRoute::Fits
    );
    assert_eq!(
        CompactionPressure::route(1_000, 1_000, 0),
        CompactionRoute::Fits,
        "exactly at the budget still fits"
    );
}

#[test]
fn with_nothing_to_truncate_an_overflow_compacts() {
    assert_eq!(
        CompactionPressure::route(2_000, 1_000, 0),
        CompactionRoute::Compact
    );
}

#[test]
fn truncation_alone_is_chosen_only_when_it_comfortably_covers_the_overflow() {
    // Overflow is 1_000; the bar is max(1_000 + 512, 1_500) = 1_512.
    assert_eq!(
        CompactionPressure::route(2_000, 1_000, 1_512),
        CompactionRoute::TruncateToolResults
    );
    assert_eq!(
        CompactionPressure::route(2_000, 1_000, 1_511),
        CompactionRoute::CompactThenTruncate
    );
    // A large overflow is judged by the 1.5x margin, not the flat buffer.
    assert_eq!(
        CompactionPressure::route(21_000, 1_000, 29_999),
        CompactionRoute::CompactThenTruncate
    );
    assert_eq!(
        CompactionPressure::route(21_000, 1_000, 30_000),
        CompactionRoute::TruncateToolResults
    );
}

#[test]
fn route_flags_say_which_steps_run() {
    assert!(!CompactionRoute::Fits.compacts() && !CompactionRoute::Fits.truncates());
    assert!(
        CompactionRoute::TruncateToolResults.truncates()
            && !CompactionRoute::TruncateToolResults.compacts()
    );
    assert!(CompactionRoute::Compact.compacts() && !CompactionRoute::Compact.truncates());
    assert!(
        CompactionRoute::CompactThenTruncate.compacts()
            && CompactionRoute::CompactThenTruncate.truncates()
    );
}
