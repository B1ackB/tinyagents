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
    pressure.note_request(&[Message::user("x"), Message::assistant("y")], 10);
    pressure.observe(Some(&usage(5_000)), &policy(100_000), 2, 10);

    let appended = Message::user("b".repeat(400));
    let messages = vec![
        Message::user("x"),
        Message::assistant("y"),
        appended.clone(),
    ];
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
fn measured_usage_never_understates_the_full_request_estimate() {
    let mut pressure = CompactionPressure::default();
    let messages = vec![Message::user("large".repeat(1_000))];
    pressure.begin_call();
    pressure.note_request(&messages, 0);
    pressure.observe(Some(&usage(2)), &policy(100_000), 2, 10);
    let (tokens, source) = pressure.prompt_tokens(&messages, 0);
    assert_eq!(source, PromptSource::Measured);
    assert_eq!(
        tokens,
        crate::token_estimation::estimate_slice_tokens(&messages)
    );
}

#[test]
fn two_ineffective_compactions_suppress_for_the_cooldown() {
    let policy = policy(1_000);
    let mut pressure = CompactionPressure::default();
    for strike in 1..=2 {
        assert!(!pressure.begin_call());
        pressure.note_compaction();
        pressure.note_request(&vec![Message::user("x"); 3], 0);
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
    pressure.note_request(&vec![Message::user("x"); 3], 0);
    pressure.observe(Some(&usage(2_000)), &policy, 2, 10);
    assert_eq!(pressure.strikes, 1);

    pressure.begin_call();
    pressure.note_compaction();
    pressure.note_request(&vec![Message::user("x"); 3], 0);
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
    pressure.note_request(&[Message::user("x")], 0);
    assert!(!pressure.observe(None, &policy, 2, 10));
    assert!(pressure.measured.is_none());
}
