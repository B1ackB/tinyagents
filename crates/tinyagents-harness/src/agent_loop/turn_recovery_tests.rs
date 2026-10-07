use super::*;

fn spent() -> TurnRecovery {
    TurnRecovery {
        truncated_empty_retries_used: 2,
        truncated_empty_nudges_used: 1,
        empty_response_retries_used: 3,
        dropped_tool_call_nudges_used: 2,
        withheld_call_nudges_used: 1,
        truncated_tool_call_retries_used: 2,
        boosted_max_tokens: Some(4096),
        truncation_base: Some(1024),
    }
}

#[test]
fn boost_doubles_the_last_sent_cap_and_clamps_at_four_times_the_base() {
    let mut recovery = TurnRecovery::default();
    recovery.boost_max_tokens(Some(1000));
    assert_eq!(recovery.boosted_max_tokens, Some(2000));
    assert_eq!(recovery.truncation_base, Some(1000));
    recovery.boost_max_tokens(Some(2000));
    assert_eq!(recovery.boosted_max_tokens, Some(4000));
    recovery.boost_max_tokens(Some(4000));
    assert_eq!(recovery.boosted_max_tokens, Some(4000), "clamped at 4x base");
    assert_eq!(recovery.truncation_base, Some(1000));
}

#[test]
fn boost_leaves_an_unset_cap_unset() {
    let mut recovery = TurnRecovery::default();
    recovery.boost_max_tokens(None);
    assert_eq!(recovery, TurnRecovery::default());
}

#[test]
fn reset_truncated_empty_clears_only_the_truncated_empty_state() {
    let mut recovery = spent();
    recovery.reset_truncated_empty();
    assert_eq!(recovery.truncated_empty_retries_used, 0);
    assert_eq!(recovery.truncated_empty_nudges_used, 0);
    assert_eq!(recovery.boosted_max_tokens, None);
    assert_eq!(recovery.truncation_base, None);
    // Untouched.
    assert_eq!(recovery.empty_response_retries_used, 3);
    assert_eq!(recovery.dropped_tool_call_nudges_used, 2);
    assert_eq!(recovery.withheld_call_nudges_used, 1);
    assert_eq!(recovery.truncated_tool_call_retries_used, 2);
}

#[test]
fn reset_after_clean_tool_turn_clears_everything() {
    let mut recovery = spent();
    recovery.reset_after_clean_tool_turn(false);
    assert_eq!(recovery, TurnRecovery::default());
}

#[test]
fn reset_after_a_truncated_tool_turn_keeps_the_retry_budget_and_boost() {
    let mut recovery = spent();
    recovery.reset_after_clean_tool_turn(true);
    // The re-prompt counters clear...
    assert_eq!(recovery.dropped_tool_call_nudges_used, 0);
    assert_eq!(recovery.withheld_call_nudges_used, 0);
    assert_eq!(recovery.empty_response_retries_used, 0);
    // ...but the truncated turn keeps what its retry needs.
    assert_eq!(recovery.truncated_tool_call_retries_used, 2);
    assert_eq!(recovery.truncated_empty_retries_used, 2);
    assert_eq!(recovery.truncated_empty_nudges_used, 1);
    assert_eq!(recovery.boosted_max_tokens, Some(4096));
    assert_eq!(recovery.truncation_base, Some(1024));
}

#[test]
fn reset_after_final_clears_everything() {
    let mut recovery = spent();
    recovery.reset_after_final();
    assert_eq!(recovery, TurnRecovery::default());
}
