use super::*;

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        ..Usage::default()
    }
}

fn detect(
    mode: ResponseOverflowDetection,
    usage: Usage,
    finish: Option<&str>,
    window: Option<u64>,
    max_tokens: Option<u32>,
) -> Option<OverflowInfo> {
    detect_response_overflow(mode, Some(&usage), finish, window, max_tokens)
}

#[test]
fn a_successful_response_whose_prompt_exceeds_the_window_is_a_silent_overflow() {
    let info = detect(
        ResponseOverflowDetection::Usage,
        usage(9_000, 100),
        Some("stop"),
        Some(8_192),
        None,
    )
    .expect("silent overflow");
    assert_eq!(info.requested, Some(9_000));
    assert_eq!(info.limit, Some(8_192));
}

#[test]
fn a_prompt_inside_the_window_is_not_an_overflow() {
    assert!(
        detect(
            ResponseOverflowDetection::UsageAndShortLength,
            usage(8_000, 100),
            Some("stop"),
            Some(8_192),
            Some(1_000),
        )
        .is_none()
    );
}

#[test]
fn without_a_known_window_nothing_is_inferred_from_usage() {
    assert!(
        detect(
            ResponseOverflowDetection::Usage,
            usage(9_000_000, 100),
            Some("stop"),
            None,
            None,
        )
        .is_none()
    );
}

#[test]
fn a_length_stop_with_no_output_and_a_full_window_is_an_overflow() {
    let info = detect(
        ResponseOverflowDetection::Usage,
        usage(8_150, 0),
        Some("length"),
        Some(8_192),
        Some(4_000),
    );
    assert!(info.is_some());
}

#[test]
fn a_length_stop_far_below_the_requested_cap_needs_the_opt_in() {
    let short = usage(5_000, 100);
    assert!(
        detect(
            ResponseOverflowDetection::Usage,
            short,
            Some("length"),
            Some(8_192),
            Some(4_000),
        )
        .is_none()
    );
    assert!(
        detect(
            ResponseOverflowDetection::UsageAndShortLength,
            short,
            Some("max_tokens"),
            Some(8_192),
            Some(4_000),
        )
        .is_some()
    );
}

#[test]
fn a_length_stop_that_used_most_of_the_cap_is_ordinary_truncation() {
    assert!(
        detect(
            ResponseOverflowDetection::UsageAndShortLength,
            usage(5_000, 3_900),
            Some("length"),
            Some(8_192),
            Some(4_000),
        )
        .is_none()
    );
}

#[test]
fn off_detects_nothing() {
    assert!(
        detect(
            ResponseOverflowDetection::Off,
            usage(9_000, 0),
            Some("length"),
            Some(8_192),
            Some(4_000),
        )
        .is_none()
    );
}

#[test]
fn a_response_without_usage_is_never_an_overflow() {
    assert!(
        detect_response_overflow(
            ResponseOverflowDetection::UsageAndShortLength,
            None,
            Some("length"),
            Some(8_192),
            Some(4_000),
        )
        .is_none()
    );
}
