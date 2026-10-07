//! Provider finish-reason helpers shared across the harness.

/// Whether a provider's finish reason says the output cap cut the reply off.
///
/// Providers disagree on the spelling — OpenAI-compatible endpoints report
/// `length`, Anthropic `max_tokens`, some gateways pass `MAX_TOKENS` through —
/// and `tinyinference` does not normalise it, so the harness matches the known
/// spellings itself. Normalisation belongs upstream in `tinyinference`; once it
/// lands this helper collapses to the single canonical value.
pub(crate) fn is_length_stop(finish_reason: Option<&str>) -> bool {
    matches!(finish_reason, Some("length" | "max_tokens" | "MAX_TOKENS"))
}

#[cfg(test)]
#[path = "finish_reason_tests.rs"]
mod tests;
