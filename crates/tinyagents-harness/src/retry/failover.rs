//! Reason-aware failover: *why* a model call failed decides what happens next.
//!
//! Retrying and falling back used to be driven by one boolean
//! ([`is_retryable`]) and applied to every error alike: a permanent `401`
//! walked the whole fallback chain with a guaranteed-identical failure per
//! hop, while a malformed request was replayed against every model on the
//! chain. This module separates the two questions:
//!
//! 1. [`FailoverReason::classify`] names the cause, reusing
//!    [`classify_provider_failure`] and the provider-body matchers from
//!    [`tinyinference_llm::failure`] rather than inventing new heuristics.
//! 2. [`decide`] is a pure function from `(reason, state)` to a
//!    [`FailoverDecision`] — [`RetrySame`](FailoverDecision::RetrySame),
//!    [`Fallback`](FailoverDecision::Fallback) or
//!    [`Surface`](FailoverDecision::Surface).
//!
//! The agent loop's model call consults [`decide`] on every failed attempt.
//!
//! # Decision table
//!
//! | Reason | Decision |
//! |---|---|
//! | `RateLimit`, `Overloaded`, `Timeout`, `Transport`, `EmptyResponse`, `Unknown` | `RetrySame` while the retry policy calls the error transient **and** attempts remain; then `Fallback` |
//! | `Auth`, `Billing`, `ModelNotFound` | `Fallback` immediately (a retry cannot change the answer) |
//! | `AuthPermanent` | `Fallback` immediately, and the model is skipped for the rest of the run ([`FailoverReason::skips_model_for_run`]) |
//! | `Format` | `Surface` (another model will not fix a malformed request) — unless the failure is model-specific capability ([`is_model_specific_format`]), then `Fallback` |
//! | `ContextOverflow` | `Surface` (compaction, not a different model, is the remedy) |
//!
//! A custom [`RetryPolicy::retry_on`] predicate keeps authority over the
//! *transient* reasons (it can veto a retry), but it cannot turn a permanent
//! reason such as `Auth` or `Format` back into a same-model retry.

use super::RetryPolicy;
use crate::error::TinyAgentsError;
use tinyinference_llm::failure::{
    ProviderFailureClass, body_indicates_auth_key_error, body_indicates_insufficient_credits,
    body_indicates_no_model_loaded, body_indicates_provider_access_policy_denied,
    body_indicates_quota_exhausted, classify_provider_failure, is_context_window_exceeded_message,
    structured_http_status,
};

/// Why a model call failed, as far as failover policy is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FailoverReason {
    /// Credentials were rejected (`401`/`403`, invalid key). May be fixed by a
    /// credential refresh, so it is not remembered across calls.
    Auth,
    /// Credentials are permanently unusable (revoked, deactivated, suspended).
    AuthPermanent,
    /// Billing or quota is exhausted (`402`, no credits, plan quota).
    Billing,
    /// A transient throttle (`429`).
    RateLimit,
    /// The provider is at capacity (`503`, `529`, "overloaded").
    Overloaded,
    /// The call or the provider timed out.
    Timeout,
    /// The request was rejected as malformed (`400`/`422`, schema errors).
    Format,
    /// The request did not fit the model's context window.
    ContextOverflow,
    /// The model does not exist (or is not loaded) at the provider.
    ModelNotFound,
    /// The model answered with nothing usable.
    EmptyResponse,
    /// Network failure or a `5xx` without a more specific cause.
    Transport,
    /// Not attributable to any of the above.
    Unknown,
}

/// What the loop should do after a failed model attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailoverDecision {
    /// Try the same model again (after backoff).
    RetrySame,
    /// Give up on this model and walk the fallback chain.
    Fallback,
    /// Fail the call now; no other model on the chain would help.
    Surface,
}

/// The per-attempt facts [`decide`] combines with the [`FailoverReason`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailoverState {
    /// The retry policy ([`RetryPolicy::is_retryable_error`], honouring a
    /// custom `retry_on`) considers the error transient.
    pub retryable: bool,
    /// The retry policy still permits another attempt on this model.
    pub attempts_remaining: bool,
    /// The failure is a capability/parameter the *model* lacks
    /// ([`is_model_specific_format`]), so a different model may succeed.
    pub model_specific: bool,
}

impl FailoverState {
    /// Builds the state for `error` on the zero-indexed `attempt` under
    /// `policy`. Pass the policy already capped by the harness's
    /// `max_retries_per_call`.
    pub fn for_error(policy: &RetryPolicy, attempt: usize, error: &TinyAgentsError) -> Self {
        Self {
            retryable: policy.is_retryable_error(error),
            attempts_remaining: policy.should_retry(attempt),
            model_specific: is_model_specific_format(error),
        }
    }
}

impl FailoverReason {
    /// Stable snake_case label for logs and telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::AuthPermanent => "auth_permanent",
            Self::Billing => "billing",
            Self::RateLimit => "rate_limit",
            Self::Overloaded => "overloaded",
            Self::Timeout => "timeout",
            Self::Format => "format",
            Self::ContextOverflow => "context_overflow",
            Self::ModelNotFound => "model_not_found",
            Self::EmptyResponse => "empty_response",
            Self::Transport => "transport",
            Self::Unknown => "unknown",
        }
    }

    /// `true` when a model that failed for this reason should be skipped for
    /// the remainder of the run (the cross-call skip hint). Only
    /// [`FailoverReason::AuthPermanent`] qualifies: every other reason can
    /// clear on its own (rate limits, outages) or be fixed by the host
    /// (a refreshed token, a topped-up balance).
    pub fn skips_model_for_run(self) -> bool {
        matches!(self, Self::AuthPermanent)
    }

    /// Classifies `error`.
    pub fn classify(error: &TinyAgentsError) -> Self {
        match error {
            TinyAgentsError::Provider(provider) => classify_text(
                provider.status,
                provider.code.as_deref(),
                &provider.message,
                provider.retryable,
            ),
            TinyAgentsError::Model(message) => classify_text(None, None, message, true),
            TinyAgentsError::ContextOverflow { .. } => Self::ContextOverflow,
            TinyAgentsError::ModelNotFound(_) => Self::ModelNotFound,
            TinyAgentsError::EmptyResponse => Self::EmptyResponse,
            TinyAgentsError::CallTimeout(_) | TinyAgentsError::Timeout(_) => Self::Timeout,
            TinyAgentsError::Validation(_) => Self::Format,
            _ => Self::Unknown,
        }
    }
}

/// The failover table. Pure; see the [module docs](self) for the rationale.
pub fn decide(reason: FailoverReason, state: FailoverState) -> FailoverDecision {
    use FailoverReason::*;
    match reason {
        RateLimit | Overloaded | Timeout | Transport | EmptyResponse | Unknown => {
            if state.retryable && state.attempts_remaining {
                FailoverDecision::RetrySame
            } else {
                FailoverDecision::Fallback
            }
        }
        Auth | AuthPermanent | Billing | ModelNotFound => FailoverDecision::Fallback,
        Format if state.model_specific => FailoverDecision::Fallback,
        Format | ContextOverflow => FailoverDecision::Surface,
    }
}

/// `true` when `error` is a request rejection caused by something *this model*
/// lacks (tool calling, image input, a sampling parameter) rather than by the
/// request being malformed for every model. Only such a [`FailoverReason::Format`]
/// failure is worth a fallback.
pub fn is_model_specific_format(error: &TinyAgentsError) -> bool {
    let message = match error {
        TinyAgentsError::Provider(provider) => provider.message.as_str(),
        TinyAgentsError::Model(message) => message.as_str(),
        _ => return false,
    };
    let lower = message.to_ascii_lowercase();
    [
        "does not support",
        "doesn't support",
        "not supported",
        "unsupported parameter",
        "unsupported_parameter",
        "unsupported value",
        "unsupported_value",
        "unsupported content",
        "unknown parameter",
        "unrecognized request argument",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

/// Markers of a credential that will not recover on its own.
const PERMANENT_AUTH_MARKERS: &[&str] = &[
    "revoked",
    "deactivated",
    "disabled",
    "suspended",
    "terminated",
    "banned",
    "permanently",
];

fn classify_text(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
    provider_retryable: bool,
) -> FailoverReason {
    use FailoverReason::*;
    let lower = match code {
        Some(code) if !code.trim().is_empty() => format!("{message} {code}").to_ascii_lowercase(),
        _ => message.to_ascii_lowercase(),
    };
    let status = status.or_else(|| structured_http_status(message));

    if is_context_window_exceeded_message(message) || lower.contains("context_length_exceeded") {
        return ContextOverflow;
    }

    let failure = classify_provider_failure(status, code, message);
    if status == Some(402)
        || failure == ProviderFailureClass::NonRetryableRateLimit
        || body_indicates_insufficient_credits(message)
        || body_indicates_quota_exhausted(message)
        || lower.contains("insufficient_quota")
        || lower.contains("current quota")
    {
        return Billing;
    }

    let key_error = body_indicates_auth_key_error(message)
        || body_indicates_provider_access_policy_denied(message);
    if matches!(status, Some(401 | 403)) || key_error {
        return if PERMANENT_AUTH_MARKERS.iter().any(|m| lower.contains(m)) {
            AuthPermanent
        } else {
            Auth
        };
    }

    if lower.contains("model_not_found")
        || lower.contains("model not found")
        || body_indicates_no_model_loaded(message)
        || (lower.contains("model")
            && (lower.contains("does not exist") || lower.contains("unknown model"))
            && !matches!(status, Some(429 | 500..=599)))
        || (status == Some(404) && lower.contains("model"))
    {
        return ModelNotFound;
    }

    if status == Some(429) || failure == ProviderFailureClass::RateLimited {
        return RateLimit;
    }
    if status == Some(408) || lower.contains("timed out") || lower.contains("timeout") {
        return Timeout;
    }
    if matches!(status, Some(503 | 529))
        || lower.contains("overloaded")
        || lower.contains("over capacity")
    {
        return Overloaded;
    }
    match status {
        Some(409) | Some(500..=599) => return Transport,
        Some(400..=499) => return Format,
        _ => {}
    }

    match failure {
        ProviderFailureClass::UpstreamUnhealthy => Transport,
        ProviderFailureClass::NonRetryable if !provider_retryable || status.is_none() => Format,
        _ => Transport,
    }
}

#[cfg(test)]
#[path = "failover_tests.rs"]
mod test;
