//! Volatility-aware outcome fingerprinting.
//!
//! The no-progress trackers decide "did this call produce the same outcome
//! again?" by comparing outcomes. Compared byte for byte, a tool whose result
//! embeds a timestamp, request id or duration looks novel on every call, so a
//! genuine loop never trips a detector. An [`OutcomeFingerprinter`] reduces an
//! outcome to the identity the detectors compare, and the default
//! implementation, [`VolatileSpanNormalizer`], blanks the spans that change on
//! every invocation of an otherwise identical call.
//!
//! The normalizer is deliberately conservative: it only rewrites spans that are
//! unmistakably volatile (timestamps, clock times, epoch-like numbers,
//! durations, attempt and pid counters, UUIDs and long hex ids). Arbitrary
//! numbers (versions, line numbers, counts, ports) are left alone, because two
//! outcomes that differ in them genuinely differ.

use std::sync::LazyLock;

use regex::{Match, Regex};

/// Reduces a tool outcome (a successful output or a failure message) to the
/// identity the no-progress trackers compare.
///
/// Two outcomes that return the same fingerprint count as "the same outcome".
/// Implementations must be deterministic and cheap; the returned string is
/// hashed by the trackers and not retained.
pub trait OutcomeFingerprinter: Send + Sync {
    /// Returns the identity of `outcome`.
    fn fingerprint(&self, outcome: &str) -> String;
}

/// The default [`OutcomeFingerprinter`]: replaces volatile spans with stable
/// placeholders and keeps everything else verbatim. See [`normalize_volatile`].
#[derive(Debug, Default, Clone, Copy)]
pub struct VolatileSpanNormalizer;

impl OutcomeFingerprinter for VolatileSpanNormalizer {
    fn fingerprint(&self, outcome: &str) -> String {
        normalize_volatile(outcome)
    }
}

/// One volatile-span rule: a pattern, its placeholder, and an optional
/// predicate that rejects a match the pattern alone cannot exclude (the `regex`
/// crate has no look-around).
struct Rule {
    pattern: Regex,
    placeholder: &'static str,
    accept: fn(&str, &Match<'_>) -> bool,
}

fn accept_all(_: &str, _: &Match<'_>) -> bool {
    true
}

/// Rejects a match glued to a preceding `.`, e.g. the tail of `1.2.3s` or
/// `1.1759752896`, which is part of a dotted number rather than a standalone
/// value.
fn not_after_dot(text: &str, found: &Match<'_>) -> bool {
    !text[..found.start()].ends_with('.')
}

/// A hex id must mix digits and hex letters. That keeps long pure-decimal
/// numbers and letter-only words out; real hashes and ids contain both.
fn mixed_hex(_: &str, found: &Match<'_>) -> bool {
    let body = found.as_str();
    let body = body.strip_prefix("0x").unwrap_or(body);
    body.bytes().any(|b| b.is_ascii_digit()) && body.bytes().any(|b| b.is_ascii_alphabetic())
}

fn rule(pattern: &str, placeholder: &'static str, accept: fn(&str, &Match<'_>) -> bool) -> Rule {
    Rule {
        pattern: Regex::new(pattern).expect("volatile-span pattern is valid"),
        placeholder,
        accept,
    }
}

/// Ordered so a wider span is consumed before a narrower one it contains (an
/// ISO timestamp before its clock time, a UUID before its hex groups).
static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    vec![
        // ISO-8601 / RFC 3339 timestamp, `T` or space separated, with optional
        // seconds, fraction and zone.
        rule(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?",
            "<timestamp>",
            accept_all,
        ),
        rule(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "<uuid>",
            accept_all,
        ),
        rule(r"\b(?:0x)?[0-9a-fA-F]{16,}\b", "<hex-id>", mixed_hex),
        rule(
            r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?\b",
            "<time>",
            accept_all,
        ),
        // Unix epoch seconds (10 digits) or milliseconds (13 digits) in the
        // 2001-2033 range, optionally with a fraction.
        rule(
            r"\b1\d{9}(?:\d{3})?(?:\.\d+)?\b",
            "<epoch>",
            not_after_dot,
        ),
        // `123ms`, `1.2s`, `250us`, and Go-style `1h2m3.5s`. The unit must be
        // attached to the number, so prose like "2 minutes ago" is untouched.
        rule(
            r"\b(?:\d+h)?(?:\d+m)?\d+(?:\.\d+)?(?:ns|µs|us|ms|s)\b",
            "<duration>",
            not_after_dot,
        ),
        rule(
            r"(?i)\b(?:attempt|retry)\s*#?\s*\d+(?:\s*(?:of|/)\s*\d+)?\b",
            "<attempt>",
            accept_all,
        ),
        rule(r"(?i)\bpid(?:\s*[=:]\s*|\s+)\d+\b", "<pid>", accept_all),
    ]
});

/// Replaces the volatile spans of `text` with stable placeholders.
///
/// Rewritten: ISO-8601 / RFC 3339 timestamps, `HH:MM:SS(.fff)` clock times,
/// 10- and 13-digit unix epochs, durations attached to their unit (`123ms`,
/// `1.2s`), `attempt N` / `retry N of M` counters, `pid N`, UUIDs and hex ids
/// of at least 16 characters. Everything else, including every other number, is
/// kept verbatim.
pub fn normalize_volatile(text: &str) -> String {
    let mut current = text.to_string();
    for rule in RULES.iter() {
        let mut out = String::with_capacity(current.len());
        let mut last = 0;
        for found in rule.pattern.find_iter(&current) {
            if !(rule.accept)(&current, &found) {
                continue;
            }
            out.push_str(&current[last..found.start()]);
            out.push_str(rule.placeholder);
            last = found.end();
        }
        out.push_str(&current[last..]);
        current = out;
    }
    current
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
