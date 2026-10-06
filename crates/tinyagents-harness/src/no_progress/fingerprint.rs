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

use std::borrow::Cow;
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
/// crate has no look-around). The replaced span is the pattern's `span` group
/// when it has one (the rest of the match is only context), else the whole
/// match.
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

// Every `\b` below is the ASCII form `(?-u:\b)`: the Unicode word boundary
// pushes the regex engine onto its slow path for non-ASCII input.

/// Ordered so a wider span is consumed before a narrower one it contains (an
/// ISO timestamp before its clock time, a UUID before its hex groups).
static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    vec![
        // ISO-8601 / RFC 3339 timestamp, `T` or space separated, with optional
        // seconds, fraction and zone.
        rule(
            r"(?-u:\b)\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?",
            "<timestamp>",
            accept_all,
        ),
        rule(
            r"(?-u:\b)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?-u:\b)",
            "<uuid>",
            accept_all,
        ),
        rule(
            r"(?-u:\b)(?:0x)?[0-9a-fA-F]{16,}(?-u:\b)",
            "<hex-id>",
            mixed_hex,
        ),
        rule(
            r"(?-u:\b)\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?-u:\b)",
            "<time>",
            accept_all,
        ),
        // Unix epoch seconds (10 digits) or milliseconds (13 digits) in the
        // 2001-2033 range, optionally with a fraction, and only as the value
        // of a time-like key (`ts=`, `"timestamp":`, `updated_at:`,
        // `createdAt=`, `mtime=`, `epoch=`). A bare number of that size is as
        // likely a byte count, an order number or a polled clock value, which
        // are content, so it is left alone.
        rule(
            r#"(?-u:\b)(?:(?i:ts|time|timestamp|epoch|mtime|ctime|atime)|[A-Za-z0-9]*(?:[_-](?i:ts|time|timestamp|epoch|at)|At|Time|Ts|Timestamp))["']?\s*[=:]\s*["']?(?P<span>1\d{9}(?:\d{3})?(?:\.\d+)?)(?-u:\b)"#,
            "<epoch>",
            not_after_dot,
        ),
        // `123ms`, `250us`, `1.2s`, `45s`, and Go-style `1h2m3.5s`. The unit
        // must be attached to the number. A bare-seconds integer is limited to
        // two digits so decades and rough counts ("the 1990s", "100s of
        // files") are untouched.
        rule(
            r"(?-u:\b)(?:(?:\d+h)?(?:\d+m)?(?:\d{1,2}|\d+\.\d+)s|\d+(?:\.\d+)?(?:ns|[µu]s|ms))(?-u:\b)",
            "<duration>",
            not_after_dot,
        ),
        rule(
            r"(?i)(?-u:\b)(?:attempt|retry)\s*#?\s*\d+(?:\s*(?:of|/)\s*\d+)?(?-u:\b)",
            "<attempt>",
            accept_all,
        ),
        rule(
            r"(?i)(?-u:\b)pid(?:\s*[=:]\s*|\s+)\d+(?-u:\b)",
            "<pid>",
            accept_all,
        ),
    ]
});

/// Fewer alphanumeric characters than this outside the volatile spans means
/// the outcome *is* its volatile content (a bare commit id, a checksum, a
/// timestamp), so [`normalize_volatile`] keeps it verbatim.
const MIN_RESIDUE_CHARS: usize = 4;

/// Replaces the volatile spans of `text` with stable placeholders.
///
/// Rewritten: ISO-8601 / RFC 3339 timestamps, `HH:MM:SS(.fff)` clock times,
/// 10- and 13-digit unix epochs that are the value of a time-like key (`ts=`,
/// `"timestamp":`, `updated_at:`), durations attached to their unit (`123ms`,
/// `1.2s`), `attempt N` / `retry N of M` counters, `pid N`, UUIDs and hex ids
/// of at least 16 characters. Everything else, including every other number, is
/// kept verbatim.
///
/// An outcome whose value is nothing but a volatile span (the stdout of
/// `git rev-parse HEAD`, a checksum, a bare timestamp) is returned unchanged:
/// when fewer than four alphanumeric characters remain outside the spans, the
/// spans are the content rather than noise around it, and two such outcomes
/// that differ are different outcomes.
pub fn normalize_volatile(text: &str) -> String {
    let mut current: Cow<'_, str> = Cow::Borrowed(text);
    for rule in RULES.iter() {
        if let Some(rewritten) = rewrite(rule, &current) {
            current = Cow::Owned(rewritten);
        }
    }
    if matches!(current, Cow::Borrowed(_)) || residue_chars(&current) < MIN_RESIDUE_CHARS {
        return text.to_string();
    }
    current.into_owned()
}

/// `text` with every accepted match of `rule` replaced, or `None` when nothing
/// matched (so the common no-match case allocates nothing).
fn rewrite(rule: &Rule, text: &str) -> Option<String> {
    let mut out: Option<String> = None;
    let mut last = 0;
    for captures in rule.pattern.captures_iter(text) {
        let whole = captures.get(0).expect("group 0 always participates");
        let span = captures.name("span").unwrap_or(whole);
        if !(rule.accept)(text, &span) {
            continue;
        }
        let out = out.get_or_insert_with(|| String::with_capacity(text.len()));
        out.push_str(&text[last..span.start()]);
        out.push_str(rule.placeholder);
        last = span.end();
    }
    out.map(|mut out| {
        out.push_str(&text[last..]);
        out
    })
}

/// Alphanumeric characters of `normalized` that are not placeholders.
fn residue_chars(normalized: &str) -> usize {
    let mut residue = normalized.to_string();
    for rule in RULES.iter() {
        residue = residue.replace(rule.placeholder, "");
    }
    residue.chars().filter(|c| c.is_alphanumeric()).count()
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
