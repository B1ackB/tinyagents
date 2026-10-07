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
//! durations, attempt and pid counters, UUIDs). Arbitrary
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

/// Keeps UUIDs that identify returned entities, such as `event_id`, in the
/// outcome identity. Request and correlation identifiers remain volatile.
fn uuid_context(text: &str, found: &Match<'_>) -> bool {
    let field_prefix = field_prefix(text, found);
    !field_prefix.ends_with("event_id\":")
        && !field_prefix.ends_with("eventid\":")
        && !field_prefix.ends_with("event_id:")
        && !field_prefix.ends_with("eventid:")
}

/// Rejects a match glued to a preceding `.`, e.g. the tail of `1.2.3s` or
/// `1.1759752896`, which is part of a dotted number rather than a standalone
/// value.
fn not_after_dot(text: &str, found: &Match<'_>) -> bool {
    !text[..found.start()].ends_with('.')
}

/// Accepts duration phrases that describe elapsed work, while keeping
/// semantic countdowns such as `lease expires in 30s` as content.
fn duration_context(text: &str, found: &Match<'_>) -> bool {
    let before = text[..found.start()].to_ascii_lowercase();
    let keyword_context = before.trim_end_matches(|character: char| {
        character.is_ascii_whitespace() || matches!(character, '"' | '\'' | ':' | '=')
    });
    if [
        "took", "elapsed", "duration", "latency", "timeout", "wait", "waited", "sleep", "sleeping",
    ]
    .iter()
    .any(|keyword| keyword_context.ends_with(keyword))
    {
        return true;
    }
    if keyword_context.ends_with("after") {
        return true;
    }
    keyword_context.ends_with("in") && {
        let preceding = &keyword_context[..keyword_context.len() - "in".len()];
        preceding.contains("<timestamp>")
            || (preceding.contains(|character: char| character.is_ascii_digit())
                && preceding.contains('-'))
    }
}

/// The span must not run straight into a word character: `2026-10-06T12:34:56Zebra`
/// is not a timestamp followed by text, and the regex crate has no
/// look-ahead to say so in the pattern.
fn not_followed_by_word(text: &str, found: &Match<'_>) -> bool {
    !text[found.end()..]
        .chars()
        .next()
        .is_some_and(|c| c.is_alphanumeric() || c == '_')
}

fn field_prefix(text: &str, found: &Match<'_>) -> String {
    text[..found.start()]
        .to_ascii_lowercase()
        .trim_end_matches(|character: char| character.is_ascii_whitespace())
        .trim_end_matches(['"', '\''])
        .trim_end_matches(|character: char| character.is_ascii_whitespace())
        .to_string()
}

/// Keeps timestamps that are explicit state fields, such as `event_at`, in
/// the outcome identity. Log prose still falls through to normalization.
fn iso_timestamp_context(text: &str, found: &Match<'_>) -> bool {
    let field_prefix = field_prefix(text, found);
    ![
        "event_at\":",
        "eventat\":",
        "created_at\":",
        "updated_at\":",
        "timestamp\":",
    ]
    .iter()
    .any(|field| field_prefix.ends_with(field))
}

fn state_timestamp_context(text: &str, found: &Match<'_>) -> bool {
    let field_prefix = field_prefix(text, found);
    [
        "event_at\":",
        "eventat\":",
        "created_at\":",
        "updated_at\":",
        "timestamp\":",
    ]
    .iter()
    .any(|field| field_prefix.ends_with(field))
}

/// Keeps PIDs that identify a newly created process rather than describing a
/// diagnostic observation about an existing one.
fn pid_context(text: &str, found: &Match<'_>) -> bool {
    let before = text[..found.start()].to_ascii_lowercase();
    let context = before
        .rsplit(['\n', '.', '!', '?'])
        .next()
        .unwrap_or(&before);
    !["started", "created", "spawned", "launched"]
        .iter()
        .any(|verb| {
            context
                .split(|character: char| !character.is_ascii_alphabetic())
                .any(|word| word == *verb)
        })
}

/// Accepts clock-shaped values only when their surrounding syntax indicates a
/// timestamp or log time, rather than a semantic position or counter.
fn clock_context(text: &str, found: &Match<'_>) -> bool {
    let before = &text[..found.start()];
    let after = &text[found.end()..];
    let preceded_by_context = before.ends_with(['[', '('])
        || ["at ", "on ", "time ", "timestamp "]
            .iter()
            .any(|prefix| before.to_ascii_lowercase().ends_with(prefix));
    let followed_by_boundary = after
        .chars()
        .next()
        .is_none_or(|character| !character.is_alphanumeric() && character != '_');
    preceded_by_context && followed_by_boundary
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
/// ISO timestamp before its clock time).
///
/// Long hex ids (commit SHAs, checksums, content hashes) are deliberately not
/// normalized: they are usually the *answer* a tool returns, so three different
/// commit ids are three different results, not a repeat.
static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    vec![
        // ISO-8601 / RFC 3339 timestamp, `T` or space separated, with optional
        // seconds, fraction and zone.
        rule(
            r"(?-u:\b)\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?",
            "<timestamp>",
            |text, found| not_followed_by_word(text, found) && iso_timestamp_context(text, found),
        ),
        rule(
            r#"(?i)(?-u:\b)(?:request|trace|correlation|session|run|call|invocation|operation|message|event)(?:[_ -]?(?:id|uuid))?\s*["']?\s*(?:[:=]\s*|\s+)["']?\s*(?P<span>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?-u:\b)"#,
            "<uuid>",
            uuid_context,
        ),
        rule(
            r#"(?i)(?-u:\b)req\s*["']?\s*(?:[:=]\s*|\s+)["']?(?P<span>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?-u:\b)"#,
            "<uuid>",
            accept_all,
        ),
        rule(
            r"(?-u:\b)(?P<span>\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?)(?-u:\b)",
            "<time>",
            clock_context,
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
            |text, found| not_after_dot(text, found) && !state_timestamp_context(text, found),
        ),
        // Measurement fields and phrases such as `duration=123ms`,
        // `elapsed 1.2s`, `took 1h2m3.5s`, and log phrases such as
        // `after 5003ms` or `...2026-10-06T12:00:01Z in 12ms`.
        rule(
            r#"(?i)(?-u:\b)(?:(?:took|elapsed|duration|latency|timeout|wait(?:ed)?|sleep(?:ing)?)(?:\s*(?:was|of|in))?|after|in)\s*["']?\s*(?:[:=]\s*)?["']?(?P<span>(?:(?:\d+h)?(?:\d+m)?(?:\d+\.\d+|\d+)s|\d+(?:\.\d+)?(?:ns|[µu]s|ms)))(?-u:\b)"#,
            "<duration>",
            |text, found| not_after_dot(text, found) && duration_context(text, found),
        ),
        rule(
            r"(?i)(?-u:\b)(?:attempt|retry)\s*#?\s*\d+(?:\s*(?:of|/)\s*\d+)?(?-u:\b)",
            "<attempt>",
            accept_all,
        ),
        rule(
            r"(?i)(?-u:\b)pid(?:\s*[=:]\s*|\s+)\d+(?-u:\b)",
            "<pid>",
            pid_context,
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
/// `1.2s`), `attempt N` / `retry N of M` counters, diagnostic `pid N` values,
/// and UUIDs. Long hex
/// ids are kept, because they are usually content. Everything else, including every other number, is
/// kept verbatim.
///
/// An outcome whose value is nothing but a volatile span (the stdout of
/// `git rev-parse HEAD`, a checksum, a bare timestamp) is returned unchanged:
/// when fewer than four alphanumeric characters remain outside the spans, the
/// spans are the content rather than noise around it, and two such outcomes
/// that differ are different outcomes.
pub fn normalize_volatile(text: &str) -> String {
    let mut current: Cow<'_, str> = Cow::Borrowed(text);
    let mut removed = 0;
    for rule in RULES.iter() {
        if let Some((rewritten, alnum)) = rewrite(rule, &current) {
            current = Cow::Owned(rewritten);
            removed += alnum;
        }
    }
    let residue = alnum_count(text).saturating_sub(removed);
    if matches!(current, Cow::Borrowed(_)) || residue < MIN_RESIDUE_CHARS {
        return text.to_string();
    }
    current.into_owned()
}

/// `text` with every accepted match of `rule` replaced, plus the number of
/// alphanumeric characters the replaced spans held, or `None` when nothing
/// matched (so the common no-match case allocates nothing).
///
/// Spans never overlap an earlier placeholder (every rule needs digits, which
/// no placeholder contains), so the removed counts sum to what the original
/// text lost.
fn rewrite(rule: &Rule, text: &str) -> Option<(String, usize)> {
    let mut out: Option<String> = None;
    let mut removed = 0;
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
        removed += alnum_count(span.as_str());
        last = span.end();
    }
    out.map(|mut out| {
        out.push_str(&text[last..]);
        (out, removed)
    })
}

fn alnum_count(text: &str) -> usize {
    text.chars().filter(|c| c.is_alphanumeric()).count()
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
