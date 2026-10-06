use super::*;

fn norm(text: &str) -> String {
    VolatileSpanNormalizer.fingerprint(text)
}

fn same(a: &str, b: &str) {
    assert_eq!(norm(a), norm(b), "expected equal fingerprints:\n  {a}\n  {b}");
}

fn differ(a: &str, b: &str) {
    assert_ne!(norm(a), norm(b), "expected distinct fingerprints:\n  {a}\n  {b}");
}

fn untouched(text: &str) {
    assert_eq!(norm(text), text, "must not be normalized");
}

#[test]
fn iso_and_rfc3339_timestamps_are_normalized() {
    same("done at 2026-10-06T12:34:56Z", "done at 2027-01-01T00:00:01Z");
    same(
        "at 2026-10-06T12:34:56.123+02:00 ok",
        "at 2026-10-07T01:02:03.999-05:00 ok",
    );
    same("2026-10-06 12:34:56 started", "2026-10-06 12:34:57 started");
    same("2026-10-06T12:34Z x", "2026-10-06T12:35Z x");
    differ("2026-10-06T12:34:56Z started", "2026-10-06T12:34:56Z stopped");
}

#[test]
fn clock_times_are_normalized() {
    same("[12:34:56] ready", "[01:02:03] ready");
    same("[12:34:56.789] ready", "[01:02:03.1] ready");
    differ("[12:34:56] ready", "[12:34:56] failed");
}

#[test]
fn epoch_like_numbers_are_normalized() {
    same("ts=1759752896 ok", "ts=1759752999 ok");
    same("ts=1759752896123 ok", "ts=1759752999456 ok");
    same("ts=1759752896.25 ok", "ts=1759752999.5 ok");
    differ("ts=1759752896 ok", "ts=1759752896 bad");
}

#[test]
fn durations_are_normalized() {
    same("took 123ms", "took 4ms");
    same("took 1.2s", "took 45s");
    same("took 250us", "took 3us");
    same("elapsed 1h2m3.5s", "elapsed 4h5m6s");
    differ("took 123ms to read", "took 123ms to write");
}

#[test]
fn attempt_and_retry_counters_are_normalized() {
    same("attempt 1 failed", "attempt 7 failed");
    same("Attempt #2 failed", "attempt 9 failed");
    same("retry 1 of 5: boom", "retry 4 of 5: boom");
    same("(attempt 2/5) boom", "(attempt 3/5) boom");
    differ("attempt 1 failed", "attempt 1 succeeded");
}

#[test]
fn pids_are_normalized() {
    same("[pid 123] exited", "[pid 98765] exited");
    same("pid=42 exited", "pid: 77 exited");
    differ("pid 1 exited", "pid 1 crashed");
}

#[test]
fn uuids_and_long_hex_ids_are_normalized() {
    same(
        "request 123e4567-e89b-12d3-a456-426614174000 failed",
        "request 00000000-1111-2222-3333-444444444444 failed",
    );
    same(
        "commit 0123456789abcdef0123456789abcdef01234567",
        "commit fedcba9876543210fedcba9876543210fedcba98",
    );
    same("id 0x0123456789abcdef", "id 0xfedcba9876543210");
    differ("req 0123456789abcdef failed", "req 0123456789abcdef passed");
}

#[test]
fn ordinary_numbers_and_identifiers_are_not_normalized() {
    untouched("version 1.2.3 and v2.10.0");
    untouched("error at src/main.rs:123:45");
    untouched("line 42 of 100");
    untouched("3 files changed, 10 insertions");
    untouched("exit code 1");
    untouched("listening on port 8080");
    untouched("id 123456789");
    untouched("id 12345678901234");
    untouched("price 1.5");
    untouched("date 2026-10-06");
    untouched("attempts allowed");
    untouched("rapid 5 pidgin");
    untouched("deadbeef cafebabe");
    untouched("a_very_long_identifier_name_without_hex");
    untouched("2 minutes ago");
    untouched("1234567890123456789012");
}

#[test]
fn epoch_lookalikes_inside_other_numbers_are_left_alone() {
    untouched("ver 1.1759752896");
    untouched("2759752896");
}

#[test]
fn only_volatile_spans_change_surrounding_text_is_kept() {
    let out = norm("rows=15 at 2026-10-06T12:34:56Z took 12ms");
    assert!(out.starts_with("rows=15 at "), "got {out}");
    assert!(!out.contains("2026"), "got {out}");
    assert!(!out.contains("12ms"), "got {out}");
}
