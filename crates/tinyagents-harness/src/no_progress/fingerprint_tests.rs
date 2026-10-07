use super::*;

fn norm(text: &str) -> String {
    VolatileSpanNormalizer.fingerprint(text)
}

fn same(a: &str, b: &str) {
    assert_eq!(
        norm(a),
        norm(b),
        "expected equal fingerprints:\n  {a}\n  {b}"
    );
}

fn differ(a: &str, b: &str) {
    assert_ne!(
        norm(a),
        norm(b),
        "expected distinct fingerprints:\n  {a}\n  {b}"
    );
}

fn untouched(text: &str) {
    assert_eq!(norm(text), text, "must not be normalized");
}

#[test]
fn iso_and_rfc3339_timestamps_are_normalized() {
    same(
        "done at 2026-10-06T12:34:56Z",
        "done at 2027-01-01T00:00:01Z",
    );
    same(
        "at 2026-10-06T12:34:56.123+02:00 ok",
        "at 2026-10-07T01:02:03.999-05:00 ok",
    );
    same("2026-10-06 12:34:56 started", "2026-10-06 12:34:57 started");
    same("2026-10-06T12:34Z started", "2026-10-06T12:35Z started");
    differ(
        r#"{"status":"ok","event_at":"2026-10-06T12:34:56Z"}"#,
        r#"{"status":"ok","event_at":"2026-10-06T12:34:57Z"}"#,
    );
    differ(
        r#"{"event_at": "2026-10-06T12:34:56Z"}"#,
        r#"{"event_at": "2026-10-06T12:34:57Z"}"#,
    );
    differ(
        "2026-10-06T12:34:56Z started",
        "2026-10-06T12:34:56Z stopped",
    );
}

#[test]
fn clock_times_are_normalized() {
    same("[12:34:56] ready", "[01:02:03] ready");
    same("[12:34:56.789] ready", "[01:02:03.1] ready");
    differ("[12:34:56] ready", "[12:34:56] failed");
    differ("position=00:00:01 processed", "position=00:00:02 processed");
}

#[test]
fn epoch_numbers_are_normalized_only_under_a_time_key() {
    same("ts=1759752896 ok", "ts=1759752999 ok");
    same("ts=1759752896123 ok", "ts=1759752999456 ok");
    same("ts=1759752896.25 ok", "ts=1759752999.5 ok");
    differ(
        r#"{"timestamp": 1759752896, "ok": true}"#,
        r#"{"timestamp": 1759752999, "ok": true}"#,
    );
    same(
        "updated_at: 1759752896 saved",
        "updated_at: 1759752999 saved",
    );
    differ(
        r#"{"status":"running","updated_at":1759752896}"#,
        r#"{"status":"running","updated_at":1759752999}"#,
    );
    differ(
        r#"{"status":"running","updated_at": 1759752896}"#,
        r#"{"status":"running","updated_at": 1759752999}"#,
    );
    same("createdAt=1759752896 saved", "createdAt=1759752999 saved");
    same("mtime=1759752896 file", "mtime=1759752999 file");
    differ("ts=1759752896 ok", "ts=1759752896 bad");
}

#[test]
fn bare_epoch_sized_numbers_are_not_normalized() {
    untouched("1759752896");
    untouched("1759752896123");
    untouched("Content-Length: 1048576000");
    untouched("-rw-r--r-- 1 u g 1073741824 Oct 6 notes.bin");
    untouched("order 1234567890 shipped");
    untouched("size=1073741824");
    untouched("uptime=1759752896 s");
    untouched("bytes: 1500000000");
}

#[test]
fn outputs_that_are_only_volatile_fall_back_to_the_raw_text() {
    differ(
        "0123456789abcdef0123456789abcdef01234567\n",
        "fedcba9876543210fedcba9876543210fedcba98\n",
    );
    differ(
        "123e4567-e89b-12d3-a456-426614174000",
        "00000000-1111-2222-3333-444444444444",
    );
    differ("2026-10-06T12:34:56Z", "2026-10-06T12:34:57Z");
    differ("12:34:56", "12:34:57");
    differ("123ms", "456ms");
    // Enough non-volatile content remains, so the spans are still blanked.
    same("failed 2026-10-06T12:34:56Z", "failed 2026-10-06T12:34:57Z");
}

#[test]
fn durations_are_normalized() {
    same("took 123ms", "took 4ms");
    same("took 1.2s", "took 45s");
    same("took 123s", "took 456s");
    same("took 250us", "took 3us");
    same("elapsed 1h2m3.5s", "elapsed 4h5m6s");
    differ("took 123ms to read", "took 123ms to write");
    differ("lease expires in 30s", "lease expires in 20s");
    same(
        "doc fetched at 2026-10-06T12:00:00Z in 10ms",
        "doc fetched at 2026-10-06T12:00:01Z in 11ms",
    );
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
fn creation_results_keep_the_pid_identity() {
    differ("worker started pid 123", "worker started pid 124");
    differ("created pid=123", "created pid=124");
    differ("spawned pid 123", "spawned pid 124");
    differ("launched pid 123", "launched pid 124");
}

#[test]
fn uuids_are_normalized_but_long_hex_ids_are_content() {
    same(
        "request 123e4567-e89b-12d3-a456-426614174000 failed",
        "request 00000000-1111-2222-3333-444444444444 failed",
    );
    same(
        r#"{"request_id":"123e4567-e89b-12d3-a456-426614174000","status":"failed"}"#,
        r#"{"request_id":"00000000-1111-2222-3333-444444444444","status":"failed"}"#,
    );
    differ(
        r#"{"job_id":"123e4567-e89b-12d3-a456-426614174000","status":"ready"}"#,
        r#"{"job_id":"00000000-1111-2222-3333-444444444444","status":"ready"}"#,
    );
    differ(
        r#"{"event_id":"123e4567-e89b-12d3-a456-426614174000","status":"ready"}"#,
        r#"{"event_id":"00000000-1111-2222-3333-444444444444","status":"ready"}"#,
    );
    // A commit id or checksum is usually the answer, not noise around it.
    differ(
        "Created commit 0123456789abcdef0123456789abcdef01234567",
        "Created commit fedcba9876543210fedcba9876543210fedcba98",
    );
    differ(
        "request id 0x0123456789abcdef",
        "request id 0xfedcba9876543210",
    );
}

#[test]
fn timestamps_need_a_boundary_after_them() {
    differ(
        "token 2026-10-06T12:34:56Zebra seen",
        "token 2026-10-07T01:02:03Zebra seen",
    );
    same(
        "built at 2026-10-06T12:34:56Z.",
        "built at 2026-10-07T01:02:03Z.",
    );
}

#[test]
fn literal_placeholder_text_counts_as_residue() {
    // The output already contains placeholder-looking words; they are
    // content and must count toward the residue, so the fingerprint is the
    // normalized text rather than the verbatim fallback.
    same(
        "<timestamp> <duration> 2026-10-06T12:34:56Z",
        "<timestamp> <duration> 2026-10-07T01:02:03Z",
    );
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
    untouched("music from the 1990s");
    untouched("hundreds: 100s of files");
    untouched("a few 100s of times");
    untouched("1234567890123456789012");
}

#[test]
fn epoch_lookalikes_inside_other_numbers_are_left_alone() {
    untouched("ts=1.1759752896");
    untouched("ts=2759752896");
}

#[test]
fn only_volatile_spans_change_surrounding_text_is_kept() {
    let out = norm("rows=15 at 2026-10-06T12:34:56Z took 12ms");
    assert!(out.starts_with("rows=15 at "), "got {out}");
    assert!(!out.contains("2026"), "got {out}");
    assert!(!out.contains("12ms"), "got {out}");
}
