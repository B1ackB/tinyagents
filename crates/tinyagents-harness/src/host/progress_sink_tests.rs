use super::*;

fn run() -> RunId {
    RunId::new("run-1")
}

fn started() -> ProgressEvent {
    ProgressEvent::Started {
        run: run(),
        thread: Some(ThreadId::new("thread-1")),
        agent: "lead".to_string(),
    }
}

fn tool_call() -> ProgressEvent {
    ProgressEvent::ToolCall {
        run: run(),
        call: CallId::new("call-1"),
        tool: "search".to_string(),
    }
}

fn tool_call_finished(success: bool, output: &str) -> ProgressEvent {
    ProgressEvent::ToolCallFinished {
        run: run(),
        call: CallId::new("call-1"),
        success,
        output: output.to_string(),
    }
}

fn token(text: &str) -> ProgressEvent {
    ProgressEvent::Token {
        run: run(),
        text: text.to_string(),
    }
}

fn finished() -> ProgressEvent {
    ProgressEvent::Finished {
        run: run(),
        usage: Some(Usage {
            input_tokens: 12,
            output_tokens: 3,
            total_tokens: 15,
            ..Usage::default()
        }),
    }
}

fn error() -> ProgressEvent {
    ProgressEvent::Error {
        run: run(),
        message: "provider unavailable".to_string(),
    }
}

fn all_variants() -> Vec<ProgressEvent> {
    vec![
        started(),
        tool_call(),
        tool_call_finished(true, "result"),
        token("hi"),
        finished(),
        error(),
    ]
}

// ── Value-type invariants ────────────────────────────────────────────────

#[test]
fn every_variant_exposes_its_run_id() {
    for ev in all_variants() {
        assert_eq!(ev.run_id(), &run(), "run id missing for {ev:?}");
    }
}

#[test]
fn only_finished_and_error_are_terminal() {
    assert!(!started().is_terminal());
    assert!(!tool_call().is_terminal());
    assert!(!token("x").is_terminal());
    assert!(finished().is_terminal());
    assert!(error().is_terminal());
}

#[test]
fn tool_completion_is_not_terminal() {
    // A turn emits one of these per tool call, several before `Finished`.
    // If this ever reads as terminal a host tears its run UI down at the
    // first tool result, which looks like a truncated turn rather than a
    // bug in this predicate.
    assert!(!tool_call_finished(true, "ok").is_terminal());
    assert!(!tool_call_finished(false, "boom").is_terminal());
}

#[test]
fn tool_completion_correlates_with_its_opening_call() {
    // The `call` id is the only thing joining the two events. If they ever
    // stop matching, a host closes the wrong timeline row and the bug
    // surfaces as a mis-rendered tool, not as a failure here.
    let (
        ProgressEvent::ToolCall { call: opened, .. },
        ProgressEvent::ToolCallFinished { call: closed, .. },
    ) = (tool_call(), tool_call_finished(true, "ok"))
    else {
        panic!("constructors changed shape");
    };
    assert_eq!(opened, closed);
}

#[test]
fn failure_is_representable_and_distinct_from_success() {
    // The whole point of #88: a host must be able to report that a tool
    // failed. If these ever compare equal, `success` has stopped carrying
    // information and every failed tool renders as a successful one.
    assert_ne!(
        tool_call_finished(true, "same"),
        tool_call_finished(false, "same")
    );
}

#[test]
fn uncaptured_output_is_representable() {
    // Empty output is a legitimate state (payload capture off), not a
    // reason to withhold the outcome — success must still round-trip.
    let ev = tool_call_finished(false, "");
    let json = serde_json::to_string(&ev).expect("serialize");
    let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, ev);
}

#[test]
fn events_round_trip_through_serde() {
    for ev in all_variants() {
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: ProgressEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, ev);
    }
}

#[test]
fn serialized_events_are_internally_tagged_by_kind() {
    let json = serde_json::to_value(tool_call()).expect("serialize");
    assert_eq!(json["kind"], "tool_call");
    assert_eq!(json["tool"], "search");

    // Distinct tag from `tool_call` — these cross a serde boundary into
    // host code, so a collision would silently merge open and close.
    let json = serde_json::to_value(tool_call_finished(false, "boom")).expect("serialize");
    assert_eq!(json["kind"], "tool_call_finished");
    assert_eq!(json["success"], false);
    assert_eq!(json["output"], "boom");
}

#[test]
fn absent_usage_is_distinct_from_zero_usage() {
    let unreported = ProgressEvent::Finished {
        run: run(),
        usage: None,
    };
    let free_call = ProgressEvent::Finished {
        run: run(),
        usage: Some(Usage::default()),
    };
    assert_ne!(unreported, free_call);
}

// ── NoopProgressSink ─────────────────────────────────────────────────────

#[tokio::test]
async fn noop_sink_accepts_every_variant_and_keeps_nothing() {
    let sink = NoopProgressSink::new();
    for ev in all_variants() {
        sink.emit(ev).await;
    }
    // Nothing observable to assert beyond "did not panic, returned unit" —
    // which is precisely the contract: emit has no failure channel.
    assert_eq!(sink, NoopProgressSink);
}

// ── RecordingProgressSink ────────────────────────────────────────────────

#[tokio::test]
async fn recording_sink_starts_empty() {
    let sink = RecordingProgressSink::new();
    assert!(sink.is_empty());
    assert_eq!(sink.len(), 0);
    assert!(sink.events().is_empty());
}

#[tokio::test]
async fn recording_sink_preserves_arrival_order() {
    let sink = RecordingProgressSink::new();
    for ev in all_variants() {
        sink.emit(ev).await;
    }
    assert_eq!(sink.events(), all_variants());
    assert_eq!(sink.len(), all_variants().len());
    assert!(!sink.is_empty());
}

#[tokio::test]
async fn recording_sink_retains_duplicate_events() {
    let sink = RecordingProgressSink::new();
    sink.emit(token("a")).await;
    sink.emit(token("a")).await;
    // Streamed chunks legitimately repeat; a recorder that deduplicated
    // would hide a real double-emit regression.
    assert_eq!(sink.events(), vec![token("a"), token("a")]);
}

#[tokio::test]
async fn recording_sink_clear_resets_the_buffer() {
    let sink = RecordingProgressSink::new();
    sink.emit(started()).await;
    sink.clear();
    assert!(sink.is_empty());
    sink.emit(finished()).await;
    assert_eq!(sink.events(), vec![finished()]);
}

#[tokio::test]
async fn recording_sink_clones_share_one_buffer() {
    let sink = RecordingProgressSink::new();
    let clone = sink.clone();
    clone.emit(started()).await;
    // The runtime hands out clones to concurrent turns; if they did not
    // share, a test would observe an empty recorder and pass vacuously.
    assert_eq!(sink.events(), vec![started()]);
}

#[tokio::test]
async fn recording_sink_records_through_a_trait_object() {
    let sink = RecordingProgressSink::new();
    let dynamic: Arc<dyn ProgressSink> = Arc::new(sink.clone());
    dynamic.emit(token("chunk")).await;
    assert_eq!(sink.events(), vec![token("chunk")]);
}

#[tokio::test]
async fn recording_sink_survives_a_poisoned_lock() {
    let sink = RecordingProgressSink::new();
    sink.emit(started()).await;

    let poisoner = sink.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoner.events.lock().expect("acquire");
        panic!("poison the mutex");
    })
    .join();

    // Progress must never escalate someone else's panic into a failed turn.
    sink.emit(finished()).await;
    assert_eq!(sink.events(), vec![started(), finished()]);
}
