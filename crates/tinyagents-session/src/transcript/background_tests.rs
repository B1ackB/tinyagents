use super::*;
use crate::transcript::{
    DisplayRecord, TranscriptLocator, TranscriptTurn, read_transcript, read_transcript_display,
};
use tempfile::tempdir;

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: Some("orchestrator".into()),
        agent_type: None,
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-10-06T00:00:00Z".into(),
        updated: "2026-10-06T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("thread-1".into()),
        task_id: None,
    }
}

fn session() -> SessionRef {
    SessionRef::scoped("thread-1", "orchestrator")
}

fn options(key: &str) -> BackgroundAppend {
    BackgroundAppend::new(
        key,
        serde_json::json!({"kind": "cron", "job_id": "job-1", "run_id": key}),
    )
}

/// Writes one turn the way the live turn path does: through the session's own
/// history handle, diffing `next` against `prev`.
fn live_turn(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    prev: &[TranscriptMessage],
    next: &[TranscriptMessage],
) {
    let handle = locator.open_session(session, meta()).unwrap();
    handle
        .append_turn(TranscriptTurn {
            prev,
            next,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

fn first_turn() -> Vec<TranscriptMessage> {
    vec![
        TranscriptMessage::user("remind me to stretch at 5"),
        TranscriptMessage::assistant("Scheduled."),
    ]
}

fn head_path(dir: &Path, session: &SessionRef) -> PathBuf {
    let locator = FileTranscriptLocator::new(dir);
    let head = locator.head_generation(session);
    resolve_keyed_transcript_path(dir, &session_stem(&head)).unwrap()
}

#[tokio::test]
async fn an_appended_message_is_resumed_as_an_assistant_turn() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    live_turn(&locator, &session(), &[], &first_turn());

    let outcome = append_background_message(
        &locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1"),
    )
    .await
    .unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::Appended { generation: 0 });
    let resumed = read_transcript(&head_path(dir.path(), &session())).unwrap();
    let last = resumed.messages.last().unwrap();
    assert_eq!(resumed.messages.len(), 3);
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "Time to stretch!");
}
