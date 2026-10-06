use super::*;
use crate::transcript::{
    DisplayRecord, TranscriptLocator, TranscriptMeta, TranscriptTurn, read_transcript,
    read_transcript_display, resolve_keyed_transcript_path, session_stem,
};
use std::path::{Path, PathBuf};
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

#[tokio::test]
async fn a_session_without_a_transcript_is_reported_and_not_created() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());

    let outcome = append_background_message(
        &locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1"),
    )
    .await
    .unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::NoSession);
    assert!(!head_path(dir.path(), &session()).exists());
}

#[tokio::test]
async fn the_display_reader_shows_the_message_with_its_provenance() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    live_turn(&locator, &session(), &[], &first_turn());

    append_background_message(
        &locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1"),
    )
    .await
    .unwrap();

    let display = read_transcript_display(&head_path(dir.path(), &session())).unwrap();
    let Some(DisplayRecord::Message(last)) = display.records.last() else {
        panic!("expected a message record last");
    };
    assert_eq!(last.message.role, "assistant");
    assert_eq!(last.message.content, "Time to stretch!");
    assert!(!last.interrupted);
    assert_eq!(
        last.background,
        Some(BackgroundOrigin {
            idempotency_key: "run-1".into(),
            provenance: serde_json::json!({"kind": "cron", "job_id": "job-1", "run_id": "run-1"}),
        })
    );
    // Rows the live turn wrote carry no background origin.
    let Some(DisplayRecord::Message(first)) = display.records.first() else {
        panic!("expected a message record first");
    };
    assert_eq!(first.background, None);
}

#[tokio::test]
async fn a_repeated_idempotency_key_is_a_duplicate_and_writes_nothing() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    live_turn(&locator, &session(), &[], &first_turn());
    let message = TranscriptMessage::assistant("Time to stretch!");
    let first = append_background_message(&locator, &session(), message.clone(), options("run-1"))
        .await
        .unwrap();
    let before = std::fs::read(head_path(dir.path(), &session())).unwrap();

    let again = append_background_message(&locator, &session(), message.clone(), options("run-1"))
        .await
        .unwrap();

    assert_eq!(first, BackgroundAppendOutcome::Appended { generation: 0 });
    assert_eq!(again, BackgroundAppendOutcome::Duplicate { generation: 0 });
    assert_eq!(
        std::fs::read(head_path(dir.path(), &session())).unwrap(),
        before
    );
    // A different key is a different delivery.
    let other = append_background_message(&locator, &session(), message, options("run-2"))
        .await
        .unwrap();
    assert_eq!(other, BackgroundAppendOutcome::Appended { generation: 0 });
}

/// Seals generation 0 and opens generation 1 with `retained`, the way a
/// compaction in the live turn path does.
fn compact(locator: &FileTranscriptLocator, retained: &[TranscriptMessage]) {
    let (_, handle) = locator.begin_generation(&session(), meta()).unwrap();
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: retained,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

#[tokio::test]
async fn a_stale_expected_generation_writes_nothing() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    live_turn(&locator, &session(), &[], &first_turn());
    compact(&locator, &[TranscriptMessage::system("summary")]);
    let sealed = resolve_keyed_transcript_path(dir.path(), &session_stem(&session())).unwrap();
    let head = head_path(dir.path(), &session());
    let (sealed_before, head_before) = (
        std::fs::read(&sealed).unwrap(),
        std::fs::read(&head).unwrap(),
    );

    let outcome = append_background_message(
        &locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1").expecting_generation(0),
    )
    .await
    .unwrap();

    assert_eq!(
        outcome,
        BackgroundAppendOutcome::StaleGeneration {
            expected: 0,
            head: 1
        }
    );
    assert_eq!(std::fs::read(&sealed).unwrap(), sealed_before);
    assert_eq!(std::fs::read(&head).unwrap(), head_before);
}

#[tokio::test]
async fn the_append_lands_in_the_head_generation() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    live_turn(&locator, &session(), &[], &first_turn());
    compact(&locator, &[TranscriptMessage::system("summary")]);

    let outcome = append_background_message(
        &locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1").expecting_generation(1),
    )
    .await
    .unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::Appended { generation: 1 });
    let head = read_transcript(&head_path(dir.path(), &session())).unwrap();
    assert_eq!(
        head.messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        ["summary", "Time to stretch!"]
    );
    let sealed = read_transcript(
        &resolve_keyed_transcript_path(dir.path(), &session_stem(&session())).unwrap(),
    )
    .unwrap();
    assert_eq!(sealed.messages.len(), 2);
}

/// The cron job finishes while the user is mid-turn in the same thread. The
/// live turn holds the turn lock from its resume read to its persist, so the
/// background append waits for it, and the next turn resumes over both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_append_waits_for_the_live_turn_and_keeps_both_in_order() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let path = head_path(dir.path(), &session());
    live_turn(&locator, &session(), &[], &first_turn());

    // Turn two starts: lock, then read its baseline.
    let turn = lock_session_turn(&locator, &session()).await.unwrap();
    let baseline = read_transcript(&path).unwrap().messages;

    let workspace = dir.path().to_path_buf();
    let delivery = tokio::spawn(async move {
        append_background_message(
            &FileTranscriptLocator::new(workspace),
            &session(),
            TranscriptMessage::assistant("Time to stretch!"),
            options("run-1"),
        )
        .await
    });
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(
        !delivery.is_finished(),
        "delivery must wait for the live turn"
    );
    assert_eq!(read_transcript(&path).unwrap().messages, baseline);

    // Turn two persists against the baseline it read, then releases the lock.
    let mut second = baseline.clone();
    second.push(TranscriptMessage::user("what's the weather?"));
    second.push(TranscriptMessage::assistant("Sunny."));
    live_turn(&locator, &session(), &baseline, &second);
    drop(turn);

    assert_eq!(
        delivery.await.unwrap().unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );

    // Turn three resumes over everything and appends on top of it.
    let _turn = lock_session_turn(&locator, &session()).await.unwrap();
    let resumed = read_transcript(&path).unwrap().messages;
    let mut third = resumed.clone();
    third.push(TranscriptMessage::user("did you remind me?"));
    third.push(TranscriptMessage::assistant("Yes, at 5."));
    live_turn(&locator, &session(), &resumed, &third);

    let contents: Vec<(String, String)> = read_transcript(&path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| (message.role, message.content))
        .collect();
    let expected: Vec<(String, String)> = [
        ("user", "remind me to stretch at 5"),
        ("assistant", "Scheduled."),
        ("user", "what's the weather?"),
        ("assistant", "Sunny."),
        ("assistant", "Time to stretch!"),
        ("user", "did you remind me?"),
        ("assistant", "Yes, at 5."),
    ]
    .into_iter()
    .map(|(role, content)| (role.to_string(), content.to_string()))
    .collect();
    assert_eq!(contents, expected);
}
