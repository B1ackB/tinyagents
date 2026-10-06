use super::*;
use crate::{DriverFailure, DriverOutcome, SessionBuilder, TranscriptCodec};
use async_trait::async_trait;
use tinyagents_session::transcript::{
    BackgroundAppend, BackgroundAppendOutcome, FileTranscriptLocator, SessionTranscript,
    TranscriptMeta, append_background_message, read_transcript, resolve_keyed_transcript_path,
    session_stem,
};
use tokio::sync::Notify;

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent".into()),
        agent_type: None,
        dispatcher: "test".into(),
        provider: None,
        model: None,
        created: "then".into(),
        updated: "then".into(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// Maps user/assistant rows one-to-one, keeping prior raw rows verbatim.
struct RoleCodec;

impl TranscriptCodec for RoleCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "user" => Message::user(&row.content),
                _ => Message::assistant(&row.content),
            })
            .collect())
    }

    fn reconcile(
        &self,
        prior: &[TranscriptMessage],
        _: &[Message],
        next: &[Message],
        _: &crate::TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        let mut rows = prior.to_vec();
        rows.extend(
            next[prior.len().min(next.len())..]
                .iter()
                .map(|message| match message {
                    Message::User(_) => TranscriptMessage::user(message.text()),
                    _ => TranscriptMessage::assistant(message.text()),
                }),
        );
        Ok(rows)
    }
}

/// Answers each turn with `reply`, but only after the test releases it.
struct GatedDriver {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl SessionDriver for GatedDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.started.notify_one();
        self.release.notified().await;
        let mut history = request.history;
        history.push(Message::assistant("Sunny."));
        Ok(DriverOutcome {
            history,
            output: Some("Sunny.".into()),
            partial: None,
            interrupted: false,
        })
    }
}

/// A background delivery issued while a turn is running waits for that turn
/// to persist, then lands after it — instead of staling the turn's baseline
/// and failing its persist.
#[tokio::test]
async fn a_running_turn_holds_off_a_background_append_into_its_session() {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
    let session_ref = SessionRef::scoped("thread-1", "agent");
    let path =
        resolve_keyed_transcript_path(directory.path(), &session_stem(&session_ref)).unwrap();
    tinyagents_session::transcript::write_transcript(
        &path,
        &[
            TranscriptMessage::user("remind me at 5"),
            TranscriptMessage::assistant("Scheduled."),
        ],
        &meta(),
        None,
    )
    .unwrap();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mut session = SessionBuilder::new(Arc::new(GatedDriver {
        started: started.clone(),
        release: release.clone(),
    }))
    .codec(Arc::new(RoleCodec))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();

    let turn = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("weather?")),
                TurnOptions {
                    resume: ResumeMode::Session,
                    ..TurnOptions::default()
                },
            )
            .await
    });
    started.notified().await;

    let delivery_locator = locator.clone();
    let delivery_session = session_ref.clone();
    let delivery = tokio::spawn(async move {
        append_background_message(
            &delivery_locator,
            &delivery_session,
            TranscriptMessage::assistant("Time to stretch!"),
            BackgroundAppend::new("run-1", serde_json::json!({"kind": "cron"})),
        )
        .await
    });
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(!delivery.is_finished(), "delivery must wait for the turn");

    release.notify_one();
    turn.await.unwrap().unwrap();
    assert_eq!(
        delivery.await.unwrap().unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );
    let contents: Vec<String> = read_transcript(&path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(
        contents,
        [
            "remind me at 5",
            "Scheduled.",
            "weather?",
            "Sunny.",
            "Time to stretch!"
        ]
    );
}
