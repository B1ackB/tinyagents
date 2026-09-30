//! Tests for [`TurnStateMirror`]: snapshot priming, transcript coalescing,
//! ordering keys and interrupted-turn finalization.

use super::*;
use crate::transcript::{
    self, DisplayRecord, TranscriptMessage, TranscriptMeta, read_transcript,
    read_transcript_display,
};
use crate::turn_state::types::{SubagentActivity, ToolTimelineEntry, ToolTimelineStatus};
use crate::turn_state::types::{TranscriptItem, TurnLifecycle};
use tempfile::tempdir;

fn fresh(thread_id: &str) -> (tempfile::TempDir, TurnStateMirror) {
    let dir = tempdir().expect("tempdir");
    let store = TurnStateStore::new(dir.path().to_path_buf());
    let mirror = TurnStateMirror::new(store, thread_id, "req-1");
    (dir, mirror)
}

fn seed_root_transcript(workspace: &std::path::Path, thread_id: &str) -> std::path::PathBuf {
    let path = transcript::resolve_keyed_transcript_path(workspace, "100_orchestrator")
        .expect("resolve path");
    let meta = TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: None,
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-07-21T00:00:00Z".into(),
        updated: "2026-07-21T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some(thread_id.to_string()),
        task_id: None,
    };
    transcript::write_transcript(
        &path,
        &[TranscriptMessage::new("user", "hello there")],
        &meta,
        None,
    )
    .expect("seed transcript");
    path
}

#[test]
fn new_flushes_a_started_snapshot() {
    let (dir, mirror) = fresh("thr_new");
    assert_eq!(mirror.snapshot().lifecycle, TurnLifecycle::Started);
    let stored = TurnStateStore::new(dir.path().to_path_buf())
        .get("thr_new")
        .expect("get")
        .expect("snapshot on disk right after new()");
    assert_eq!(stored.request_id, "req-1");
    assert_eq!(stored.lifecycle, TurnLifecycle::Started);
}

#[test]
fn ordering_keys_are_monotonic_and_independent() {
    let (_dir, mut m) = fresh("thr");
    assert_eq!((m.next_seq(), m.next_seq(), m.next_seq()), (0, 1, 2));
    assert_eq!((m.next_tool_seq(), m.next_tool_seq()), (0, 1));
    assert_eq!(m.next_seq(), 3, "tool keys do not consume transcript keys");
}

#[test]
fn narration_coalesces_within_a_round_and_splits_across_rounds() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_narration(1, "Hel");
    m.push_transcript_narration(1, "lo");
    m.push_transcript_narration(2, "next");
    assert_eq!(m.state.transcript.len(), 2);
    match &m.state.transcript[0] {
        TranscriptItem::Narration { round, seq, text } => {
            assert_eq!((*round, *seq, text.as_str()), (1, 0, "Hello"));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn thinking_coalesces_and_an_intervening_tool_starts_a_new_block() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_thinking(1, "a");
    m.push_transcript_thinking(1, "b");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_thinking(1, "c");
    assert_eq!(m.state.transcript.len(), 3);
    match &m.state.transcript[0] {
        TranscriptItem::Thinking {
            text,
            started_at,
            ended_at,
            ..
        } => {
            assert_eq!(text, "ab");
            assert!(started_at.is_some() && ended_at.is_some());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tool_call_is_recorded_once_per_call_id() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_tool(1, "call-2");
    assert_eq!(m.state.transcript.len(), 2);
}

#[test]
fn subagent_prose_coalesces_per_kind_and_iteration() {
    let (_dir, mut m) = fresh("thr");
    m.state.tool_timeline.push(ToolTimelineEntry {
        id: "subagent:t1".into(),
        name: "spawn".into(),
        round: 1,
        status: ToolTimelineStatus::Running,
        args_buffer: None,
        display_name: None,
        detail: None,
        source_tool_name: None,
        subagent: Some(SubagentActivity::default()),
        failure: None,
        output: None,
        seq: Some(0),
    });
    m.push_subagent_prose("t1", 1, "x", false);
    m.push_subagent_prose("t1", 1, "y", false);
    m.push_subagent_prose("t1", 1, "think", true);
    m.push_subagent_prose("unknown", 1, "dropped", false);
    let activity = m
        .find_subagent_entry_mut("t1")
        .unwrap()
        .subagent
        .as_ref()
        .unwrap();
    assert_eq!(activity.transcript.len(), 2);
}

#[test]
fn finish_marks_an_unfinished_turn_interrupted() {
    let (dir, mut m) = fresh("thr_new");
    m.state.streaming_text = "orphan partial".into();
    // No transcript exists yet: must not panic, snapshot keeps the partial.
    m.finish();
    let stored = TurnStateStore::new(dir.path().to_path_buf())
        .get("thr_new")
        .expect("get")
        .expect("snapshot present");
    assert_eq!(stored.lifecycle, TurnLifecycle::Interrupted);
    assert_eq!(stored.streaming_text, "orphan partial");
}

#[test]
fn finish_appends_interrupted_partial_to_existing_transcript() {
    let dir = tempdir().expect("tempdir");
    let path = seed_root_transcript(dir.path(), "thr_abc");
    let store = TurnStateStore::new(dir.path().to_path_buf());
    let mut m = TurnStateMirror::new(store, "thr_abc", "req-9");
    m.state.iteration = 2;
    m.state.thinking = "hmm".into();
    m.state.streaming_text = "half an answer".into();
    m.finish();

    let model = read_transcript(&path).expect("read model context");
    assert!(
        !model
            .messages
            .iter()
            .any(|msg| msg.content.contains("half an answer")),
        "interrupted partial must be excluded from the model context"
    );
    let display = read_transcript_display(&path).expect("read display");
    let partial = display
        .records
        .iter()
        .find_map(|r| match r {
            DisplayRecord::Message(msg) if msg.interrupted => Some(msg),
            _ => None,
        })
        .expect("display must include the interrupted partial");
    assert_eq!(partial.message.content, "half an answer");
    assert_eq!(partial.request_id.as_deref(), Some("req-9"));
    assert_eq!(partial.iteration, Some(2));
    assert_eq!(partial.reasoning_content.as_deref(), Some("hmm"));
}

#[test]
fn finish_after_completion_writes_no_partial() {
    let dir = tempdir().expect("tempdir");
    let path = seed_root_transcript(dir.path(), "thr_done");
    let store = TurnStateStore::new(dir.path().to_path_buf());
    let mut m = TurnStateMirror::new(store, "thr_done", "req-done");
    m.state.streaming_text = "final answer".into();
    m.turn_completed = true;
    m.finish();
    let display = read_transcript_display(&path).expect("read display");
    assert!(
        !display
            .records
            .iter()
            .any(|r| matches!(r, DisplayRecord::Message(msg) if msg.interrupted)),
        "a completed turn must not append an interrupted partial"
    );
}

#[test]
fn finish_keeps_a_terminal_outcome_the_driver_already_recorded() {
    let dir = tempdir().expect("tempdir");
    let path = seed_root_transcript(dir.path(), "thr_settled");
    let store = TurnStateStore::new(dir.path().to_path_buf());
    let mut m = TurnStateMirror::new(store.clone(), "thr_settled", "req-s");
    m.state.streaming_text = "delivered reply".into();
    // The turn driver settles the snapshot as completed behind the mirror's back.
    let mut settled = m.state.clone();
    settled.lifecycle = TurnLifecycle::Completed;
    store.put(&settled).expect("settle");
    m.finish();
    let stored = store.get_turn("thr_settled", "req-s").unwrap().unwrap();
    assert_eq!(stored.lifecycle, TurnLifecycle::Completed);
    let display = read_transcript_display(&path).expect("read display");
    assert!(
        !display
            .records
            .iter()
            .any(|r| matches!(r, DisplayRecord::Message(msg) if msg.interrupted)),
        "a settled turn's reply must not be duplicated as a partial"
    );
}
