//! Live dual-write, shadow read and persisted-shape tests (neutral projector).

use std::path::Path;

use serde_json::json;
use tempfile::TempDir;
use tinyagents_harness::store::{AppendStore, JsonlAppendStore};

use super::convert::{JournalProjector, journal_messages, plain_journal_message, stream_name};
use super::live::{ShadowReadOutcome, shadow_read_compare, write_live_turn};
use super::ops::store_root;
use super::types::{ItemLedgerRecord, JournalMessage};
use crate::transcript::{SessionTranscript, read_transcript};

const STEM: &str = "1719000000_orchestrator";

fn seed(ws: &Path, assistant: &str) -> SessionTranscript {
    let dir = ws.join("session_raw");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{STEM}.jsonl"));
    std::fs::write(
        &path,
        format!(
            "{{\"_meta\":{{\"agent\":\"orchestrator\",\"dispatcher\":\"native\",\
             \"created\":\"2024-01-01T00:00:00Z\",\"updated\":\"2024-01-01T00:05:00Z\",\
             \"turn_count\":1,\"input_tokens\":1,\"output_tokens\":1,\"cached_input_tokens\":0,\
             \"charged_amount_usd\":0.0,\"thread_id\":\"t-1\"}}}}\n\
             {{\"role\":\"user\",\"content\":\"hi\"}}\n\
             {{\"role\":\"assistant\",\"content\":\"{assistant}\"}}\n"
        ),
    )
    .unwrap();
    read_transcript(&path).unwrap()
}

async fn readback(ws: &Path) -> Vec<JournalMessage> {
    JsonlAppendStore::new(store_root(ws).join("journal"))
        .read_from(&stream_name(STEM), 0)
        .await
        .unwrap()
        .into_iter()
        .map(|(_, v)| serde_json::from_value(v).unwrap())
        .collect()
}

#[tokio::test]
async fn live_write_matches_projection_and_is_rewritten_each_turn() {
    let ws = TempDir::new().unwrap();
    let t = seed(ws.path(), "done");
    write_live_turn(ws.path(), STEM, &t, plain_journal_message)
        .await
        .unwrap();
    write_live_turn(ws.path(), STEM, &t, plain_journal_message)
        .await
        .unwrap();
    let got = readback(ws.path()).await;
    assert_eq!(got, journal_messages(&t, plain_journal_message));
    assert_eq!(got.len(), 2, "second write replaces, not appends");
}

#[tokio::test]
async fn shadow_read_match_unavailable_and_divergence() {
    let ws = TempDir::new().unwrap();
    let t = seed(ws.path(), "done");
    assert_eq!(
        shadow_read_compare(ws.path(), STEM, &t, plain_journal_message).await,
        ShadowReadOutcome::Unavailable
    );
    write_live_turn(ws.path(), STEM, &t, plain_journal_message)
        .await
        .unwrap();
    assert_eq!(
        shadow_read_compare(ws.path(), STEM, &t, plain_journal_message).await,
        ShadowReadOutcome::Match { messages: 2 }
    );
    let changed = seed(ws.path(), "different");
    assert_eq!(
        shadow_read_compare(ws.path(), STEM, &changed, plain_journal_message).await,
        ShadowReadOutcome::Divergence {
            legacy: 2,
            shadow: 2,
            first_diff: Some(1)
        }
    );
}

#[test]
fn host_projector_seam_controls_journal_content() {
    let ws = TempDir::new().unwrap();
    let t = seed(ws.path(), "done");
    let project: JournalProjector = |m| {
        let mut rec = plain_journal_message(m);
        rec.extra_metadata = Some(json!({"host": true}));
        rec
    };
    let recs = journal_messages(&t, project);
    assert!(
        recs.iter()
            .all(|r| r.extra_metadata == Some(json!({"host": true})))
    );
}

/// Persisted shapes are wire contracts: pin them as literal JSON.
#[test]
fn persisted_record_shapes_are_unchanged() {
    let msg = JournalMessage {
        id: Some("c1".into()),
        role: "tool".into(),
        content: "x".into(),
        extra_metadata: Some(json!({"k": 1})),
    };
    assert_eq!(
        serde_json::to_value(&msg).unwrap(),
        json!({"id":"c1","role":"tool","content":"x","extra_metadata":{"k":1}})
    );
    let bare = JournalMessage {
        id: None,
        role: "user".into(),
        content: "y".into(),
        extra_metadata: None,
    };
    assert_eq!(
        serde_json::to_value(&bare).unwrap(),
        json!({"role":"user","content":"y"})
    );

    let ledger = ItemLedgerRecord {
        version: 1,
        session_key: "s".into(),
        source: "session_raw/s.jsonl".into(),
        size: 3,
        mtime_ms: 4,
        stream: "session.s.messages".into(),
        messages: 2,
        imported_at: "2024-01-01T00:00:00Z".into(),
    };
    assert_eq!(
        serde_json::to_value(&ledger).unwrap(),
        json!({"version":1,"session_key":"s","source":"session_raw/s.jsonl","size":3,"mtime_ms":4,
               "stream":"session.s.messages","messages":2,"imported_at":"2024-01-01T00:00:00Z"})
    );
}

#[test]
fn ledger_key_is_lowercase_hex_sha256() {
    // sha256("abc")
    assert_eq!(
        super::ops::ledger_key("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
