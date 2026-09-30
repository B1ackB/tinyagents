//! Tests for the read-only replay helpers.

use std::sync::Arc;

use super::*;
use crate::events::{AgentEvent, EventSink, HarnessRunStatus};
use crate::ids::{ComponentId, HarnessPhase, ThreadId};
use crate::observability::{
    FanOutSink, FileStatusStore, JournalSink, StoreEventJournal, mint_run_id,
};
use crate::store::{FileStore, JsonlAppendStore};

fn root(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ta-replay-{tag}-{}", uuid::Uuid::new_v4()))
}

fn seed_run_events(journal_root: &std::path::Path, count: usize) -> String {
    let run_id = mint_run_id();
    let journal: Arc<dyn HarnessEventJournal> =
        Arc::new(StoreEventJournal::new(JsonlAppendStore::new(journal_root)));
    let sink = EventSink::with_stream_id(run_id.as_str());
    let journal_sink = Arc::new(JournalSink::new(journal, run_id.clone()));
    sink.subscribe(Arc::new(FanOutSink::new().with(journal_sink.clone())));
    for i in 0..count {
        sink.emit(AgentEvent::ToolStarted {
            call_id: format!("c{i}").into(),
            tool_name: format!("tool-{i}"),
            input: None,
        });
    }
    journal_sink.flush();
    run_id.as_str().to_string()
}

#[tokio::test]
async fn read_run_events_page_pages_and_drains() {
    let dir = root("page");
    let run_id = seed_run_events(&dir, 3);
    let journal = StoreEventJournal::new(JsonlAppendStore::new(&dir));

    let page1 = read_run_events_page(&journal, &run_id, 0, 2).await.unwrap();
    assert_eq!(page1.events.len(), 2);
    assert_eq!(page1.events[0].offset, 0);
    assert_eq!(page1.events[1].offset, 1);
    assert_eq!(page1.next_offset, Some(2));

    let page2 = read_run_events_page(&journal, &run_id, 2, 2).await.unwrap();
    assert_eq!(page2.events.len(), 1);
    assert_eq!(page2.next_offset, None);

    let exact = read_run_events_page(&journal, &run_id, 0, 3).await.unwrap();
    assert_eq!(exact.events.len(), 3);
    assert_eq!(exact.next_offset, None);

    let empty = read_run_events_page(&journal, "run.does-not-exist", 0, 10)
        .await
        .unwrap();
    assert!(empty.events.is_empty());
    assert_eq!(empty.next_offset, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn read_run_status_none_for_unknown_run() {
    let dir = root("status");
    let store = FileStatusStore::new(FileStore::new(&dir));
    assert!(read_run_status(&store, "run.nope").await.unwrap().is_none());
}

#[tokio::test]
async fn list_active_runs_returns_started_and_filters() {
    let dir = root("active");
    let store = FileStatusStore::new(FileStore::new(&dir));

    let run_a = mint_run_id();
    let mut a = HarnessRunStatus::new(run_a.clone(), ComponentId::new("m".to_string()))
        .with_thread(ThreadId::new("thread-A"));
    a.mark_running(HarnessPhase::Model);
    store.put_status(a).await.unwrap();

    let run_b = mint_run_id();
    let mut b = HarnessRunStatus::new(run_b, ComponentId::new("m".to_string()))
        .with_thread(ThreadId::new("thread-B"));
    b.mark_running(HarnessPhase::Model);
    b.mark_completed();
    store.put_status(b).await.unwrap();

    let active = list_active_runs(&store, None, None).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].run_id.as_str(), run_a.as_str());
    assert_eq!(
        list_active_runs(&store, Some("thread-A"), None)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        list_active_runs(&store, Some("thread-B"), None)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        list_active_runs(&store, Some("nope"), None)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        list_active_runs(&store, None, Some(run_a.as_str()))
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        list_active_runs(&store, Some("thread-A"), Some("other-root"))
            .await
            .unwrap()
            .is_empty()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
