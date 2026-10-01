//! Tests for the orphaned-run startup sweep.

use super::*;
use crate::events::HarnessRunStatus;
use crate::ids::{ComponentId, ExecutionStatus, HarnessPhase};
use crate::observability::{FileStatusStore, mint_run_id};
use crate::store::FileStore;

fn store(tag: &str) -> (FileStatusStore, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("ta-reaper-{tag}-{}", uuid::Uuid::new_v4()));
    (FileStatusStore::new(FileStore::new(&root)), root)
}

async fn seed_status(store: &FileStatusStore, kind: ExecutionStatus) -> String {
    let run_id = mint_run_id();
    let mut status =
        HarnessRunStatus::new(run_id.clone(), ComponentId::new("mock-model".to_string()));
    match kind {
        ExecutionStatus::Pending => {}
        ExecutionStatus::Running => status.mark_running(HarnessPhase::Model),
        ExecutionStatus::Interrupted => status.mark_interrupted(),
        other => panic!("seed_status only seeds non-terminal states, got {other:?}"),
    }
    store.put_status(status).await.unwrap();
    run_id.as_str().to_string()
}

#[tokio::test]
async fn reap_cancels_every_active_run_and_spares_terminal_ones() {
    let (store, root) = store("all");
    let pending = seed_status(&store, ExecutionStatus::Pending).await;
    let running = seed_status(&store, ExecutionStatus::Running).await;
    let interrupted = seed_status(&store, ExecutionStatus::Interrupted).await;

    let done = mint_run_id();
    let mut done_status =
        HarnessRunStatus::new(done.clone(), ComponentId::new("mock-model".to_string()));
    done_status.mark_running(HarnessPhase::Model);
    done_status.mark_completed();
    store.put_status(done_status).await.unwrap();

    assert_eq!(reap_orphaned_runs(&store).await, 3);

    for run_id in [&pending, &running, &interrupted] {
        let status = store.get_status(run_id).await.unwrap().expect("present");
        assert_eq!(status.status, ExecutionStatus::Cancelled);
        assert_eq!(status.current_phase, HarnessPhase::Done);
        assert_eq!(status.error.as_deref(), Some(ORPHAN_REAP_REASON));
        assert!(status.ended_at.is_some());
    }
    let done_after = store.get_status(done.as_str()).await.unwrap().unwrap();
    assert_eq!(done_after.status, ExecutionStatus::Completed);
    assert!(done_after.error.is_none());

    assert!(store.list_active().await.unwrap().is_empty());
    assert_eq!(reap_orphaned_runs(&store).await, 0, "idempotent");
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn reap_on_empty_store_is_a_noop() {
    let (store, root) = store("empty");
    assert_eq!(reap_orphaned_runs(&store).await, 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn mark_cancelled_sets_terminal_fields() {
    let mut status = HarnessRunStatus::new(mint_run_id(), ComponentId::new("m".to_string()));
    status.mark_running(HarnessPhase::Model);
    status.mark_cancelled("why");
    assert_eq!(status.status, ExecutionStatus::Cancelled);
    assert_eq!(status.current_phase, HarnessPhase::Done);
    assert_eq!(status.error.as_deref(), Some("why"));
    assert!(status.ended_at.is_some());
}
