use super::*;
use crate::testkit::conformance::session_store_conformance;
use crate::transcript::FileTranscriptLocator;
use tinyagents_harness::store::{FileStore, JsonlAppendStore};

/// The on-disk building blocks, assembled the way a desktop host's provider
/// assembles them: everything under one workspace, shared by every agent.
struct FileStores {
    workspace: std::path::PathBuf,
}

impl SessionStoreProvider for FileStores {
    fn for_agent(&self, _agent_id: &str) -> AgentStores {
        AgentStores {
            transcripts: Arc::new(FileTranscriptLocator::new(&self.workspace)),
            turn_states: Arc::new(TurnStateStore::new(self.workspace.clone())),
            kv: Arc::new(FileStore::new(self.workspace.join("kv"))),
            journal: Arc::new(JsonlAppendStore::new(self.workspace.join("journal"))),
        }
    }
}

#[tokio::test]
async fn the_file_building_blocks_meet_the_session_store_contract() {
    let dir = tempfile::tempdir().unwrap();
    let provider = FileStores {
        workspace: dir.path().to_path_buf(),
    };
    session_store_conformance(&provider).await;
    assert!(provider.recover().is_ok(), "recovery defaults to a no-op");
    assert!(provider.destination_key().is_none());
    assert!(
        provider.workspace_dir().is_none(),
        "only a provider that says so is file-backed"
    );
}

#[test]
fn the_file_turn_state_store_is_a_turn_states() {
    let dir = tempfile::tempdir().unwrap();
    let turns: Arc<dyn TurnStates> = Arc::new(TurnStateStore::new(dir.path().to_path_buf()));
    let state = TurnState::started("t", "r", 4, "2026-01-01T00:00:00Z");
    turns.put(&state).unwrap();
    assert!(turns.put_unless_completed(&state).unwrap());
    assert_eq!(turns.get("t").unwrap().unwrap().request_id, "r");
    assert_eq!(turns.list().unwrap().len(), 1);
    assert_eq!(
        turns.mark_all_interrupted("2026-01-01T00:01:00Z").unwrap(),
        1
    );
    assert!(turns.delete("t").unwrap());
    assert_eq!(turns.clear_all().unwrap(), 0);
}
