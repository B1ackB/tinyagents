use super::*;
use crate::transcript::FileTranscriptLocator;
use std::time::Duration;

/// Whether `lock_session_turn` for `session` would have to wait right now.
async fn is_held(locator: &dyn TranscriptLocator, session: &SessionRef) -> bool {
    tokio::time::timeout(
        Duration::from_millis(20),
        lock_session_turn(locator, session),
    )
    .await
    .is_err()
}

#[tokio::test]
async fn a_held_turn_lock_blocks_the_same_session_until_released() {
    let dir = tempfile::tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "agent");

    let guard = lock_session_turn(&locator, &session).await.unwrap();
    assert!(is_held(&locator, &session).await);

    drop(guard);
    assert!(!is_held(&locator, &session).await);
}

#[tokio::test]
async fn every_generation_and_every_locator_over_one_workspace_share_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let session = SessionRef::scoped("thread-1", "agent");
    let _guard = lock_session_turn(&FileTranscriptLocator::new(dir.path()), &session)
        .await
        .unwrap();

    // A compaction moves the head on; it is still the same conversation. A
    // host also rebuilds its locator per turn rather than sharing one.
    let rebuilt = FileTranscriptLocator::new(dir.path());
    assert!(is_held(&rebuilt, &session.next_generation().next_generation()).await);
}

#[tokio::test]
async fn other_sessions_and_other_workspaces_are_independent() {
    let dir = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "agent");
    let _guard = lock_session_turn(&locator, &session).await.unwrap();

    assert!(!is_held(&locator, &SessionRef::scoped("thread-2", "agent")).await);
    assert!(!is_held(&locator, &SessionRef::scoped("thread-1", "other-agent")).await);
    assert!(!is_held(&FileTranscriptLocator::new(elsewhere.path()), &session).await);
}

/// A locator that cannot name its destination cannot be told apart from any
/// other, so there is nothing to key a lock on.
struct Anonymous;

impl TranscriptLocator for Anonymous {
    fn latest_for_agent(&self, _: &str) -> Option<Arc<dyn super::super::TranscriptRead>> {
        None
    }
    fn root_for_thread(&self, _: &str) -> Option<Arc<dyn super::super::TranscriptRead>> {
        None
    }
    fn open_stem(
        &self,
        _: &str,
        _: super::super::TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn super::super::TranscriptHistory>> {
        anyhow::bail!("not file backed")
    }
}

#[tokio::test]
async fn an_anonymous_locator_has_no_turn_lock() {
    assert!(
        lock_session_turn(&Anonymous, &SessionRef::scoped("thread-1", "agent"))
            .await
            .is_none()
    );
}
