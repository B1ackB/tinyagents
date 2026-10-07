use super::*;
use crate::transcript::{FileTranscriptLocator, TranscriptHistory, TranscriptMeta, TranscriptRead};

/// Whether `lock_session_turn` for `session` would have to wait right now.
///
/// No clock: an uncontended tokio mutex is acquired on its first poll, so a
/// lock still pending after that poll is held by someone else.
async fn is_held(locator: &dyn TranscriptLocator, session: &SessionRef) -> bool {
    tokio::select! {
        biased;
        _guard = lock_session_turn(locator, session) => false,
        () = tokio::task::yield_now() => true,
    }
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
async fn missing_workspace_paths_are_normalized_for_locking() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("not-created");
    let equivalent = missing.join("child").join("..");
    let session = SessionRef::scoped("thread-1", "agent");

    let first = FileTranscriptLocator::new(&missing);
    let second = FileTranscriptLocator::new(&equivalent);
    let _guard = lock_session_turn(&first, &session).await.unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            lock_session_turn(&second, &session),
        )
        .await
        .is_err()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn missing_workspace_below_a_symlink_is_normalized_for_locking() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("real");
    std::fs::create_dir(&real).unwrap();
    let link = dir.path().join("link");
    symlink(&real, &link).unwrap();
    let session = SessionRef::scoped("thread-1", "agent");
    let first = FileTranscriptLocator::new(real.join("not-created"));
    let second = FileTranscriptLocator::new(link.join("not-created"));

    let _guard = lock_session_turn(&first, &session).await.unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            lock_session_turn(&second, &session),
        )
        .await
        .is_err()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_workspace_symlink_is_normalized_for_locking() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("future");
    let link = dir.path().join("link");
    symlink(&real, &link).unwrap();
    let session = SessionRef::scoped("thread-1", "agent");

    let _guard = lock_session_turn(&FileTranscriptLocator::new(&real), &session)
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            lock_session_turn(&FileTranscriptLocator::new(&link), &session),
        )
        .await
        .is_err()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn parent_components_are_resolved_after_symlinks() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("real");
    std::fs::create_dir_all(real.join("nested")).unwrap();
    let link = dir.path().join("link");
    symlink(real.join("nested"), &link).unwrap();
    let session = SessionRef::scoped("thread-1", "agent");

    let _guard = lock_session_turn(
        &FileTranscriptLocator::new(real.join("workspace")),
        &session,
    )
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            lock_session_turn(
                &FileTranscriptLocator::new(link.join("..").join("workspace")),
                &session,
            ),
        )
        .await
        .is_err()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn absolute_parent_components_clamp_at_root() {
    let session = SessionRef::scoped("thread-1", "agent");
    let _guard = lock_session_turn(
        &FileTranscriptLocator::new(std::path::Path::new("/../tmp")),
        &session,
    )
    .await
    .unwrap();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            lock_session_turn(&FileTranscriptLocator::new("/tmp"), &session),
        )
        .await
        .is_err()
    );
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
    fn latest_for_agent(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        None
    }
    fn root_for_thread(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        None
    }
    fn open_stem(&self, _: &str, _: TranscriptMeta) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
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

#[cfg(unix)]
#[tokio::test]
async fn workspaces_sharing_a_symlinked_session_raw_share_the_lock() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let shared = dir.path().join("shared-raw");
    std::fs::create_dir(&shared).unwrap();
    let first_ws = dir.path().join("ws-a");
    let second_ws = dir.path().join("ws-b");
    std::fs::create_dir(&first_ws).unwrap();
    std::fs::create_dir(&second_ws).unwrap();
    symlink(&shared, first_ws.join("session_raw")).unwrap();
    symlink(&shared, second_ws.join("session_raw")).unwrap();
    let session = SessionRef::scoped("thread-1", "agent");
    let first = FileTranscriptLocator::new(&first_ws);
    let second = FileTranscriptLocator::new(&second_ws);

    let _guard = lock_session_turn(&first, &session).await.unwrap();
    assert!(is_held(&second, &session).await);
}
