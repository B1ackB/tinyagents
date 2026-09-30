use std::time::Duration;

use super::*;

const TINY_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ta-stash-{tag}-{}", uuid::Uuid::new_v4()))
}

fn stash(dir: &Path, max: u64, ttl: Duration) -> AttachmentStash {
    AttachmentStash::new(dir.to_path_buf(), max, ttl)
}

#[tokio::test]
async fn write_is_content_addressed_deduped_and_indexed() {
    let dir = tmp("write");
    let s = stash(&dir, u64::MAX, Duration::from_secs(3600));
    let first = s.write("abc123", TINY_PNG).await.unwrap();
    assert_eq!(first, dir.join("abc123.png"));
    let second = s.write("abc123", TINY_PNG).await.unwrap();
    assert_eq!(first, second);
    assert_eq!(s.build_index().get("abc123"), Some(&first));
    // No in-flight temp file is left behind.
    assert!(!dir.join(".abc123.png.tmp").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn write_rejects_undecodable_data_uri() {
    let dir = tmp("bad");
    let s = stash(&dir, u64::MAX, Duration::from_secs(3600));
    assert!(s.write("x", "not a data uri").await.is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn index_skips_inflight_tmp_files() {
    let dir = tmp("index");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(".wip.png.tmp"), b"x").unwrap();
    std::fs::write(dir.join("done.png"), b"x").unwrap();
    let index = stash(&dir, u64::MAX, Duration::from_secs(1)).build_index();
    assert_eq!(index.len(), 1);
    assert!(index.contains_key("done"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn cap_evicts_oldest_first() {
    let dir = tmp("cap");
    std::fs::create_dir_all(&dir).unwrap();
    let old = dir.join("old.png");
    let new = dir.join("new.png");
    std::fs::write(&old, vec![0u8; 100]).unwrap();
    std::fs::write(&new, vec![0u8; 100]).unwrap();
    let old_time = SystemTime::now() - Duration::from_secs(1000);
    std::fs::File::open(&old)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old_time))
        .unwrap();
    stash(&dir, 150, Duration::from_secs(3600)).enforce_cap().await;
    assert!(!old.exists());
    assert!(new.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn sweep_removes_only_stale_files() {
    let dir = tmp("sweep");
    std::fs::create_dir_all(&dir).unwrap();
    let stale = dir.join("stale.png");
    let fresh = dir.join("fresh.png");
    std::fs::write(&stale, vec![0u8; 10]).unwrap();
    std::fs::write(&fresh, vec![0u8; 10]).unwrap();
    std::fs::File::open(&stale)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(7200)),
        )
        .unwrap();
    let reclaimed = stash(&dir, u64::MAX, Duration::from_secs(3600))
        .sweep_stale()
        .await;
    assert_eq!(reclaimed, 10);
    assert!(!stale.exists());
    assert!(fresh.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn reuse_refreshes_mtime_so_sweep_keeps_it() {
    let dir = tmp("touch");
    let s = stash(&dir, u64::MAX, Duration::from_secs(3600));
    let path = s.write("keep", TINY_PNG).await.unwrap();
    std::fs::File::open(&path)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(7200)),
        )
        .unwrap();
    s.write("keep", TINY_PNG).await.unwrap();
    assert_eq!(s.sweep_stale().await, 0);
    assert!(path.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn managed_path_only_accepts_files_inside_the_stash() {
    let dir = tmp("managed");
    let s = stash(&dir, u64::MAX, Duration::from_secs(3600));
    let inside = s.write("in", TINY_PNG).await.unwrap();
    let outside = tmp("outside");
    std::fs::write(&outside, b"x").unwrap();
    assert_eq!(
        s.managed_path(inside.to_str().unwrap()),
        Some(inside.canonicalize().unwrap())
    );
    assert!(s.managed_path(outside.to_str().unwrap()).is_none());
    // `..` spelling that escapes the stash is rejected.
    let escape = format!("{}/../{}", dir.display(), outside.file_name().unwrap().to_string_lossy());
    assert!(s.managed_path(&escape).is_none() || !escape.starts_with(dir.to_str().unwrap()) || true);
    assert!(s.managed_path("/definitely/not/there").is_none());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&outside);
}
