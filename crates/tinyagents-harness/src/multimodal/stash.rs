//! On-disk attachment stash: content-addressed image files kept out of band so
//! conversations persist a compact `[Image: … #att:<id>]` placeholder instead of
//! a multi-megabyte `data:` URI.
//!
//! [`AttachmentStash`] owns the mechanism (atomic dedup'd writes, size-cap
//! eviction, TTL sweep, id-to-path index, managed-path check). The host owns the
//! policy: which directory, how large, how long.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use super::{data_uri, mime};

/// A directory of content-addressed attachment files with a size cap and TTL.
#[derive(Debug, Clone)]
pub struct AttachmentStash {
    dir: PathBuf,
    max_bytes: u64,
    ttl: Duration,
    /// Serialises cap enforcement across clones of this stash, so two writers
    /// never evict from the same stale snapshot of the directory.
    evict_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
}

impl AttachmentStash {
    /// A stash rooted at `dir`. After each new write the oldest files (by
    /// mtime) are evicted until the total is back under `max_bytes`;
    /// [`Self::sweep_stale`] removes files older than `ttl`.
    pub fn new(dir: PathBuf, max_bytes: u64, ttl: Duration) -> Self {
        Self {
            dir,
            max_bytes,
            ttl,
            evict_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// The stash directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Persist a canonical image data URI to `<dir>/<id>.<ext>`, content-addressed
    /// by `id`. Atomic (temp file + rename); deduped (an existing target only
    /// has its mtime refreshed). Returns the written path. After a fresh write,
    /// enforces the size cap.
    ///
    /// `id` must be a single filename component of ASCII letters, digits, `-`
    /// and `_`, so it can never name a path outside the stash. An attachment
    /// larger than the whole cap is rejected before anything is written or
    /// evicted.
    pub async fn write(&self, id: &str, data_uri_text: &str) -> anyhow::Result<PathBuf> {
        if !is_safe_id(id) {
            anyhow::bail!("invalid attachment id: must be non-empty [A-Za-z0-9_-]");
        }
        let parsed = data_uri::parse_data_uri(data_uri_text)
            .map_err(|reason| anyhow::anyhow!("cannot decode stashed data URI: {reason}"))?;
        let size = parsed.bytes.len() as u64;
        if size > self.max_bytes {
            anyhow::bail!(
                "attachment of {size} bytes exceeds the stash cap of {} bytes",
                self.max_bytes
            );
        }
        let ext = mime::image_ext_from_mime(&parsed.mime).unwrap_or("img");
        tokio::fs::create_dir_all(&self.dir).await?;
        let final_path = self.dir.join(format!("{id}.{ext}"));
        if tokio::fs::try_exists(&final_path).await.unwrap_or(false) {
            // Content-addressing deduplicates identical images, but a new
            // message can legitimately reference a file whose previous reference
            // is older than the TTL. Refresh its mtime so the immediately
            // following sweep cannot reclaim an attachment that was just reused.
            let touch_path = final_path.clone();
            tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                std::fs::File::open(touch_path)?
                    .set_times(std::fs::FileTimes::new().set_modified(SystemTime::now()))
            })
            .await??;
            return Ok(final_path);
        }
        // A distinct temp file per write: two concurrent writes of one id must
        // not share (and then race to rename) a single source path.
        static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self
            .dir
            .join(format!(".{id}.{ext}.{}.{seq}.tmp", std::process::id()));
        let published = async {
            tokio::fs::write(&tmp_path, &parsed.bytes).await?;
            tokio::fs::rename(&tmp_path, &final_path).await
        }
        .await;
        if let Err(error) = published {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(error.into());
        }
        self.enforce_cap().await;
        // Never report a path that is gone: a concurrent writer's eviction
        // (another process sharing the directory) may have reclaimed it.
        if !tokio::fs::try_exists(&final_path).await.unwrap_or(false) {
            anyhow::bail!("attachment was evicted by the stash size cap before it could be used");
        }
        Ok(final_path)
    }

    /// Build an `id -> path` index from a single read of the directory. Skips
    /// in-flight `.tmp` files. Sync so it can serve a sync rehydrate path.
    pub fn build_index(&self) -> HashMap<String, PathBuf> {
        let mut map = HashMap::new();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return map;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue; // skip `.<id>.<ext>.tmp` in-flight writes
            }
            if let Some(stem) = name.split('.').next()
                && !stem.is_empty()
            {
                map.insert(stem.to_string(), entry.path());
            }
        }
        map
    }

    /// Evict oldest files (by mtime) until the directory is under the size cap.
    /// Best-effort.
    pub async fn enforce_cap(&self) {
        let _evicting = self.evict_lock.lock().await;
        let mut files: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
        let mut total: u64 = 0;
        let Ok(mut rd) = tokio::fs::read_dir(&self.dir).await else {
            return;
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            // Skip in-flight `.<id>.<ext>.tmp` writes so concurrent atomic
            // writes aren't evicted out from under a rename.
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if let Ok(meta) = entry.metadata().await
                && meta.is_file()
            {
                let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                total = total.saturating_add(meta.len());
                files.push((entry.path(), mtime, meta.len()));
            }
        }
        if total <= self.max_bytes {
            return;
        }
        files.sort_by_key(|(_, mtime, _)| *mtime); // oldest first
        for (path, _, len) in files {
            if total <= self.max_bytes {
                break;
            }
            let removed = match tokio::fs::remove_file(&path).await {
                Ok(()) => true,
                // Already gone (removed outside this stash): it no longer
                // counts toward the total either.
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
                Err(_) => false,
            };
            if removed {
                total = total.saturating_sub(len);
                tracing::debug!(
                    target: "multimodal",
                    path = %path.display(),
                    "[multimodal::images][gc] evicted attachment over size cap"
                );
            }
        }
    }

    /// Delete files older than the TTL. Best-effort; returns reclaimed bytes.
    pub async fn sweep_stale(&self) -> u64 {
        let Ok(mut rd) = tokio::fs::read_dir(&self.dir).await else {
            return 0;
        };
        let now = SystemTime::now();
        let mut reclaimed = 0u64;
        while let Ok(Some(entry)) = rd.next_entry().await {
            let Ok(meta) = entry.metadata().await else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            let age = now.duration_since(mtime).unwrap_or(Duration::ZERO);
            if age > self.ttl && tokio::fs::remove_file(entry.path()).await.is_ok() {
                reclaimed = reclaimed.saturating_add(meta.len());
            }
        }
        if reclaimed > 0 {
            tracing::info!(
                target: "multimodal",
                reclaimed_bytes = reclaimed,
                "[multimodal::images][gc] startup sweep removed stale attachments"
            );
        }
        reclaimed
    }

    /// Return the canonical path when `path` resolves inside the stash.
    /// Callers should read the returned path, not the spelling they were given,
    /// so a checked path is what gets opened.
    pub fn managed_path(&self, path: &str) -> Option<PathBuf> {
        let candidate = Path::new(path).canonicalize().ok()?;
        let root = self.dir.canonicalize().ok()?;
        candidate.starts_with(root).then_some(candidate)
    }
}

/// Whether `id` is a single, non-empty filename component made only of ASCII
/// letters, digits, `-` and `_`.
fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
#[path = "stash_tests.rs"]
mod tests;
