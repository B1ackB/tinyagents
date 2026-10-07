//! Durable, file-backed [`HarnessStatusStore`] plus the run-id and redaction
//! helpers a host needs to attach a durable journal to a run.
//!
//! The crate ships [`super::types::InMemoryStatusStore`], which does not survive a process
//! restart. [`FileStatusStore`] overwrites one `run_status/<run_id>.json` file
//! per run under a [`FileStore`] and answers the lineage and liveness queries
//! by enumerating that namespace, which is what lets a supervisor reattach
//! after the UI died and find every active descendant of a root run.

use async_trait::async_trait;

use super::HarnessStatusStore;
use crate::error::{Result, TinyAgentsError};
use crate::events::HarnessRunStatus;
use crate::ids::{ExecutionStatus, RunId};
use std::sync::Arc;

use crate::store::{FileStore, Store};

/// KV namespace the durable per-run [`HarnessRunStatus`] snapshots live under
/// (`<kv root>/run_status/<run_id>.json`). Slash-free so it round-trips the
/// [`FileStore`] name sanitizer. Part of the on-disk format.
pub const STATUS_NS: &str = "run_status";

fn status_key(run_id: &str) -> String {
    let safe = is_safe_status_key(run_id);
    if safe && !run_id.starts_with("x-") {
        return run_id.to_string();
    }
    let mut encoded = String::from("x-");
    for byte in run_id.as_bytes() {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}

fn is_safe_status_key(run_id: &str) -> bool {
    !run_id.is_empty()
        && !run_id.bytes().all(|byte| byte == b'.')
        && run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

/// Name fragments that mark an environment variable as credential material.
const SECRET_NAME_MARKERS: [&str; 7] = [
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "BEARER",
];

/// Shortest value (in bytes, after trimming) worth masking.
const MIN_SECRET_LEN: usize = 8;

/// Mints a fresh, slash-free, process-unique run id (`run.<32-hex>`).
///
/// One id serves a whole turn: the `EventSink::with_stream_id` prefix (so
/// persisted `event_id`s are the restart-stable `{run_id}-evt-{offset}`), the
/// journal stream key, and the status-store key. The `simple()` uuid form (no
/// hyphens) stays inside [`FileStore`]'s allowed-character set.
pub fn mint_run_id() -> RunId {
    RunId::new(format!("run.{}", uuid::Uuid::new_v4().simple()))
}

/// Values of process environment variables whose name looks like a secret
/// (contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL` or
/// `BEARER`, case-insensitively) and whose trimmed value is at least eight
/// bytes long.
///
/// Seed a [`super::RedactingSink`] with this so an API key or bearer token
/// echoed into model text, a tool argument fragment or an error string is never
/// persisted in the clear. Short values are left alone so unrelated config is
/// not masked.
pub fn process_env_secrets() -> Vec<String> {
    secrets_from_vars(std::env::vars())
}

/// Pure core of [`process_env_secrets`], over an explicit variable list.
pub fn secrets_from_vars(vars: impl IntoIterator<Item = (String, String)>) -> Vec<String> {
    let mut secrets = Vec::new();
    for (name, value) in vars {
        let upper = name.to_ascii_uppercase();
        if SECRET_NAME_MARKERS.iter().any(|m| upper.contains(m))
            && value.trim().len() >= MIN_SECRET_LEN
        {
            secrets.push(value);
        }
    }
    tracing::debug!(
        count = secrets.len(),
        "[journal] redaction policy seeded with secret value(s)"
    );
    secrets
}

/// A durable [`HarnessStatusStore`] backed by a [`FileStore`] KV namespace.
///
/// Each snapshot is compact (ids, phase, counters, timestamps, error), never
/// prompts or payloads.
pub struct FileStatusStore {
    kv: Arc<dyn Store>,
}

impl FileStatusStore {
    /// Wrap a kv store as a durable status store.
    pub fn new(kv: FileStore) -> Self {
        Self::over(Arc::new(kv))
    }

    /// Wrap any [`Store`] — a host's own database included — as a durable
    /// status store. Records keep the [`FileStore`]-safe key encoding, so
    /// they read back identically whichever store holds them.
    pub fn over(kv: Arc<dyn Store>) -> Self {
        Self { kv }
    }

    /// Enumerate every persisted status snapshot (best-effort per record: a
    /// corrupt or legacy record is skipped, never fatal). Malformed JSON and a
    /// schema mismatch are both per-record decode failures; a filesystem read
    /// error is still returned.
    async fn all(&self) -> Result<Vec<HarnessRunStatus>> {
        let keys = self.kv.list(STATUS_NS).await?;
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            let value = match self.kv.get(STATUS_NS, &key).await {
                Ok(value) => value,
                Err(TinyAgentsError::Serialization(err)) => {
                    tracing::debug!("[journal] skipping malformed run status key={key} err={err}");
                    continue;
                }
                Err(err) => return Err(err),
            };
            if let Some(value) = value {
                match serde_json::from_value::<HarnessRunStatus>(value) {
                    Ok(status) => out.push(status),
                    Err(err) => {
                        tracing::debug!(
                            "[journal] skipping undecodable run status key={key} err={err}"
                        )
                    }
                }
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl HarnessStatusStore for FileStatusStore {
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()> {
        let key = status_key(status.run_id.as_str());
        let run_id = status.run_id.as_str();
        let legacy_value = if key != run_id {
            self.kv.get(STATUS_NS, run_id).await?
        } else {
            None
        };
        let value = serde_json::to_value(&status)?;
        self.kv.put(STATUS_NS, &key, value).await?;
        if legacy_value
            .as_ref()
            .and_then(|value| serde_json::from_value::<HarnessRunStatus>(value.clone()).ok())
            .is_some_and(|existing| existing.run_id == status.run_id)
        {
            self.kv.delete(STATUS_NS, run_id).await?;
        }
        Ok(())
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<HarnessRunStatus>> {
        let key = status_key(run_id);
        let value = match self.kv.get(STATUS_NS, &key).await? {
            Some(value) => Some(value),
            None if key != run_id => self.kv.get(STATUS_NS, run_id).await?,
            None => None,
        };
        match value {
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<HarnessRunStatus>> {
        Ok(self
            .all()
            .await?
            .into_iter()
            .filter(|s| {
                s.thread_id
                    .as_ref()
                    .is_some_and(|t| t.as_str() == thread_id)
            })
            .collect())
    }

    async fn list_by_root(&self, root_run_id: &str) -> Result<Vec<HarnessRunStatus>> {
        Ok(self
            .all()
            .await?
            .into_iter()
            .filter(|s| s.root_run_id.as_str() == root_run_id)
            .collect())
    }

    async fn list_active(&self) -> Result<Vec<HarnessRunStatus>> {
        Ok(self.all().await?.into_iter().filter(is_active).collect())
    }
}

/// Whether a run is still live (`Pending`, `Running` or `Interrupted`).
pub fn is_active(status: &HarnessRunStatus) -> bool {
    matches!(
        status.status,
        ExecutionStatus::Pending | ExecutionStatus::Running | ExecutionStatus::Interrupted
    )
}

#[cfg(test)]
#[path = "file_status_tests.rs"]
mod tests;
