//! Live dual-write of new session turns into the TinyAgents store, and the
//! store-backed shadow read that checks it against the legacy transcript.
//!
//! Additive and best-effort: the legacy `session_raw/*.jsonl` transcript stays
//! the primary and authoritative writer; [`write_live_turn`] mirrors each
//! *already-persisted* turn into the same store layout the importer produces
//! (`{workspace}/tinyagents_store/{kv,journal}`), reusing [`super::convert`]
//! normalization so live and imported records are shape-identical. Whether the
//! mirror or the shadow read runs at all is the host's decision (config flag,
//! kill-switch env var); this module only does the work. A store-write failure
//! must never fail or alter a chat turn: the caller treats every error as
//! non-fatal (log + swallow).

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use tinyagents_harness::store::{AppendStore, Store};

use crate::transcript::SessionTranscript;

use super::convert::{
    JournalProjector, build_descriptor, effective_thread_id, journal_messages, sanitize_store_name,
    stream_name,
};
use super::ops::{SessionStores, open_session_stores, rewrite_journal_stream};
use super::types::{DescriptorSource, JournalMessage, NS_SESSIONS, SessionDescriptor};

static LIVE_REWRITE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

/// Mirror one completed turn's transcript into the TinyAgents store.
///
/// Best-effort: the caller must treat any returned error as non-fatal (log and
/// swallow) — it never fails or alters the legacy chat turn.
///
/// Mirrors the importer's **full-rewrite** semantics: the legacy JSONL
/// transcript is rewritten in full on every turn (not appended), and the
/// `JsonlAppendStore` has no truncate, so the journal stream file is dropped and
/// re-appended each turn. This keeps the store stream shape-identical to an
/// import of the final JSONL. The descriptor is upserted in `NS_SESSIONS`
/// exactly as the importer would, reusing [`build_descriptor`].
pub async fn write_live_turn(
    workspace: &Path,
    session_key: &str,
    transcript: &SessionTranscript,
    project: JournalProjector,
) -> Result<()> {
    let _rewrite_guard = LIVE_REWRITE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    tracing::debug!(
        "[session-store] dual-write enter stem={session_key} workspace={} messages={}",
        workspace.display(),
        transcript.messages.len()
    );

    let SessionStores {
        kv,
        journal,
        journal_root,
    } = open_session_stores(workspace);

    let stream = stream_name(session_key);

    let values = journal_messages(transcript, project)
        .into_iter()
        .enumerate()
        .map(|(idx, record)| {
            serde_json::to_value(record).with_context(|| format!("serialize live message {idx}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let message_count = values.len();
    rewrite_journal_stream(&journal, &journal_root, &stream, values)
        .await
        .with_context(|| format!("rewrite live journal stream {stream}"))?;

    // Descriptor: same projection the importer uses. No run-ledger join here
    // (live turns have no `agent_runs` link yet) and zero warnings; the source
    // pointer records the workspace-relative JSONL twin.
    let (thread_id, synthesized) =
        effective_thread_id(session_key, transcript.meta.thread_id.as_deref());
    let mut descriptor = build_descriptor(
        session_key,
        transcript,
        thread_id,
        synthesized,
        Vec::new(),
        DescriptorSource {
            jsonl: Some(format!("session_raw/{session_key}.jsonl")),
            md: None,
        },
        chrono::Utc::now().to_rfc3339(),
        0,
    );
    let descriptor_key = sanitize_store_name(session_key);
    if let Ok(Some(existing)) = kv.get(NS_SESSIONS, &descriptor_key).await
        && let Ok(existing) = serde_json::from_value::<SessionDescriptor>(existing)
    {
        if existing.source.jsonl.is_some() {
            descriptor.source.jsonl = existing.source.jsonl;
        }
        if existing.source.md.is_some() {
            descriptor.source.md = existing.source.md;
        }
    }
    let descriptor_value =
        serde_json::to_value(&descriptor).context("serialize live session descriptor")?;
    kv.put(NS_SESSIONS, &descriptor_key, descriptor_value)
        .await
        .context("descriptor write failed")?;

    tracing::debug!(
        "[session-store] dual-write exit stem={session_key} stream={stream} messages={message_count}"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Store-backed SHADOW READ (issue #4249, sessions 04.2 phase 2)
//
// Beside the legacy authoritative transcript reader
// (`session/turn/session_io.rs` → `try_load_session_transcript`), read the
// same session's messages back from the crate journal store, normalize both
// sides through the same `convert` machinery the dual-write uses, compare, and
// log divergence. Legacy stays authoritative: this observes + logs only and
// never affects, fails, or slows the authoritative read.
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of one shadow-read comparison. Returned for tests/observability;
/// the compact divergence summary is also logged (`[session_shadow_read]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowReadOutcome {
    /// The store stream rendered exactly the legacy transcript's messages.
    Match { messages: usize },
    /// The store stream diverged from the legacy render. Carries only compact
    /// counts + the first differing index — never message bodies (PII).
    Divergence {
        legacy: usize,
        shadow: usize,
        first_diff: Option<usize>,
    },
    /// No shadow available: the store read errored, or the stream is
    /// empty/absent for a non-empty legacy transcript (e.g. dual-write was off
    /// when this session was written). Treated as non-divergent — the legacy
    /// read is authoritative regardless.
    Unavailable,
}

/// Read a session's messages back from the crate journal store
/// (`{workspace}/tinyagents_store/journal`, stream `session.{stem}.messages`)
/// as normalized [`JournalMessage`]s — the same shape the importer and live
/// dual-write write. A missing stream yields an empty vec (not an error).
async fn read_shadow_messages(workspace: &Path, session_key: &str) -> Result<Vec<JournalMessage>> {
    let _rewrite_guard = LIVE_REWRITE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await;
    let SessionStores { journal, .. } = open_session_stores(workspace);
    let stream = stream_name(session_key);
    let records = journal
        .read_from(&stream, 0)
        .await
        .with_context(|| format!("shadow read of stream {stream}"))?;
    let mut out = Vec::with_capacity(records.len());
    for (offset, value) in records {
        let msg: JournalMessage = serde_json::from_value(value)
            .with_context(|| format!("shadow record shape at offset {offset}"))?;
        out.push(msg);
    }
    Ok(out)
}

/// Shadow-read the given session back from the store and compare it against the
/// legacy transcript, logging divergence. Legacy stays authoritative — the
/// caller ignores the returned outcome for control flow (it exists for tests /
/// observability). Best-effort: any store-read error is logged at debug and
/// reported as [`ShadowReadOutcome::Unavailable`]; it never breaks or slows the
/// authoritative read.
///
/// Both sides are normalized through the importer's `convert` machinery
/// ([`journal_messages`]) so live/legacy renders are directly comparable, then
/// compared by message count and normalized content. On mismatch a **compact**
/// summary (counts + first differing index) is warn-logged; message bodies are
/// never emitted (PII).
pub async fn shadow_read_compare(
    workspace: &Path,
    session_key: &str,
    legacy: &SessionTranscript,
    project: JournalProjector,
) -> ShadowReadOutcome {
    let expected = journal_messages(legacy, project);
    tracing::debug!(
        "[session_shadow_read] enter stem={session_key} workspace={} legacy_messages={}",
        workspace.display(),
        expected.len()
    );

    let shadow = match read_shadow_messages(workspace, session_key).await {
        Ok(v) => v,
        Err(err) => {
            tracing::debug!(
                "[session_shadow_read] store read error stem={session_key}: {err:#} — no shadow available"
            );
            return ShadowReadOutcome::Unavailable;
        }
    };

    // Empty/absent store stream against a non-empty legacy transcript: the
    // session simply was not mirrored (dual-write off when it was written).
    // Treat as "no shadow" rather than a spurious divergence.
    if shadow.is_empty() && !expected.is_empty() {
        tracing::debug!(
            "[session_shadow_read] no store stream stem={session_key} legacy_messages={} — no shadow available",
            expected.len()
        );
        return ShadowReadOutcome::Unavailable;
    }

    if shadow == expected {
        tracing::debug!(
            "[session_shadow_read] parity OK stem={session_key} messages={}",
            expected.len()
        );
        return ShadowReadOutcome::Match {
            messages: expected.len(),
        };
    }

    // Divergence: first index where the two normalized renders differ, or the
    // shorter length when one is a strict prefix of the other. Compact only.
    let first_diff = expected
        .iter()
        .zip(shadow.iter())
        .position(|(a, b)| a != b)
        .or_else(|| (expected.len() != shadow.len()).then(|| expected.len().min(shadow.len())));
    tracing::warn!(
        "[session_shadow_read] DIVERGENCE stem={session_key} legacy_count={} shadow_count={} first_diff={first_diff:?}",
        expected.len(),
        shadow.len()
    );
    ShadowReadOutcome::Divergence {
        legacy: expected.len(),
        shadow: shadow.len(),
        first_diff,
    }
}
