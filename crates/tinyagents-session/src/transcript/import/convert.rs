//! Pure conversion helpers: stem lineage, stream naming, descriptor
//! assembly, and message-record projection.

use crate::transcript::{SessionTranscript, TranscriptMessage};

use super::types::{
    DescriptorImport, DescriptorSource, DescriptorUsage, IMPORT_VERSION, JournalMessage,
    SessionDescriptor,
};

/// Parent session key from the `__` stem chain.
///
/// Stems are `{parent_chain}__{unix_ts}_{agent_id}`; the parent key is
/// everything before the **last** `__`. Roots (no `__`) have no parent.
pub fn parent_session_key(stem: &str) -> Option<String> {
    stem.rfind("__").map(|idx| stem[..idx].to_string())
}

/// Encode a session key as a collision-free TinyAgents store name.
/// Bytes outside ASCII alphanumerics, `_`, and `.` become `-xx` hex escapes,
/// including `-` itself, so encoded bytes cannot collide with literal text.
/// Safe existing names stay unchanged. Dot-only keys (which the store rejects
/// as path traversal) escape every dot so they stay distinct from each other
/// and from the `"session"` fallback used for empty input.
pub fn sanitize_store_name(name: &str) -> String {
    use std::fmt::Write as _;
    if name.is_empty() {
        return "session".to_string();
    }
    let dot_only = name.bytes().all(|b| b == b'.');
    let mut encoded = String::with_capacity(name.len());
    for byte in name.bytes() {
        if !dot_only && (byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.')) {
            encoded.push(char::from(byte));
        } else {
            write!(&mut encoded, "-{byte:02x}").expect("String writes cannot fail");
        }
    }
    encoded
}

/// Journal stream name for a session.
///
/// Per-session streams (`session.{stem}.messages`) rather than per-thread:
/// multiple transcript files can share one `_meta.thread_id`, and appending
/// them into a shared stream would interleave sessions. The descriptor
/// carries `thread_id` so thread-level views can still be projected.
pub fn stream_name(session_key: &str) -> String {
    format!("session.{}.messages", sanitize_store_name(session_key))
}

/// Effective thread id for a transcript: `_meta.thread_id` when present,
/// otherwise a synthesized stable id. Returns `(thread_id, synthesized)`.
pub fn effective_thread_id(session_key: &str, meta_thread_id: Option<&str>) -> (String, bool) {
    match meta_thread_id {
        Some(t) if !t.is_empty() => (t.to_string(), false),
        _ => (
            format!("imported-{}", sanitize_store_name(session_key)),
            true,
        ),
    }
}

/// Build the `sessions/{session_key}` descriptor from a parsed transcript.
#[allow(clippy::too_many_arguments)]
pub fn build_descriptor(
    session_key: &str,
    transcript: &SessionTranscript,
    thread_id: String,
    thread_id_synthesized: bool,
    run_ids: Vec<String>,
    source: DescriptorSource,
    imported_at: String,
    warnings: usize,
) -> SessionDescriptor {
    let meta = &transcript.meta;
    SessionDescriptor {
        session_key: session_key.to_string(),
        parent_session_key: parent_session_key(session_key),
        thread_id,
        thread_id_synthesized,
        task_id: meta.task_id.clone(),
        run_ids,
        stream: stream_name(session_key),
        dispatcher: meta.dispatcher.clone(),
        agent_name: meta.agent_name.clone(),
        agent_id: meta.agent_id.clone(),
        agent_type: meta.agent_type.clone(),
        provider: meta.provider.clone(),
        model: meta.model.clone(),
        created: meta.created.clone(),
        updated: meta.updated.clone(),
        turn_count: meta.turn_count,
        usage: DescriptorUsage {
            input: meta.input_tokens,
            output: meta.output_tokens,
            cached_input: meta.cached_input_tokens,
            cost_usd: meta.charged_amount_usd,
        },
        source,
        import: DescriptorImport {
            version: IMPORT_VERSION,
            imported_at,
            warnings,
        },
    }
}

/// Host-supplied projection of one durable transcript row into its journal
/// record. The journal must carry whatever host metadata the host folds into a
/// row when it reads a transcript back (OpenHuman reconstructs
/// `openhuman_turn_usage` and friends), so the importer takes the projection as
/// a parameter instead of assuming a message type.
pub type JournalProjector = fn(TranscriptMessage) -> JournalMessage;

/// Neutral projector: copies `id`, `role`, the row's legacy string `content`
/// ([`TranscriptMessage::legacy_content`]: a tool round or image stays in the
/// journal's established string form) and `extra_metadata`, and adds nothing. For hosts with no message-shape sidecars, and for
/// tests.
pub fn plain_journal_message(message: TranscriptMessage) -> JournalMessage {
    JournalMessage {
        content: message.legacy_content(),
        id: message.id,
        role: message.role,
        extra_metadata: message.extra_metadata,
    }
}

/// Project a transcript's messages into journal records.
pub fn journal_messages(
    transcript: &SessionTranscript,
    project: JournalProjector,
) -> Vec<JournalMessage> {
    transcript.messages.iter().cloned().map(project).collect()
}
