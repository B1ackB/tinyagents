//! Out-of-band appends into a session's durable transcript.

use super::history::FileTranscriptLocator;
use super::history::TranscriptLocator;
use super::jsonl::build_message_line;
use super::paths::resolve_keyed_transcript_path;
use super::session::{SessionRef, session_stem};
use super::types::{
    BackgroundAppend, BackgroundAppendOutcome, BackgroundOrigin, TranscriptMessage,
};
use super::writer::append_bytes;
use anyhow::Context;

/// Appends `message` to the head generation of `session`.
pub async fn append_background_message(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    message: TranscriptMessage,
    options: BackgroundAppend,
) -> anyhow::Result<BackgroundAppendOutcome> {
    let head = locator.head_generation(&session.first_generation());
    if !locator.session_exists(&head) {
        return Ok(BackgroundAppendOutcome::NoSession);
    }
    let path = resolve_keyed_transcript_path(locator.workspace_dir(), &session_stem(&head))?;
    let mut line = build_message_line(&message, None, None, false);
    line.background = Some(BackgroundOrigin {
        idempotency_key: options.idempotency_key,
        provenance: options.provenance,
    });
    let mut buf = serde_json::to_string(&line).context("serialise background message line")?;
    buf.push('\n');
    append_bytes(&path, buf.as_bytes())?;
    Ok(BackgroundAppendOutcome::Appended {
        generation: head.generation,
    })
}

#[cfg(test)]
#[path = "background_tests.rs"]
mod tests;
