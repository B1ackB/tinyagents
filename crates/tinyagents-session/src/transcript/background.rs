//! Out-of-band appends into a session's durable transcript.

use super::history::FileTranscriptLocator;
use super::history::TranscriptLocator;
use super::jsonl::{LineKind, build_message_line, classify_line};
use super::paths::resolve_keyed_transcript_path;
use super::reader::read_jsonl_lines;
use super::session::{SessionRef, session_stem};
use super::types::{
    BackgroundAppend, BackgroundAppendOutcome, BackgroundOrigin, TranscriptMessage,
};
use super::writer::append_bytes;
use anyhow::Context;
use std::path::Path;

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
    if let Some(expected) = options.expected_generation
        && expected != head.generation
    {
        return Ok(BackgroundAppendOutcome::StaleGeneration {
            expected,
            head: head.generation,
        });
    }
    let path = resolve_keyed_transcript_path(locator.workspace_dir(), &session_stem(&head))?;
    if holds_idempotency_key(&path, &options.idempotency_key)? {
        return Ok(BackgroundAppendOutcome::Duplicate {
            generation: head.generation,
        });
    }
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

/// Whether any message line of the transcript at `path` was delivered with
/// `key`. Lines that do not parse are skipped, exactly as the readers skip them.
fn holds_idempotency_key(path: &Path, key: &str) -> anyhow::Result<bool> {
    let matches = |origin: &Option<BackgroundOrigin>| {
        origin
            .as_ref()
            .is_some_and(|origin| origin.idempotency_key == key)
    };
    for line in read_jsonl_lines(path)?.into_iter().flatten() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let found = match classify_line(line) {
            Ok(LineKind::Message(message)) => matches(&message.background),
            Ok(LineKind::Compaction(compaction)) => compaction
                .replacement
                .iter()
                .any(|message| matches(&message.background)),
            _ => false,
        };
        if found {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
#[path = "background_tests.rs"]
mod tests;
