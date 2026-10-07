//! Out-of-band appends into a session's durable transcript.
//!
//! A conversation's transcript is normally written only by its own live turns.
//! Some work finishes *outside* any turn but belongs in the conversation — a
//! scheduled job created from a chat reports back into that chat, so the next
//! turn can see it ("did you remind me?"). [`append_background_message`] is
//! that writer: one plain assistant message, appended to the session's head
//! generation, idempotent per caller key, and marked with where it came from.
//!
//! # Concurrency contract
//!
//! The live turn path appends a *delta* against the baseline it read when the
//! turn started, and validates that baseline under the file lock. A foreign
//! line written between that read and the persist makes the baseline stale and
//! the user's turn fails to persist. So:
//!
//! - **A live turn must hold [`lock_session_turn`] from its resume read
//!   through its persist.** `tinyagents-runtime`'s `Session::turn` does this
//!   for every session-bound target whose locator names its destination.
//!   A host driving the transcript some other way must take the lock itself.
//! - **The background append awaits that same lock**, so it lands *between*
//!   turns, never inside one. It can therefore wait for as long as a turn
//!   runs; bound it with a timeout if the caller cannot wait.
//! - **Never await it from inside a turn of the same session** (for example a
//!   tool that delivers into its own conversation): the lock is not reentrant
//!   and that deadlocks.
//! - **The next turn must re-read the transcript** to see the message, which a
//!   turn resumed with `ResumeMode::Session` (or any mode other than `Never`)
//!   does on every turn. A session whose turns never resume keeps a stale
//!   baseline and its next persist fails rather than hiding the line.
//! - The turn lock is **in-process**. Another process appending mid-turn is
//!   still caught by the baseline check (that turn fails to persist; nothing is
//!   corrupted), and the append itself takes the same cross-process file locks
//!   as every other writer, so bytes never interleave.

use super::history::seed_meta_for_discovered;
use super::history::{FileTranscriptHistory, FileTranscriptLocator, TranscriptLocator};
use super::jsonl::{LineKind, build_message_line, classify_line};
use super::paths::resolve_keyed_transcript_path;
use super::reader::read_jsonl_lines;
use super::session::{SessionRef, session_stem};
use super::turn_lock::lock_session_turn_with_notification;
use super::types::{
    BackgroundAppend, BackgroundAppendOutcome, BackgroundOrigin, TranscriptMessage,
};
use super::writer::append_bytes;
use anyhow::Context;
use std::path::Path;
use tokio::sync::Notify;

/// How many times the head may move under an unpinned append before it gives
/// up. Each move is a compaction or a fork landing between head resolution
/// and the write lock; more than a couple in a row means something is wrong.
const MAX_HEAD_MOVES: usize = 3;

/// Appends `message` — one plain assistant message — to the head generation of
/// `session` under `locator`, outside any turn.
///
/// - **Head only.** The head is resolved from the session's first generation,
///   whatever `session.generation` says. With
///   [`BackgroundAppend::expected_generation`] set, a head that has moved on
///   returns [`BackgroundAppendOutcome::StaleGeneration`] and writes nothing.
/// - **Idempotent.** A head generation that already holds a line with the same
///   [`BackgroundAppend::idempotency_key`] returns
///   [`BackgroundAppendOutcome::Duplicate`] and writes nothing. The key is
///   checked within the head generation only: once a compaction has carried
///   the message into a successor, that successor holds it as an ordinary row.
/// - **Never creates a session.** No transcript yet is
///   [`BackgroundAppendOutcome::NoSession`].
///
/// The line is an ordinary message line, so the model-context reader
/// ([`read_transcript`](super::read_transcript)) replays it as an assistant row
/// and the display reader shows it. It additionally carries a top-level
/// `background` field (a [`BackgroundOrigin`]: the key and the caller's
/// `provenance`), surfaced as [`DisplayMessage::background`](super::DisplayMessage::background);
/// readers that predate the field ignore it. No `_meta` line is appended, so
/// turn counts and usage totals are untouched, and the `.md` companion is next
/// re-rendered by the following turn.
///
/// Awaits the session's turn lock first; see the module documentation for the
/// contract this places on live turns. The file I/O itself is synchronous and
/// small (one scan of the head file, one append).
///
/// # Errors
///
/// When `message` is not a plain assistant message (another role, tool calls,
/// or an interrupted partial), when the key is empty, on I/O failure, or when
/// another process holds the head's successor reserved mid-compaction.
pub async fn append_background_message(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    message: TranscriptMessage,
    options: BackgroundAppend,
) -> anyhow::Result<BackgroundAppendOutcome> {
    append_background_message_impl(locator, session, message, options, None).await
}

/// Test-oriented variant that signals immediately before waiting for the
/// shared turn lock. The append and production path are otherwise identical.
#[doc(hidden)]
pub async fn append_background_message_with_lock_notification(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    message: TranscriptMessage,
    options: BackgroundAppend,
    lock_attempted: &Notify,
) -> anyhow::Result<BackgroundAppendOutcome> {
    append_background_message_impl(locator, session, message, options, Some(lock_attempted)).await
}

async fn append_background_message_impl(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    message: TranscriptMessage,
    options: BackgroundAppend,
    lock_attempted: Option<&Notify>,
) -> anyhow::Result<BackgroundAppendOutcome> {
    let message = validate(message, &options)?;
    let root = session.first_generation();
    let stem = session_stem(&root);
    let key = options.idempotency_key.as_str();
    tracing::debug!(session = %stem, idempotency_key = %key, expected_generation = ?options.expected_generation, "[transcript-background] append requested");

    let _turn = lock_session_turn_with_notification(locator, &root, lock_attempted).await;
    for _ in 0..MAX_HEAD_MOVES {
        let head = locator.head_generation(&root);
        if !locator.session_exists(&head) {
            tracing::debug!(session = %stem, idempotency_key = %key, "[transcript-background] no transcript; nothing written");
            return Ok(BackgroundAppendOutcome::NoSession);
        }
        if let Some(stale) = stale(&options, head.generation) {
            tracing::debug!(session = %stem, idempotency_key = %key, ?stale, "[transcript-background] head moved on; nothing written");
            return Ok(stale);
        }
        match append_to_head(locator, &head, &message, &options)? {
            Attempt::Done(outcome) => {
                tracing::debug!(session = %stem, idempotency_key = %key, ?outcome, "[transcript-background] append finished");
                return Ok(outcome);
            }
            Attempt::HeadMoved => {
                tracing::debug!(session = %stem, idempotency_key = %key, generation = head.generation, "[transcript-background] head sealed during append; re-resolving");
            }
        }
    }
    anyhow::bail!(
        "session {stem} head kept moving during a background append ({MAX_HEAD_MOVES} attempts)"
    )
}

/// The result of one locked attempt against a resolved head.
enum Attempt {
    Done(BackgroundAppendOutcome),
    /// A successor generation appeared between resolution and the lock.
    HeadMoved,
}

/// Scans and appends to `head` under its file write locks.
fn append_to_head(
    locator: &FileTranscriptLocator,
    head: &SessionRef,
    message: &TranscriptMessage,
    options: &BackgroundAppend,
) -> anyhow::Result<Attempt> {
    let workspace = locator.workspace_dir();
    let path = resolve_keyed_transcript_path(workspace, &session_stem(head))?;
    let successor =
        resolve_keyed_transcript_path(workspace, &session_stem(&head.next_generation()))?;
    let history = FileTranscriptHistory::opened_at(path.clone(), seed_meta_for_discovered(""));
    history.with_write_locks(|| {
        if successor.is_file() {
            return Ok(Attempt::HeadMoved);
        }
        let generation = head.generation;
        if holds_idempotency_key(&path, &options.idempotency_key)? {
            return Ok(Attempt::Done(BackgroundAppendOutcome::Duplicate {
                generation,
            }));
        }
        let mut line = build_message_line(message, message.turn_usage.as_ref(), None, false);
        line.ts = line.ts.or_else(|| Some(chrono::Utc::now().to_rfc3339()));
        line.background = Some(BackgroundOrigin {
            idempotency_key: options.idempotency_key.clone(),
            provenance: options.provenance.clone(),
        });
        let mut buf = serde_json::to_string(&line).context("serialise background message line")?;
        buf.push('\n');
        append_bytes(&path, buf.as_bytes())?;
        Ok(Attempt::Done(BackgroundAppendOutcome::Appended {
            generation,
        }))
    })
}

/// `StaleGeneration` when the caller pinned a generation other than `head`.
fn stale(options: &BackgroundAppend, head: u32) -> Option<BackgroundAppendOutcome> {
    options
        .expected_generation
        .filter(|expected| *expected != head)
        .map(|expected| BackgroundAppendOutcome::StaleGeneration { expected, head })
}

/// Rejects anything but a plain, complete assistant message with a usable key.
///
/// A tool-calling row would leave calls with no results in the model context,
/// and an interrupted row would be skipped by the model-context reader — the
/// opposite of what a delivery is for.
fn validate(
    message: TranscriptMessage,
    options: &BackgroundAppend,
) -> anyhow::Result<TranscriptMessage> {
    let normalized = message.normalized();
    anyhow::ensure!(
        !options.idempotency_key.trim().is_empty(),
        "background append needs a non-empty idempotency key"
    );
    anyhow::ensure!(
        normalized.role == "assistant",
        "background append only writes assistant messages, got role `{}`",
        normalized.role
    );
    anyhow::ensure!(
        !normalized.is_typed()
            && normalized
                .turn_usage
                .as_ref()
                .is_none_or(|usage| usage.tool_calls.is_empty())
            && !normalized.interrupted
            && normalized.tool_failure.is_none(),
        "background append only writes plain, complete assistant messages"
    );
    Ok(normalized)
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
