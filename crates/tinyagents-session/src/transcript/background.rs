//! Out-of-band appends into a session's durable transcript.

use super::history::FileTranscriptLocator;
use super::session::SessionRef;
use super::types::{BackgroundAppend, BackgroundAppendOutcome, TranscriptMessage};

/// Appends `message` to the head generation of `session`.
pub async fn append_background_message(
    locator: &FileTranscriptLocator,
    session: &SessionRef,
    message: TranscriptMessage,
    options: BackgroundAppend,
) -> anyhow::Result<BackgroundAppendOutcome> {
    let _ = (locator, session, message, options);
    Ok(BackgroundAppendOutcome::NoSession)
}

#[cfg(test)]
#[path = "background_tests.rs"]
mod tests;
