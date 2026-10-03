//! [`push_ephemeral_instruction`]: add per-call guidance to a model request
//! without disturbing the cached prompt prefix.

use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelProfile, ModelRequest};

/// Header that introduces guidance appended to an existing tail message.
pub const HARNESS_NOTE_HEADER: &str = "[harness note]";

/// Adds `text` to `request` as guidance for this call only.
pub fn push_ephemeral_instruction(
    request: &mut ModelRequest,
    text: impl Into<String>,
    _profile: Option<&ModelProfile>,
) {
    request.messages.push(Message::system(text.into()));
}

#[cfg(test)]
#[path = "ephemeral_tests.rs"]
mod tests;
