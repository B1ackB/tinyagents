//! [`push_ephemeral_instruction`]: add per-call guidance to a model request
//! without disturbing the cached prompt prefix.

use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::{ModelProfile, ModelRequest};

/// Header that introduces guidance appended to an existing tail message, so
/// the model can tell the harness's words from the tool's or the user's.
pub const HARNESS_NOTE_HEADER: &str = "[harness note]";
const EPHEMERAL_SECTION: &str = "__tinyagents_ephemeral_instruction";

/// Moves tail guidance created for a non-hoisting profile out of system
/// messages if a later middleware selected a hoisting model.
pub(crate) fn rehome_ephemeral_system_instructions(
    request: &mut ModelRequest,
    profile: Option<&ModelProfile>,
) {
    if !profile.is_some_and(|profile| profile.hoists_system_messages) {
        return;
    }
    let mut notes = Vec::new();
    request.messages.retain(|message| {
        if let Message::System(system) = message
            && system.sections.contains_key(EPHEMERAL_SECTION)
        {
            notes.push(message.text());
            false
        } else {
            true
        }
    });
    for note in notes {
        push_ephemeral_instruction(request, note, profile);
    }
}

/// Adds `text` to `request` as guidance for this call only.
///
/// Where the guidance goes depends on the target model's `profile`:
///
/// - **Default** (no profile, or one without
///   [`ModelProfile::hoists_system_messages`]): a tail `Message::System`. On a
///   wire that keeps system messages in place, a tail message leaves the
///   cached prefix alone.
/// - **Hoisting** (for example DeepSeek, whose chat template moves every
///   system turn to the prompt head): no system message, because new system
///   text there rewrites the cached prefix (#6962). The text is appended to
///   the tail message as a block opened by [`HARNESS_NOTE_HEADER`] when the
///   tail is a tool result (unless it is `trusted_verbatim`) or a user turn.
///   Otherwise it goes in a new trailing user turn, which keeps role
///   alternation valid for providers that reject consecutive user turns.
///
/// Only the request is edited; a caller that wants the text on the durable
/// transcript must write it there itself.
pub fn push_ephemeral_instruction(
    request: &mut ModelRequest,
    text: impl Into<String>,
    profile: Option<&ModelProfile>,
) {
    let text = text.into();
    if !profile.is_some_and(|profile| profile.hoists_system_messages) {
        tracing::trace!("[tinyagents::mw] ephemeral instruction placed as a tail system message");
        let mut message = Message::system(text);
        if let Message::System(system) = &mut message {
            system.sections.insert(EPHEMERAL_SECTION.to_string(), None);
        }
        request.messages.push(message);
        return;
    }
    let note = format!("{HARNESS_NOTE_HEADER}\n{text}");
    match request.messages.last_mut() {
        Some(Message::Tool(tool)) if !tool.trusted_verbatim => {
            tracing::debug!(
                "[tinyagents::mw] ephemeral instruction appended to the last tool result (model hoists system messages)"
            );
            tool.content.push(ContentBlock::Text(format!("\n\n{note}")));
        }
        Some(Message::User(user)) => {
            tracing::debug!(
                "[tinyagents::mw] ephemeral instruction appended to the tail user turn (model hoists system messages)"
            );
            user.content.push(ContentBlock::Text(format!("\n\n{note}")));
        }
        _ => {
            tracing::debug!(
                "[tinyagents::mw] ephemeral instruction added as a trailing user turn (model hoists system messages)"
            );
            request.messages.push(Message::user(note));
        }
    }
}

#[cfg(test)]
#[path = "ephemeral_tests.rs"]
mod tests;
