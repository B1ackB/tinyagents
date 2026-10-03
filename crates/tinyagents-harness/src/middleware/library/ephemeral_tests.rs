//! Tests for [`push_ephemeral_instruction`].

use super::*;
use tinyinference_llm::message::{ContentBlock, Message, ToolMessage};
use tinyinference_llm::model::{ModelProfile, ModelRequest};

fn hoisting() -> ModelProfile {
    ModelProfile {
        hoists_system_messages: true,
        mid_conversation_system_messages: true,
        ..ModelProfile::default()
    }
}

fn request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        messages,
        ..Default::default()
    }
}

fn system_count(request: &ModelRequest) -> usize {
    request
        .messages
        .iter()
        .filter(|message| matches!(message, Message::System(_)))
        .count()
}

#[test]
fn non_hoisting_profile_appends_a_tail_system_message() {
    for profile in [None, Some(ModelProfile::default())] {
        let mut req = request(vec![Message::system("persona"), Message::user("hi")]);
        push_ephemeral_instruction(&mut req, "check your work", profile.as_ref());
        assert_eq!(req.messages.len(), 3);
        assert!(matches!(req.messages.last(), Some(Message::System(_))));
        assert_eq!(req.messages.last().unwrap().text(), "check your work");
    }
}

#[test]
fn a_later_hoisting_selection_rehomes_earlier_ephemeral_system_text() {
    let leading = Message::system("stable persona");
    let mut req = request(vec![leading.clone(), Message::user("question")]);
    push_ephemeral_instruction(&mut req, "artifact: outputs/result.json", None);

    rehome_ephemeral_system_instructions(&mut req, Some(&hoisting()));

    assert_eq!(system_count(&req), 1);
    assert_eq!(req.messages[0], leading);
    assert!(
        req.messages
            .last()
            .unwrap()
            .text()
            .contains("artifact: outputs/result.json")
    );
}

#[test]
fn a_mid_conversation_non_hoisting_selection_keeps_ephemeral_system_text() {
    let profile = ModelProfile {
        mid_conversation_system_messages: true,
        ..ModelProfile::default()
    };
    let mut req = request(vec![Message::system("persona"), Message::user("question")]);
    push_ephemeral_instruction(&mut req, "check your work", None);

    rehome_ephemeral_system_instructions(&mut req, Some(&profile));

    assert_eq!(system_count(&req), 2);
    assert_eq!(req.messages.last().unwrap().text(), "check your work");
}

#[test]
fn a_hoisting_selection_rehomes_even_without_mid_conversation_system_support() {
    let profile = ModelProfile {
        hoists_system_messages: true,
        ..ModelProfile::default()
    };
    let mut req = request(vec![Message::system("persona"), Message::user("question")]);
    push_ephemeral_instruction(&mut req, "check your work", None);

    rehome_ephemeral_system_instructions(&mut req, Some(&profile));

    assert_eq!(system_count(&req), 1);
    assert!(
        req.messages
            .last()
            .unwrap()
            .text()
            .contains("check your work")
    );
    assert!(matches!(req.messages.last(), Some(Message::User(_))));
}

#[test]
fn hoisting_profile_appends_to_the_last_tool_result_without_a_system_message() {
    let leading = Message::system("persona");
    let mut req = request(vec![
        leading.clone(),
        Message::user("hi"),
        Message::tool("call-1", "result body"),
    ]);

    push_ephemeral_instruction(&mut req, "check your work", Some(&hoisting()));

    assert_eq!(req.messages.len(), 3, "no message added");
    assert_eq!(system_count(&req), 1, "no new system message");
    assert_eq!(req.messages[0], leading, "leading system message untouched");
    let Some(Message::Tool(tool)) = req.messages.last() else {
        panic!("tail must still be the tool result");
    };
    let text = req.messages.last().unwrap().text();
    assert!(text.starts_with("result body"), "{text}");
    assert!(text.contains(HARNESS_NOTE_HEADER), "{text}");
    assert!(text.ends_with("check your work"), "{text}");
    assert_eq!(tool.tool_call_id, "call-1");
}

#[test]
fn hoisting_profile_appends_to_a_tail_user_message() {
    let mut req = request(vec![Message::system("persona"), Message::user("hi")]);

    push_ephemeral_instruction(&mut req, "check your work", Some(&hoisting()));

    assert_eq!(req.messages.len(), 2, "no consecutive user turns");
    assert_eq!(system_count(&req), 1);
    let text = req.messages.last().unwrap().text();
    assert!(text.starts_with("hi"), "{text}");
    assert!(text.ends_with("check your work"), "{text}");
}

#[test]
fn hoisting_profile_adds_a_user_reminder_after_an_assistant_tail() {
    let mut req = request(vec![
        Message::system("persona"),
        Message::user("hi"),
        Message::assistant("hello"),
    ]);

    push_ephemeral_instruction(&mut req, "check your work", Some(&hoisting()));

    assert_eq!(req.messages.len(), 4);
    assert_eq!(system_count(&req), 1);
    assert!(matches!(req.messages.last(), Some(Message::User(_))));
    assert!(
        req.messages
            .last()
            .unwrap()
            .text()
            .contains("check your work")
    );
}

#[test]
fn hoisting_profile_never_edits_a_verbatim_tool_result() {
    let verbatim = Message::Tool(ToolMessage {
        tool_call_id: "call-1".into(),
        content: vec![ContentBlock::Text("exact bytes".into())],
        trusted_verbatim: true,
        artifact: None,
    });
    let mut req = request(vec![Message::user("hi"), verbatim.clone()]);

    push_ephemeral_instruction(&mut req, "check your work", Some(&hoisting()));

    assert_eq!(req.messages[1], verbatim, "verbatim result untouched");
    assert_eq!(req.messages.len(), 3);
    assert!(matches!(req.messages.last(), Some(Message::User(_))));
    assert_eq!(system_count(&req), 0);
}
