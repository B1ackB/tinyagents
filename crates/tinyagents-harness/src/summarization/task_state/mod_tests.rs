use super::*;

use serde_json::json;
use tinyinference_llm::message::AssistantMessage;
use tinyinference_llm::tool::ToolCall;

use crate::testkit::ScriptedModel;

fn shell(id: &str, command: &str, output: &str) -> Vec<Message> {
    vec![
        Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, "shell", json!({ "command": command }))],
            usage: None,
            origin: None,
        }),
        Message::tool(id, output),
    ]
}

fn history() -> Vec<Message> {
    let mut m = vec![Message::user("Implement default arguments in anko.")];
    m.extend(shell("c1", "cat vm/vm.go", "package vm"));
    m.extend(shell(
        "c2",
        "go test ./vm/...",
        "FAIL\nvm_test.go:9: boom\nCommand failed (exit code 1)",
    ));
    m
}

const STATE_REPLY: &str = r#"{"goal": "default args", "requirements": ["invalid default argument declaration"], "todos_open": ["fix error text"], "test_command": "go test ./vm/...", "next_step": "edit parser"}"#;

#[tokio::test]
async fn checkpoint_holds_model_state_and_ledger() {
    let model = Arc::new(ScriptedModel::replies(vec![STATE_REPLY]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let record = summarizer.summarize(&history()).await.unwrap();
    let body = record.summary.text();

    assert!(body.starts_with(TASK_STATE_HEADER));
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("- invalid default argument declaration"));
    assert!(body.contains("<read-files>\nvm/vm.go\n</read-files>"));
    assert!(body.contains("→ FAILED: vm_test.go:9: boom"));
    assert_eq!(model.requests().len(), 1);
    // The transcript is fenced and the instruction comes last.
    let prompt = model.requests()[0].messages[1].text();
    assert!(prompt.contains("<transcript>"));
    assert!(prompt.trim_end().ends_with("error lines exactly."));
}

#[tokio::test]
async fn second_compaction_updates_the_previous_state_and_carries_files() {
    let model = Arc::new(ScriptedModel::replies(vec![
        STATE_REPLY,
        r#"{"goal": "default args", "todos_done": ["fix error text"]}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let first = summarizer.summarize(&history()).await.unwrap();

    let later = shell("c3", "sed -i 's/x/y/' parser/parser.go.y", "");
    let request = SummaryRequest::new(later).with_previous_summary(first.summary.text());
    let second = summarizer.summarize_request(&request).await.unwrap();
    let body = second.summary.text();

    // The previous model state reached the model as JSON to update.
    let prompt = model.requests()[1].messages[1].text();
    assert!(prompt.contains("<previous_state>"));
    assert!(prompt.contains("\"todos_open\":[\"fix error text\"]"));
    // Carried exactly: original task and files from the first window.
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("<modified-files>\nparser/parser.go.y\n</modified-files>"));
    assert!(body.contains("<read-files>\nvm/vm.go\n</read-files>"));
    assert!(body.contains("## Done\n- fix error text"));
}

#[tokio::test]
async fn markup_then_bad_json_degrades_to_the_ledger_instead_of_failing() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<｜DSML｜invoke name=\"shell\">",
        "I think the state is fine.",
    ]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let record = summarizer.summarize(&history()).await.unwrap();
    assert_eq!(model.requests().len(), 2, "one retry, then give up");
    assert!(record.provenance.reason.contains("ledger only"));
    let body = record.summary.text();
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("→ FAILED: vm_test.go:9: boom"));
}

#[tokio::test]
async fn a_model_outage_with_nothing_to_fall_back_on_is_an_error() {
    // No replies scripted: every call fails. Plain chat with no tool calls and
    // no user task gives the ledger nothing to carry.
    let model = Arc::new(ScriptedModel::replies(Vec::<&str>::new()));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let err = summarizer
        .summarize(&[Message::assistant("thinking")])
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn long_histories_are_folded_in_sequential_chunks() {
    let mut messages = vec![Message::user("task")];
    for i in 0..6 {
        messages.extend(shell(
            &format!("c{i}"),
            &format!("echo {i}"),
            &"x".repeat(400),
        ));
    }
    let replies: Vec<String> = (0..10)
        .map(|i| format!("{{\"goal\": \"step {i}\"}}"))
        .collect();
    let model = Arc::new(ScriptedModel::replies(replies));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m").with_max_chunk_tokens(250);
    let record = summarizer.summarize(&messages).await.unwrap();

    let requests = model.requests();
    assert!(
        requests.len() > 1,
        "expected several chunks, got {}",
        requests.len()
    );
    // Every chunk after the first updates the state the previous one wrote,
    // and no chunk opens on an orphaned tool result.
    for (i, request) in requests.iter().enumerate().skip(1) {
        let prompt = request.messages[1].text();
        assert!(
            prompt.contains(&format!("\"goal\":\"step {}\"", i - 1)),
            "chunk {i}"
        );
        assert!(
            !prompt.contains("<transcript>\ntool:"),
            "chunk {i} starts on a tool result"
        );
    }
    assert!(
        record
            .summary
            .text()
            .contains(&format!("step {}", requests.len() - 1))
    );
}

#[tokio::test]
async fn merge_unions_files_and_keeps_the_later_state() {
    let model = Arc::new(ScriptedModel::replies(vec![
        STATE_REPLY,
        r#"{"goal": "later", "requirements": ["second half requirement"]}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let a = summarizer.summarize(&history()).await.unwrap();
    let b = summarizer
        .summarize(&shell("c9", "touch new/file.rs", ""))
        .await
        .unwrap();
    let merged = summarizer.merge(&[a, b]).await.unwrap();
    let body = merged.summary.text();
    assert!(body.contains("## Goal\nlater"));
    assert!(body.contains("second half requirement"));
    assert!(body.contains("invalid default argument declaration"));
    assert!(body.contains("<modified-files>\nnew/file.rs\n</modified-files>"));
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("invalid default argument declaration"));
    let (ledger, _) = parse_carried(&body);
    assert!(ledger.commands.iter().any(|command| command.failed));
}

#[tokio::test]
async fn merge_refreshes_revisited_file_before_capping() {
    let summarizer = TaskStateSummarizer::new(Arc::new(ScriptedModel::new(vec![])), "m");
    let record = |files: Vec<String>| SummaryRecord {
        summary: Message::user(render_task_state(
            &TaskState::default(),
            &TaskLedger {
                files_modified: files,
                ..Default::default()
            },
        )),
        provenance: CompressionProvenance {
            source_ids: vec![],
            original_token_estimate: 0,
            summary_token_estimate: 0,
            reason: String::new(),
        },
        usage: None,
    };
    let first = record(
        (0..ledger::MAX_FILES_MODIFIED)
            .map(|i| format!("f{i}"))
            .collect(),
    );
    let second = record(vec!["f0".into(), "new".into()]);

    let merged = summarizer.merge(&[first, second]).await.unwrap();
    let (_, state) = parse_carried(&merged.summary.text());
    let state = state.unwrap();
    assert_eq!(state.current_hypothesis.as_deref(), Some(""));
    assert_eq!(state.test_command.as_deref(), Some(""));
    assert_eq!(state.next_step.as_deref(), Some(""));

    let (ledger, _) = parse_carried(&merged.summary.text());
    assert!(ledger.files_modified.contains(&"f0".to_string()));
    assert!(ledger.files_modified.contains(&"new".to_string()));
    assert!(!ledger.files_modified.contains(&"f1".to_string()));
}

#[tokio::test]
async fn later_split_summary_clears_obsolete_live_task_fields() {
    let model = Arc::new(ScriptedModel::replies(vec![
        r#"{"current_hypothesis":"old theory","test_command":"cargo test old","next_step":"retry old work"}"#,
        r#"{"current_hypothesis":"","test_command":"","next_step":""}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let first = summarizer.summarize(&history()).await.unwrap();
    let second = summarizer
        .summarize(&shell("c9", "echo done", ""))
        .await
        .unwrap();
    let merged = summarizer.merge(&[first, second]).await.unwrap();
    let (_, state) = parse_carried(&merged.summary.text());
    let state = state.unwrap();
    assert_eq!(state.current_hypothesis.as_deref(), Some(""));
    assert_eq!(state.test_command.as_deref(), Some(""));
    assert_eq!(state.next_step.as_deref(), Some(""));
}

#[tokio::test]
async fn later_split_summary_omitting_live_task_fields_preserves_earlier_values() {
    let model = Arc::new(ScriptedModel::replies(vec![
        r#"{"current_hypothesis":"old theory","test_command":"cargo test old","next_step":"retry old work"}"#,
        r#"{"goal":"updated goal"}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let first = summarizer.summarize(&history()).await.unwrap();
    let second = summarizer
        .summarize(&shell("c9", "echo done", ""))
        .await
        .unwrap();
    let merged = summarizer.merge(&[first, second]).await.unwrap();
    let (_, state) = parse_carried(&merged.summary.text());
    let state = state.unwrap();
    assert_eq!(state.current_hypothesis.as_deref(), Some("old theory"));
    assert_eq!(state.test_command.as_deref(), Some("cargo test old"));
    assert_eq!(state.next_step.as_deref(), Some("retry old work"));
}

#[test]
fn oversized_tool_pair_stays_in_one_chunk() {
    let summarizer = TaskStateSummarizer::new(Arc::new(ScriptedModel::new(vec![])), "m")
        .with_max_chunk_tokens(1);
    let messages = shell("large", "cat huge.log", &"x".repeat(1000));
    let chunks = summarizer.chunks(&messages);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), 2);
}

#[test]
fn bounded_caps_lists_and_items() {
    let many = |n: usize| (0..n).map(|i| format!("item {i}")).collect::<Vec<_>>();
    let state = TaskState {
        requirements: many(60),
        decisions: many(40),
        todos_done: many(40),
        todos_open: many(40),
        errors_and_fixes: vec!["e".repeat(5_000)],
        current_hypothesis: Some("h".repeat(5_000)),
        ..TaskState::default()
    }
    .bounded();
    assert_eq!(state.requirements.len(), 40);
    assert_eq!(
        state.requirements[0], "item 0",
        "requirements keep the first"
    );
    assert_eq!(state.decisions.len(), 12);
    assert_eq!(
        state.decisions[11], "item 39",
        "history keeps the most recent"
    );
    assert_eq!(state.todos_done[0], "item 28");
    assert_eq!(state.todos_open[0], "item 0", "open work keeps the oldest");
    assert!(state.errors_and_fixes[0].chars().count() <= 401);
    assert!(state.current_hypothesis.unwrap().chars().count() <= 801);
}

#[tokio::test]
async fn checkpoint_size_stays_bounded_across_many_compactions() {
    // A model that only ever adds to its lists, as the live DeepSWE run did.
    let replies: Vec<String> = (0..30)
        .map(|i| {
            let grow = |p: &str| (0..(i + 1) * 3).map(|j| format!("\"{p} {j}: {}\"", "x".repeat(200))).collect::<Vec<_>>().join(",");
            format!(
                "{{\"goal\": \"g\", \"decisions\": [{}], \"todos_done\": [{}], \"errors_and_fixes\": [{}]}}",
                grow("decision"),
                grow("done"),
                grow("error")
            )
        })
        .collect();
    let model = Arc::new(ScriptedModel::replies(replies));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let mut previous: Option<String> = None;
    let mut sizes = Vec::new();
    for i in 0..30 {
        let mut request =
            SummaryRequest::new(shell(&format!("c{i}"), &format!("cat src/f{i}.rs"), "ok"));
        if let Some(p) = &previous {
            request = request.with_previous_summary(p.clone());
        }
        let body = summarizer
            .summarize_request(&request)
            .await
            .unwrap()
            .summary
            .text();
        sizes.push(body.len());
        previous = Some(body);
    }
    let last = *sizes.last().unwrap();
    // 3 lists x 12 items x ~215 chars, plus the ledger: well under 12k chars.
    assert!(last < 12_000, "checkpoint grew to {last} chars: {sizes:?}");
}

#[tokio::test]
async fn merge_keeps_the_first_halfs_state_and_commands() {
    let model = Arc::new(ScriptedModel::replies(vec![
        STATE_REPLY,
        r#"{"goal": "", "todos_open": ["second half work"]}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let a = summarizer.summarize(&history()).await.unwrap();
    let b = summarizer
        .summarize(&shell("c9", "cargo build", "ok"))
        .await
        .unwrap();
    let body = summarizer.merge(&[a, b]).await.unwrap().summary.text();
    // The empty later goal does not erase the earlier one; lists union.
    assert!(body.contains("## Goal\ndefault args"));
    assert!(body.contains("- invalid default argument declaration"));
    assert!(body.contains("- fix error text"));
    assert!(body.contains("- second half work"));
    // Commands from both halves survive.
    assert!(body.contains("- `go test ./vm/...` → FAILED: vm_test.go:9: boom"));
    assert!(body.contains("- `cargo build` → ok"));
}

#[tokio::test]
async fn a_second_compaction_carries_earlier_commands_forward() {
    let model = Arc::new(ScriptedModel::replies(vec![STATE_REPLY, STATE_REPLY]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let first = summarizer.summarize(&history()).await.unwrap();
    let second = summarizer
        .summarize_request(
            &SummaryRequest::new(vec![Message::assistant("thinking")])
                .with_previous_summary(first.summary.text()),
        )
        .await
        .unwrap();
    assert!(
        second
            .summary
            .text()
            .contains("- `go test ./vm/...` → FAILED: vm_test.go:9: boom")
    );
}

#[test]
fn an_oversized_tool_group_is_never_split() {
    let mut messages = vec![Message::user("task")];
    messages.extend(shell("c1", "cat big", &"x".repeat(4_000)));
    messages.push(Message::user("after"));
    let summarizer = TaskStateSummarizer::new(Arc::new(ScriptedModel::replies(vec!["{}"])), "m")
        .with_max_chunk_tokens(50);
    for chunk in summarizer.chunks(&messages) {
        assert!(
            !matches!(chunk.first(), Some(Message::Tool(_))),
            "a chunk opens on an orphaned tool result"
        );
    }
}

#[test]
fn merge_closes_open_work_the_other_half_finished() {
    let earlier = TaskState {
        todos_open: vec!["a".into(), "b".into()],
        ..TaskState::default()
    };
    let later = TaskState {
        todos_done: vec!["a".into()],
        ..TaskState::default()
    };
    let merged = earlier.merged_with(later);
    assert_eq!(merged.todos_open, vec!["b"]);
    assert_eq!(merged.todos_done, vec!["a"]);
}

#[test]
fn later_split_state_can_reopen_completed_work() {
    let first = TaskState {
        todos_done: vec!["run tests".into()],
        todos_open: vec!["update docs".into()],
        ..TaskState::default()
    };
    let later = TaskState {
        todos_open: vec!["run tests".into()],
        todos_done: vec!["update docs".into()],
        ..TaskState::default()
    };
    let merged = first.merged_with(later);
    assert_eq!(merged.todos_open, vec!["run tests"]);
    assert_eq!(merged.todos_done, vec!["update docs"]);
}
