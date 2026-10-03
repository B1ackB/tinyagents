use super::*;

use serde_json::json;
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::tool::ToolCall;

fn call(id: &str, name: &str, arguments: Value) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(id, name, arguments)],
        usage: None,
        origin: None,
    })
}

#[test]
fn shell_reads_and_writes_come_from_common_idioms() {
    let (reads, writes) = shell_files(
        "cd /app && cat vm/vm.go | head -20; sed -n '1,40p' parser/parser.go.y && \
         sed -i 's/a/b/' ast/expr.go && echo hi > notes.txt && cp a/x.go b/y.go",
    );
    assert_eq!(reads, vec!["vm/vm.go", "parser/parser.go.y"]);
    assert_eq!(writes, vec!["ast/expr.go", "notes.txt", "b/y.go"]);
}

#[test]
fn heredoc_bodies_are_data_but_their_redirect_is_a_write() {
    let (reads, writes) = shell_files(
        "cat <<'EOF' > vm/zprobe_test.go\npackage vm\ncat secret.txt\nEOF\ngo test ./vm/...",
    );
    assert!(
        reads.is_empty(),
        "heredoc body must not count as a read: {reads:?}"
    );
    assert_eq!(writes, vec!["vm/zprobe_test.go"]);
}

#[test]
fn redirects_to_devices_and_fds_are_not_files() {
    let (_, writes) = shell_files("go build ./... 2>&1 >/dev/null; make > /dev/null 2> err.log");
    assert!(writes.is_empty(), "{writes:?}");
}

#[test]
fn ledger_absorbs_task_files_and_command_outcomes() {
    let messages = vec![
        Message::user("Add default arguments to anko functions."),
        call("c1", "shell", json!({"command": "cat vm/vm.go"})),
        Message::tool("c1", "package vm"),
        call(
            "c2",
            "apply_patch",
            json!({"patch": "*** Begin Patch\n*** Update File: vm/vmExprFunction.go\n@@\n-a\n+b\n*** End Patch"}),
        ),
        Message::tool("c2", "applied"),
        call("c3", "shell", json!({"command": "go test ./vm/..."})),
        Message::tool(
            "c3",
            "--- FAIL: TestDefaultArgs (0.00s)\n    default_arguments_test.go:16: expected substring: invalid default argument declaration\nFAIL\nCommand failed (exit code 1)",
        ),
        call("c4", "file_read", json!({"path": "./ast/expr.go"})),
        Message::tool("c4", "..."),
    ];
    let mut ledger = TaskLedger::default();
    ledger.absorb(&messages);

    assert_eq!(
        ledger.original_task.as_deref(),
        Some("Add default arguments to anko functions.")
    );
    assert_eq!(ledger.files_modified, vec!["vm/vmExprFunction.go"]);
    assert_eq!(ledger.files_read, vec!["vm/vm.go", "ast/expr.go"]);
    assert_eq!(ledger.commands.len(), 2);
    assert!(!ledger.commands[0].failed);
    let failed = &ledger.commands[1];
    assert!(failed.failed);
    assert_eq!(
        failed.error.as_deref(),
        Some(
            "default_arguments_test.go:16: expected substring: invalid default argument declaration"
        )
    );
}

#[test]
fn sigpipe_exit_is_not_a_failure() {
    assert!(failure_of("line\nCommand failed (exit code 141)").is_none());
    assert!(failure_of("ok\nexit code: 0").is_none());
    assert!(failure_of("{\"exit_code\": 2, \"output\": \"boom\"}").is_some());
    assert!(failure_of("Traceback (most recent call last):\n  ...\nValueError: bad").is_some());
}

#[test]
fn python_exception_line_wins_the_signature() {
    let out = "Traceback (most recent call last):\n  File \"x.py\", line 3, in <module>\nKeyError: 'sequence'\nCommand failed (exit code 1)";
    assert_eq!(
        error_signature(out).as_deref(),
        Some("KeyError: 'sequence'")
    );
}

#[test]
fn only_recent_commands_are_kept() {
    let mut messages = Vec::new();
    for i in 0..(MAX_COMMANDS + 5) {
        let id = format!("c{i}");
        messages.push(call(&id, "shell", json!({"command": format!("echo {i}")})));
        messages.push(Message::tool(&id, "ok"));
    }
    let mut ledger = TaskLedger::default();
    ledger.absorb(&messages);
    assert_eq!(ledger.commands.len(), MAX_COMMANDS);
    assert_eq!(
        ledger.commands.last().unwrap().command,
        format!("echo {}", MAX_COMMANDS + 4)
    );
}

#[test]
fn file_lists_keep_recent_entries() {
    let mut ledger = TaskLedger {
        files_read: (0..MAX_FILES_READ + 2)
            .map(|i| format!("read-{i}"))
            .collect(),
        files_modified: (0..MAX_FILES_MODIFIED + 2)
            .map(|i| format!("write-{i}"))
            .collect(),
        ..TaskLedger::default()
    };
    ledger.absorb(&[]);
    assert_eq!(ledger.files_read.len(), MAX_FILES_READ);
    assert_eq!(ledger.files_read[0], "read-2");
    assert_eq!(ledger.files_modified.len(), MAX_FILES_MODIFIED);
    assert_eq!(ledger.files_modified[0], "write-2");
}

#[test]
fn a_carried_task_is_not_replaced_by_a_later_user_message() {
    let mut ledger = TaskLedger {
        original_task: Some("the real task".into()),
        ..TaskLedger::default()
    };
    ledger.absorb(&[Message::user("continue")]);
    assert_eq!(ledger.original_task.as_deref(), Some("the real task"));
}

#[test]
fn tool_kinds_follow_common_harness_names() {
    assert_eq!(ToolKind::of("bash"), ToolKind::Shell);
    assert_eq!(ToolKind::of("str_replace_editor"), ToolKind::Edit);
    assert_eq!(ToolKind::of("write_file"), ToolKind::Edit);
    assert_eq!(ToolKind::of("read_file"), ToolKind::Read);
    assert_eq!(ToolKind::of("web_search_tool"), ToolKind::Other);
}

#[test]
fn revisited_file_remains_recent_when_list_is_capped() {
    let mut ledger = TaskLedger {
        files_read: (0..MAX_FILES_READ).map(|i| format!("f{i}")).collect(),
        ..TaskLedger::default()
    };
    push_unique(&mut ledger.files_read, "f0".into());
    push_unique(&mut ledger.files_read, "new".into());
    ledger.cap();
    assert!(!ledger.files_read.contains(&"f1".to_string()));
    assert_eq!(ledger.files_read[MAX_FILES_READ - 2..], ["f0", "new"]);
}

#[test]
fn exit_code_mentioned_in_output_is_not_a_failure() {
    let output = "src/main.rs:42: println!(\"exit code {}\", c)\nsrc/main.rs:57: done";
    assert!(failure_of(output).is_none());
    assert!(failure_of("exit code:\n57").is_none());
}

#[test]
fn file_lists_have_a_rendered_size_bound() {
    let mut ledger = TaskLedger {
        files_read: (0..MAX_FILES_READ)
            .map(|i| format!("{}-{i}", "x".repeat(100)))
            .collect(),
        ..TaskLedger::default()
    };
    ledger.cap();
    assert!(ledger.files_read.iter().map(String::len).sum::<usize>() <= MAX_FILE_LIST_CHARS);
    assert_eq!(
        ledger.files_read.last().unwrap(),
        &format!("{}-{}", "x".repeat(100), MAX_FILES_READ - 1)
    );
}

#[test]
fn file_list_cap_counts_escaped_path_expansion() {
    let mut ledger = TaskLedger {
        files_read: (0..MAX_FILES_READ)
            .map(|i| format!("{i}-{}", "&".repeat(40)))
            .collect(),
        ..TaskLedger::default()
    };
    ledger.cap();
    assert!(
        ledger
            .files_read
            .iter()
            .map(|path| escaped_path_len(path) + 1)
            .sum::<usize>()
            <= MAX_FILE_LIST_CHARS
    );
    assert!(ledger.files_read.len() < MAX_FILES_READ);
    assert!(
        ledger
            .files_read
            .last()
            .unwrap()
            .starts_with(&(MAX_FILES_READ - 1).to_string())
    );
}
