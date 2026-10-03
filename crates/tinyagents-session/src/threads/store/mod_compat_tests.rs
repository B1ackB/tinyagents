use super::*;

// On-disk compatibility with workspaces written by TinyMemory's
// `tinymemory-conversations` crate, which this module was moved from.
//
// The fixture below is the exact byte layout that crate wrote: the same root
// (`<workspace>/memory/conversations`), the same `threads.jsonl` log entries,
// and per-thread message files named by the lowercase hex of the thread id.
// The paths are spelled out as literals on purpose so a change to the
// filename derivation fails here rather than silently orphaning transcripts.

use tempfile::TempDir;

const LEGACY_THREADS_JSONL: &str = concat!(
    r#"{"op":"upsert","thread_id":"default:chat-1","title":"First chat","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z","labels":["work"]}"#,
    "\n",
    r#"{"op":"message_appended","thread_id":"default:chat-1","last_message_at":"2026-01-01T00:01:00Z"}"#,
    "\n",
    r#"{"op":"message_appended","thread_id":"default:chat-1","last_message_at":"2026-01-01T00:02:00Z"}"#,
    "\n",
    r#"{"op":"upsert","thread_id":"proactive:morning_briefing","title":"Morning briefing","created_at":"2026-01-02T00:00:00Z","updated_at":"2026-01-02T00:00:00Z"}"#,
    "\n",
    r#"{"op":"stats","thread_id":"proactive:morning_briefing","message_count":1,"last_message_at":"2026-01-02T00:00:05Z"}"#,
    "\n",
    r#"{"op":"upsert","thread_id":"default:child","title":"Branch","created_at":"2026-01-03T00:00:00Z","updated_at":"2026-01-03T00:00:00Z","parent_thread_id":"default:chat-1","labels":["agent-task"],"personality_id":"researcher"}"#,
    "\n",
    r#"{"op":"upsert","thread_id":"gone","title":"Gone","created_at":"2026-01-04T00:00:00Z","updated_at":"2026-01-04T00:00:00Z"}"#,
    "\n",
    r#"{"op":"delete","thread_id":"gone","deleted_at":"2026-01-04T00:00:01Z"}"#,
    "\n",
);

/// `default:chat-1`'s message log, as `ThreadMessage` serialized it.
const LEGACY_CHAT_MESSAGES: [&str; 2] = [
    r#"{"id":"user:1","content":"hello legacy world","type":"text","extraMetadata":{"scope":"web"},"sender":"user","createdAt":"2026-01-01T00:01:00Z"}"#,
    r#"{"id":"agent:run-1","content":"Hi there, from the old store","type":"text","extraMetadata":null,"sender":"assistant","createdAt":"2026-01-01T00:02:00Z"}"#,
];

const LEGACY_BRIEFING_MESSAGE: &str = r#"{"id":"system:b1","content":"Your morning briefing","type":"text","extraMetadata":{},"sender":"assistant","createdAt":"2026-01-02T00:00:05Z"}"#;

/// Lay the legacy fixture out under `workspace` exactly where the old crate
/// put it, and return the conversation root.
fn write_legacy_fixture(workspace: &Path) -> PathBuf {
    let root = workspace.join("memory").join("conversations");
    let threads_dir = root.join("threads");
    fs::create_dir_all(&threads_dir).expect("create fixture dirs");
    fs::write(root.join("threads.jsonl"), LEGACY_THREADS_JSONL).expect("write threads log");
    // hex("default:chat-1")
    fs::write(
        threads_dir.join("64656661756c743a636861742d31.jsonl"),
        format!("{}\n{}\n", LEGACY_CHAT_MESSAGES[0], LEGACY_CHAT_MESSAGES[1]),
    )
    .expect("write chat log");
    // hex("proactive:morning_briefing")
    fs::write(
        threads_dir.join("70726f6163746976653a6d6f726e696e675f6272696566696e67.jsonl"),
        format!("{LEGACY_BRIEFING_MESSAGE}\n"),
    )
    .expect("write briefing log");
    root
}

#[test]
fn loads_threads_written_by_tinymemory_conversations() {
    let temp = TempDir::new().expect("tempdir");
    write_legacy_fixture(temp.path());
    let store = ConversationStore::new(temp.path().to_path_buf());

    let threads = store.list_threads().expect("list legacy threads");
    let ids: Vec<&str> = threads.iter().map(|thread| thread.id.as_str()).collect();
    // Newest `last_message_at` first; the tombstoned thread is gone.
    assert_eq!(
        ids,
        [
            "default:child",
            "proactive:morning_briefing",
            "default:chat-1"
        ]
    );

    let child = &threads[0];
    assert_eq!(child.title, "Branch");
    assert_eq!(child.parent_thread_id.as_deref(), Some("default:chat-1"));
    assert_eq!(child.personality_id.as_deref(), Some("researcher"));
    assert_eq!(child.labels, ["tasks"]);
    assert_eq!(child.message_count, 0);
    assert_eq!(child.last_message_at, "2026-01-03T00:00:00Z");

    let briefing = &threads[1];
    assert_eq!(briefing.labels, ["briefing"]);
    assert_eq!(briefing.message_count, 1);
    assert_eq!(briefing.last_message_at, "2026-01-02T00:00:05Z");

    let chat = &threads[2];
    assert_eq!(chat.title, "First chat");
    assert_eq!(chat.labels, ["general"]);
    // No stats baseline in the log: backfilled from the message file.
    assert_eq!(chat.message_count, 2);
    assert_eq!(chat.last_message_at, "2026-01-01T00:02:00Z");
    assert!(chat.is_active);
    assert_eq!(chat.chat_id, None);
}

#[test]
fn legacy_messages_round_trip_byte_for_byte() {
    let temp = TempDir::new().expect("tempdir");
    write_legacy_fixture(temp.path());
    let store = ConversationStore::new(temp.path().to_path_buf());

    let messages = store
        .get_messages("default:chat-1")
        .expect("read legacy messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message_type, "text");
    assert_eq!(
        messages[0].extra_metadata,
        serde_json::json!({"scope": "web"})
    );
    assert_eq!(messages[1].extra_metadata, serde_json::Value::Null);
    for (message, legacy_line) in messages.iter().zip(LEGACY_CHAT_MESSAGES) {
        let reencoded = serde_json::to_string(message).expect("encode message");
        assert_eq!(reencoded, legacy_line);
    }
    assert_eq!(reply_run_id_of(&messages[1]), Some("run-1"));
}

fn reply_run_id_of(message: &ThreadMessage) -> Option<&str> {
    super::super::types::reply_run_id(&message.id)
}

#[test]
fn appends_to_legacy_thread_in_the_legacy_format() {
    let temp = TempDir::new().expect("tempdir");
    let root = write_legacy_fixture(temp.path());
    let store = ConversationStore::new(temp.path().to_path_buf());

    store
        .append_message(
            "default:chat-1",
            ThreadMessage {
                id: "user:2".to_string(),
                content: "a follow-up".to_string(),
                message_type: "text".to_string(),
                extra_metadata: serde_json::Value::Null,
                sender: "user".to_string(),
                created_at: "2026-01-05T00:00:00Z".to_string(),
            },
        )
        .expect("append to legacy thread");

    let chat_log = fs::read_to_string(root.join("threads/64656661756c743a636861742d31.jsonl"))
        .expect("read chat log");
    assert_eq!(
        chat_log.lines().collect::<Vec<_>>(),
        [
            LEGACY_CHAT_MESSAGES[0],
            LEGACY_CHAT_MESSAGES[1],
            r#"{"id":"user:2","content":"a follow-up","type":"text","extraMetadata":null,"sender":"user","createdAt":"2026-01-05T00:00:00Z"}"#,
        ]
    );

    let threads_log = fs::read_to_string(root.join("threads.jsonl")).expect("read threads log");
    assert!(threads_log.starts_with(LEGACY_THREADS_JSONL));
    let appended: serde_json::Value =
        serde_json::from_str(threads_log.lines().last().unwrap()).unwrap();
    assert_eq!(appended["op"], "message_appended");
    assert_eq!(appended["thread_id"], "default:chat-1");
    assert_eq!(appended["last_message_at"], "2026-01-05T00:00:00Z");
    assert_eq!(appended["message_bytes"], chat_log.len() as u64);

    let chat = store
        .list_threads()
        .expect("list threads")
        .into_iter()
        .find(|thread| thread.id == "default:chat-1")
        .expect("legacy thread still listed");
    assert_eq!(chat.message_count, 3);
    assert_eq!(chat.last_message_at, "2026-01-05T00:00:00Z");
}

#[test]
fn indexes_legacy_messages_for_cross_thread_search() {
    let temp = TempDir::new().expect("tempdir");
    write_legacy_fixture(temp.path());
    let store = ConversationStore::new(temp.path().to_path_buf());

    let hits = store
        .search_cross_thread_messages("legacy world", 5, None)
        .expect("search legacy messages");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].thread_id, "default:chat-1");
    assert_eq!(hits[0].message_id, "user:1");
    assert_eq!(hits[0].role, "user");

    let excluded = store
        .search_cross_thread_messages("legacy world", 5, Some("default:chat-1"))
        .expect("search with exclusion");
    assert!(excluded.is_empty());
}
