# `session::threads` — chat thread and message store

A workspace-backed store for the product-facing chat log: the threads a UI
lists (title, labels, parent, personality, message count, last activity) and
the messages it renders inside each one. It also keeps an in-memory inverted
index for substring search across every thread, and a small subscriber that
mirrors channel turns (Slack, Telegram, ...) into the same store.

It is plain files plus `serde`: no SQLite, no network, no memory engine.

## Where it came from

This module was moved verbatim from TinyMemory's `tinymemory-conversations`
crate. Chat history is session data, not memory, so it lives with the other
session persistence here. Nothing about the format changed:

- the root and metadata file names are the same; short thread IDs retain the
  same per-thread hex naming, while long IDs use bounded hash names;
- every serde attribute (`camelCase` fields, the `type` and `extraMetadata`
  wire keys, the `op`-tagged log entries) is the same;
- every public name is the same except two types, renamed because
  TinyAgents' dependency-boundary guard
  (`tinyagents-integration-tests/tests/dependency_boundary.rs`) reserves
  `ConversationMessage` and `ConversationMessagePatch` as OpenHuman domain
  type names that host-independent crates must not use. Type names never
  reach the disk, so the rename does not affect the format.

A host switches by changing the import path and the two type names:

| Old | New |
| --- | --- |
| `tinymemory_conversations::X` | `tinyagents_session::threads::X` |
| `ConversationMessage` | `ThreadMessage` |
| `ConversationMessagePatch` | `ThreadMessagePatch` |

Every other name (`ConversationStore`, `ConversationThread`,
`CreateConversationThread`, `ConversationPurgeStats`, `CrossThreadHit`, the
and the free functions) is unchanged. The `bus` persistence subscriber was
not carried over: hosts keep their own (OpenHuman's lives in
`openhuman-core`'s `threads::store::bus`). A host that wants to keep
its own spelling can import `ThreadMessage as ConversationMessage`.

The store, its wire types, and `CrossThreadHit` are also re-exported from the
crate root (`tinyagents_session::ConversationStore`, ...). The free functions
(`list_threads`, `append_message`, ...) stay under `threads::` so they are
not confused with the SQLite session operations at the root.

Errors stay `Result<_, String>` rather than the crate's `TinyAgentsError`.
That keeps the switch a pure path change for existing callers; converting
the error type is a separate API change.

`store/mod_compat_tests.rs` holds a fixture in the old crate's exact byte
layout and proves it loads, re-encodes byte-for-byte, accepts appends in the
same format, and is searchable.

## File layout

Every entry point takes the **workspace directory** and derives the root:

```text
{workspace_dir}/memory/conversations/
├── threads.jsonl                      ← append-only thread metadata log
└── threads/
    ├── {hex-or-sha256(thread_id)}.jsonl ← one message log per thread
    └── .conversations-{uuid}.tmp      ← transient, during atomic rewrites
```

- **`threads.jsonl`** is a log, not a table. Each line is one
  `{"op": ...}` entry, folded in order into the current thread state:
  - `upsert` — create or update a thread (`title`, `created_at`,
    `updated_at`, optional `parent_thread_id`, `labels`, `personality_id`).
    `labels: Some` replaces the label set; `None` keeps it, or infers one
    for a new thread (`briefing`, `notification`, or `general`).
  - `delete` — tombstone; the thread disappears from the fold.
  - `message_appended` — bumps the message count by one and sets
    `last_message_at`, so listing threads never reads message files.
  - `stats` — an absolute count/timestamp snapshot. Written to backfill
    threads whose messages predate `message_appended`, and to repair the
    trail after a crash between the two appends.
- **`threads/{hex-or-sha256}.jsonl`** holds one `ThreadMessage` per line
  (`id`, `content`, `type`, `extraMetadata`, `sender`, `createdAt`). The
  filename is the lowercase hex of the thread id's UTF-8 bytes for IDs up to
  124 bytes. Longer IDs use `sha256-` followed by the lowercase hex SHA-256
  digest of those bytes, keeping the filename within common filesystem limits.
- Appends are `O_APPEND` + `fsync`. Edits that must change existing lines
  (`update_message`, `delete_messages_from`) write a sibling temp file,
  `fsync` it, and rename it over the original, so a crash never leaves a
  half-written transcript.
- Readers skip blank and unparseable lines, so one corrupt line never loses
  the rest of a thread.

Message ids with the `agent:` prefix (`run_reply_message_id`) are minted
deterministically and may be presented twice by two writers; `append_message`
returns the stored message instead of writing a duplicate. Channel events use
`{role}:{message_id}` IDs with `extraMetadata.scope` set to `"channel"`;
`append_message` also returns the stored message on redelivery. Other IDs are
UUID-fresh and append without a lookup.

## Concurrency

Each conversation root has one lock set, shared by every `ConversationStore`
handle that points at it (roots are canonicalized first, so relative and
symlinked paths agree). The registry holds `Weak` references, so temporary
workspaces do not leak lock sets.

| Lock | Kind | Guards |
| --- | --- | --- |
| lifecycle | `RwLock` per root | read by every operation; write by `purge_threads` |
| thread | `Mutex` per thread id | that thread's message file |
| metadata | `Mutex` per root | reads and appends of `threads.jsonl` |
| index build | `Mutex` per root | only one cold index build at a time |

**Lock order:** lifecycle → thread → metadata → index cache. No code takes a
thread or metadata lock while holding the global index cache lock. So writers
on different threads append concurrently, and only the short
`threads.jsonl` append is serialized per root.

All of this is in-process. Two processes writing the same workspace are not
coordinated.

## The cross-thread index

`search_cross_thread_messages(query, limit, exclude_thread_id)` answers from
an in-memory inverted index over message content:

- **Tokenizing** (`tokenize.rs`): normalize (lowercase, strip combining
  marks and Latin diacritics, fold full-width ASCII), then index character
  bigrams for runs of CJK text and trigrams for everything else.
- **Querying** (`inverted_index.rs`): intersect the posting lists of each
  query term's n-grams to get candidates, then verify each candidate with an
  exact substring match on the normalized content. Score is
  `matched_terms / total_terms`, newest first on ties. Terms shorter than
  three bytes are dropped. A query whose candidate set is very large returns
  recent hits without the verify pass, to cap latency.
- **Lifetime:** the index is never written to disk. It is built from the
  JSONL files the first time a process searches a root, cached per root for
  the life of the process, and updated in place by appends, edits, deletes,
  and purges. The JSONL files are always the source of truth.
- **Cold build without stalls:** the build snapshots live thread ids under
  the metadata lock (no message I/O), releases it, reads each thread under
  its own thread lock, and journals any append that lands meanwhile. The
  journal is folded in when the built index is published, so no message is
  missed and unrelated writers are never blocked by the scan.

## Public surface

- **Store:** `ConversationStore` (`new`, `ensure_thread`, `list_threads`,
  `get_messages`, `append_message`, `update_thread_title`,
  `update_thread_labels`, `update_message`, `delete_messages_from`,
  `delete_thread`, `purge_threads`, `search_cross_thread_messages`) and
  `ConversationPurgeStats`.
- **Free functions:** the same operations taking a workspace path —
  `ensure_thread`, `list_threads`, `get_messages`, `append_message`,
  `update_thread_title`, `update_thread_labels`, `update_message`,
  `delete_messages_from`, `delete_thread`, `purge_threads`.
- **Types:** `ConversationThread`, `ThreadMessage`,
  `CreateConversationThread`, `ThreadMessagePatch`, `CrossThreadHit`.
- **Deterministic ids:** `DETERMINISTIC_MESSAGE_ID_PREFIX`,
  `run_reply_message_id`, `is_deterministic_message_id`, `reply_run_id`.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | module docs and re-exports |
| `types.rs` | wire types and deterministic message ids |
| `tokenize.rs` | normalization and n-gram tokenizer |
| `inverted_index.rs` | in-memory search index |
| `store/mod.rs` | `ConversationStore`, log entries, JSONL helpers, free functions |
| `store/ops.rs` | public CRUD and search methods |
| `store/index.rs` | log folding, stat repair, cold index build |
| `store/locks.rs` | per-root lock registry |
| `*_tests.rs` | unit tests; `store/mod_compat_tests.rs` is the legacy-format fixture |
