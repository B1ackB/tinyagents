//! Workspace-backed chat thread/message store.
//!
//! JSONL threads and messages under `<workspace>/memory/conversations/`, a
//! local trigram / CJK-bigram inverted index for cross-thread substring
//! search.
//!
//! This is the *product-facing* chat log a UI lists and renders (threads with
//! titles, labels, message counts). It is independent of the SQLite session
//! history ([`crate::ops`]) and of the model-facing [`crate::transcript`]
//! files: nothing here touches `session_db/` or `session_raw/`.
//!
//! The module was moved verbatim from TinyMemory's `tinymemory-conversations`
//! crate. The on-disk format, paths, serde wire names, and public names are
//! unchanged apart from two type names (`ConversationMessage` ->
//! [`ThreadMessage`], `ConversationMessagePatch` -> [`ThreadMessagePatch`]),
//! which TinyAgents' dependency-boundary guard reserves for OpenHuman. A host
//! switches by replacing `tinymemory_conversations::` with
//! `tinyagents_session::threads::` plus those two names, and existing
//! workspaces keep loading.
//! Errors stay `Result<_, String>` for that reason; see `README.md` beside
//! this file for the layout, locking, and index design.
//!
//! ## Layout
//!
//! - `types` - the on-disk wire types (threads, messages, patches, hits).
//! - `tokenize` - multilingual normalization + character n-gram tokenizer.
//! - `inverted_index` - in-memory index over message content.
//! - `store` - the JSONL [`ConversationStore`] and its free-function API.

mod inverted_index;
mod store;
mod tokenize;
mod types;

pub use store::{
    ConversationPurgeStats, ConversationStore, append_message, delete_messages_from, delete_thread,
    ensure_thread, get_messages, list_threads, purge_threads, update_message, update_thread_labels,
    update_thread_title, update_thread_working_dir,
};
pub use types::{
    ConversationThread, CreateConversationThread, CrossThreadHit, DETERMINISTIC_MESSAGE_ID_PREFIX,
    ThreadMessage, ThreadMessagePatch, is_deterministic_message_id, reply_run_id,
    run_reply_message_id,
};
