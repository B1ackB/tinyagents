//! Restart-survivable snapshots of in-flight agent turns.
//!
//! [`store`] is the persistence layer: one JSON file per turn under the
//! workspace, written atomically and serialized through a process-wide mutex;
//! [`store::mark_all_interrupted`] flags any non-terminal snapshot left over
//! from an unclean shutdown at cold boot. [`types`] holds the wire/storage
//! shapes (camelCase, mirroring a chat-runtime UI slice).
//!
//! [`mirror::TurnStateMirror`] is the writer: it owns the in-memory snapshot,
//! the flush policy, the size caps, interrupted-turn finalization and the
//! transcript bookkeeping helpers. Translating a host's own progress events into
//! mutations stays host work.

pub mod mirror;
pub mod store;
pub mod types;

pub use mirror::TurnStateMirror;
pub use store::TurnStateStore;
pub use types::{
    SubagentActivity, SubagentToolCall, ToolTimelineEntry, ToolTimelineStatus, TurnLifecycle,
    TurnPhase, TurnState,
};

#[cfg(test)]
mod shape_test;
