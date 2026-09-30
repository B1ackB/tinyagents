//! Restart-survivable snapshots of in-flight agent turns.
//!
//! [`store`] is the persistence layer: one JSON file per turn under the
//! workspace, written atomically and serialized through a process-wide mutex;
//! [`store::mark_all_interrupted`] flags any non-terminal snapshot left over
//! from an unclean shutdown at cold boot. [`types`] holds the wire/storage
//! shapes (camelCase, mirroring a chat-runtime UI slice).
//!
//! Filling a snapshot from a host's progress events is host work: the host
//! builds a [`types::TurnState`], mutates it, and calls [`store::put`] at
//! iteration / tool boundaries.

pub mod store;
pub mod types;

pub use store::TurnStateStore;
pub use types::{
    SubagentActivity, SubagentToolCall, ToolTimelineEntry, ToolTimelineStatus, TurnLifecycle,
    TurnPhase, TurnState,
};

#[cfg(test)]
mod shape_test;
