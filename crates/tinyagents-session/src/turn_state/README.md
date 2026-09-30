# Turn state

This module persists restart-survivable snapshots of active and recently
completed agent turns. It is separate from transcript storage: snapshots hold
live progress and UI-oriented tool/sub-agent state, while transcript JSONL
remains the durable conversation history.

## Public surface

`TurnStateStore` in `store.rs` loads, writes, queries, settles, and recovers
snapshots. `types.rs` defines the serializable `TurnState`, lifecycle, phase,
tool timeline, and sub-agent activity shapes. The crate re-exports the store
and common types from its turn-state module.

## Persistence and lifecycle

Each request has a snapshot beneath the workspace conversation turn-state
directory. Writes use a temporary file and atomic replacement, and a
process-wide lock serializes read-modify-write operations. Startup recovery
marks leftover non-terminal turns interrupted. Completed snapshots are pruned
to the newest retention window per thread; active snapshots are retained.
Settlement is idempotent for missing or already-terminal snapshots.

Snapshots are workspace-local operational state, not an audit log. Callers
should persist progress at meaningful turn boundaries and settle a turn when
its final lifecycle becomes known.
