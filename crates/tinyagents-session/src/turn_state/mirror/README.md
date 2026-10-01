# session::turn_state::mirror

`TurnStateMirror` keeps the authoritative `TurnState` snapshot of one in-flight
turn in sync with the agent loop and writes it through to a `TurnStateStore`,
so a UI that reattaches (or a process that restarts) can rebuild the live turn.

## Design

The host translates its own progress events into calls on the mirror: it
mutates `state` directly, calls the `push_*` helpers for transcript and
timeline bookkeeping, and calls `flush` at iteration and tool boundaries. The
mirror never flushes per streaming delta.

## Public surface

- `TurnStateMirror::new`, `snapshot`, `flush`, `finish`, and the
  `turn_completed` flag the host sets when it sees completion.
- `push` helpers: `push_transcript_narration`, `push_transcript_thinking`,
  `push_transcript_tool`, `push_subagent_prose`, `find_subagent_entry_mut`,
  and the ordering keys `next_seq` / `next_tool_seq`.
- `caps`: `MAX_PERSISTED_TRANSCRIPT_ITEM`, `TRANSCRIPT_TRUNCATION_MARKER`,
  `append_capped_transcript_text`, `cap_persisted_output`,
  `cap_persisted_args`.

## Operational constraints

- Persisted tool arguments, tool output and prose are size-capped, with an
  explicit truncation marker, so one large result cannot bloat every snapshot
  write.
- Narration and thinking coalesce within a round; a tool call or a new round
  starts a new block. Ordering keys are monotonic per turn.
- `finish` runs when the bridge exits. The bridge can outlive its turn, and the
  turn driver settles the snapshot itself (`TurnStateStore::settle_turn`), so
  `finish` writes `Interrupted` only through
  `TurnStateStore::put_unless_completed`: the completed check and the write
  are one store operation under the store lock. A completed turn is kept, an
  unreadable snapshot is left alone, and only a turn actually marked
  interrupted gets its partial answer appended to the session transcript
  (display-only; the model-context reader skips it).
- The partial is appended only when the thread's root transcript already
  exists. An interrupted first turn keeps its partial in the snapshot only.
