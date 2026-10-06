# The session store port

`tinyagents_session::port` is where a host plugs in its own storage for
everything an agent persists while it runs. A host implements one
`SessionStoreProvider`; the runtime asks it for each agent's `AgentStores`
and never names a file, a database or a directory itself.

```rust
pub trait SessionStoreProvider: Send + Sync {
    fn for_agent(&self, agent_id: &str) -> AgentStores;
    fn recover(&self) -> anyhow::Result<()> { Ok(()) }          // once, at boot
    fn destination_key(&self) -> Option<String> { None }        // for logs
    fn workspace_dir(&self) -> Option<PathBuf> { None }         // file-backed only
}

pub struct AgentStores {
    pub transcripts: Arc<dyn TranscriptLocator>,  // what the model sees
    pub turn_states: Arc<dyn TurnStates>,         // snapshots of turns in flight
    pub kv: Arc<dyn Store>,                       // run status, goals, todos
    pub journal: Arc<dyn AppendStore>,            // each run's events
}
```

## Why a port

The file and SQLite layout (`session_raw/`, `session_db/sessions.db`,
`tinyagents_store/`, turn-state files) suits one operator on one machine.
A cloud host serving many users from one process needs every agent's state
in a shared database, scoped by agent, so a conversation outlives the
process that served it. The port lets both use the same runtime.

## Isolation

`for_agent` is the only way to obtain stores, and the handles it returns are
bound to that agent. A provider serving several users from one backend must
make it impossible for one agent's handles to reach another's data; the
single-user desktop layout shares one workspace between its agents and does
not claim isolation.

## Sync and async

`TranscriptLocator`/`TranscriptHistory` and `TurnStates` are synchronous:
the turn path that commits a transcript is a chain of sync methods. An
implementation over an async client bridges at its own boundary (for tokio,
`block_in_place` on a multi-threaded runtime). `Store` and `AppendStore` are
the harness's async traits; `Arc<dyn Store>` and `Arc<dyn AppendStore>`
implement them, so code generic over a store accepts injected handles, and
`FileStatusStore::over` keeps run status in any `Store`.

## Bundled pieces

- `InMemorySessionStores`: per-agent, process-lifetime stores
  (`InMemoryTranscriptLocator`, `InMemoryTurnStates`, the harness's
  in-memory stores). For tests and hosts that keep nothing.
- `TurnStateStore` implements `TurnStates`, `FileTranscriptLocator`
  implements `TranscriptLocator`; together with `open_session_stores` they
  are the building blocks of a file-backed provider. This crate does not
  assemble one: the host that owns a layout does (OpenHuman's
  `openhuman-store-sqlite`).
- `TranscriptLocator::append_interrupted_partial` carries the display-only
  partial of an interrupted turn without a file path.

## Conformance

`testkit::conformance::session_store_conformance(&provider)` exercises a
whole provider: transcripts (sessions, thread and agent lookups, compaction
generations, partials kept out of the replay), turn states (conditional
writes, settling, interrupted-marking) and the key-value and journal stores.
`session_store_isolation_conformance` checks that two agents cannot see each
other's data. Both run here against `InMemorySessionStores` and the file
building blocks, and in each host against its own provider.
