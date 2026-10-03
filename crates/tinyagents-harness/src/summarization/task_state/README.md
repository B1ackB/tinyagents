# summarization::task_state

Typed task-state compaction. A `TaskStateSummarizer` replaces the older part
of a transcript with one checkpoint that has two halves.

## Design

- **Ledger** (`ledger.rs`, `TaskLedger`): facts copied from the transcript
  without a model. The first user message (the task, verbatim up to a cap),
  files modified and read (most-recent order), and the last 15 shell commands
  with how each ended and the most informative error line.
- **State** (`types.rs`, `TaskState`): fields only a reader of the
  conversation can state (goal, requirements, constraints, decisions, errors
  and fixes, done/open work, hypothesis, test command, next step). One
  structured model call per chunk *updates* the previous state.
- **Render and parse** (`render.rs`): both halves are written as one body
  (tagged `<original-task>`, `<modified-files>`, `<read-files>` blocks plus
  `## ` sections). `parse_carried` reads them back so the next compaction
  carries them forward exactly, including `## Recent commands`. Sections are
  only read after the closing `</original-task>`, so headings in the user's
  task are never mistaken for state.
- **Summarizer** (`mod.rs`): `summarize_request` carries the previous
  checkpoint, absorbs the new messages into the ledger, then folds the history
  in chunks of at most `with_max_chunk_tokens` estimated tokens. Chunks cut
  only where tool calls stay paired with their results; one group larger than
  the limit is sent whole. `merge` joins split halves field by field (lists
  union in order, a later scalar wins unless empty, commands concatenate and
  are capped).

## Public surface

`TaskStateSummarizer` (`new`, `with_max_chunk_tokens`, `with_response_format`),
`TaskState` (`bounded`), `TaskLedger`, `CommandRecord`, `render_task_state`,
`parse_carried`, `parse_state_reply`, `TASK_STATE_HEADER`.

## Operational constraints

- **Failure degrades, it does not fail.** A model error, tool-call markup, or
  no parseable JSON after two attempts keeps the previous state and the full
  ledger. Only a failure with nothing to fall back on is an error.
- **Bounded size.** `TaskState::bounded` and `TaskLedger::cap` cap every list and scalar so a
  checkpoint stays a few thousand tokens however many times it is carried.
- **No sensitive logging.** Logs carry counts and the model id, never message
  text.
- Unit tests live in the sibling `*_tests.rs` files.
