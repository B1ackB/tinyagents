# Task-state summarization

`TaskStateSummarizer` turns older conversation history into a checkpoint with two parts. `TaskLedger` extracts the original task, files and recent command outcomes from messages and tool calls. `TaskState` records goals, requirements, decisions and next steps returned by a model. The module root coordinates bounded sequential model updates; `ledger.rs` extracts facts, `render.rs` writes and restores checkpoints, and `types.rs` defines the public data types.

The public entry point is `TaskStateSummarizer::new(model, model_id)`. `with_max_chunk_tokens` limits each update request, and `with_response_format` asks providers for structured output. The summarizer implements `Summarizer`, including `merge` for independently summarized halves. `parse_carried` and `render_task_state` round-trip the checkpoint body.

Tool calls and results stay in the same chunk even when their combined size exceeds the configured chunk budget. Model updates run in order and receive the prior state. A failed update keeps the previous state and the deterministic ledger. Lists and individual fields are capped to keep future checkpoints small; file lists retain the most recent entries. The original task is escaped inside its tagged block so task text cannot forge checkpoint sections or file lists.

`SummaryRecord::usage` sums provider usage across update calls, including skipped responses when the provider reports usage.
