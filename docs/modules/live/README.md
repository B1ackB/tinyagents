# `tinyagents-live`

Live voice agent sessions on top of the harness.

## Why it exists

[`tinyliveagents`](https://github.com/tinyhumansai/tinyliveagents) gives every
live voice provider (Gemini Live direct or relayed, ElevenLabs Agents, the
Sarvam STT → chat → TTS cascade) one session type, and deliberately never runs
a tool: a `LiveEvent::ToolCall` goes to the host. Running that call *safely* —
admission, allow-lists, approvals, tool policy, timeouts, credential
scrubbing — is exactly what the harness already does for typed turns, so a
live session should reuse that pipeline rather than call `Tool::execute`
directly.

## Surface

- `LiveAgent::new(Arc<AgentHarness<State, Ctx>>, Arc<State>)`.
- `LiveAgent::start(&dyn LiveProvider, LiveConfig, RunContext<Ctx>, LiveAgentOptions)`
  declares the harness's direct tools (and deferred ones with
  `include_deferred_tools`) after any the host declared, connects, and returns
  a `LiveAgentSession`.
- `LiveAgentSession::sender()` is the provider's `LiveSender` (audio, text,
  interrupt, close); `recv()` yields `LiveAgentEvent::Live(LiveEvent)` for every
  provider event plus `ToolStarted` / `ToolFinished` around each call.
- `tool_declarations(&harness, include_deferred)` for hosts that only need the
  declarations (to mint a relay ticket, for example).

## How a call runs

1. The driver task forwards every provider event to the host and queues
   `ToolCall`s for the worker; `ToolCallCancelled` marks ids cancelled.
2. One worker task owns the session's `RunContext` — limits, budgets and
   events accumulate across the conversation like one agent run — and runs
   calls serially through `agent_loop::phases::execute_tool_batch`.
3. The result text is the folded tool message; the error flag comes from the
   harness's `ToolCompleted { error }` / `ToolFailed` events (the tool message
   itself carries none). A call with no tool message was deferred (it needs an
   approval or external answer) and is answered with an error result.
4. A call cancelled before it starts is skipped. One cancelled while running
   finishes (tools have side effects) but its result is not sent.
5. `LiveAgentOptions::tool_timeout` bounds each call; the harness's own
   per-tool timeout policy applies inside it.

## What stays with the host

Choosing the provider and minting relay tickets or signed URLs, building the
harness and `RunContext` (origin, approval scope, workspace), audio capture
and playback, and persisting transcripts.

## Testing

`src/runner_tests.rs` drives a scripted `LiveProvider` built from
`LiveSession::from_channels`: allowed calls, failing tools, `before_tool`
denials, approval deferrals, unknown tools, cancellation before and during a
call, timeouts and teardown. `examples/sarvam_agent.rs` runs the whole path
against Sarvam (`SARVAM_API_KEY`).
