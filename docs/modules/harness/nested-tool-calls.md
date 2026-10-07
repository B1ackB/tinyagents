# Nested tool calls (C9)

A tool that needs another tool's result calls it through the harness instead of
reaching into the registry:

```rust
async fn execute_with_context(&self, args, _opts, context) -> anyhow::Result<ToolResult> {
    let harness = context
        .and_then(tinytools::ToolRunContext::host_extension)
        .and_then(|any| any.downcast_ref::<ToolExecutionContext>())
        .expect("installed by the agent loop");
    let found = harness.call_tool("search", json!({"q": "tinyagents"})).await?;
    // ...
}
```

`ToolExecutionContext::call_tool(name, arguments) -> Result<ToolResult>`
(`crates/tinyagents-harness/src/tool/types.rs`). The call goes through the same
admission and execution path as a model-issued call, so composing tools cannot
sidestep policy, validation or limits. Source: `src/tool/nested.rs` (the seam)
and `src/agent_loop/nested.rs` (the runner).

## What a nested call goes through

| Stage | Behaviour |
| --- | --- |
| Lookup | `ToolRegistry::model_dispatch` plus the hosted run's tool allow-list. An unknown or disallowed name is `Err(ToolNotFound)`. |
| Arguments | Injected-argument preparation, the `InvalidArgsPolicy` normalisation, schema validation. Invalid arguments are an `Err` naming the tool. |
| Approval | A tool that is external or declares `approval_required` **fails**: `nested call '<name>' requires approval; nested calls cannot be deferred`. The parent is never deferred. A `CallDeferred`/`ApprovalRequired` raised during execution is the same refusal. |
| Host authorization | `SecurityGate::authorize_tool` for a hosted run; a denial is an `Err`. |
| Tool-wrap onion | `ToolMiddleware::wrap_tool` runs around the tool, so a policy middleware can deny or rewrite a nested call. |
| Timeouts | The per-tool timeout policy and the run's remaining wall-clock budget. |
| Budget | One `max_tool_calls` pool shared with model-issued calls (below). |
| Cancellation | The run's token: a cancelled run refuses the next nested call, and dropping the parent drops its in-flight nested calls. |

A tool that reports its own failure (`ToolResult::is_error`) is an `Ok` result,
exactly as for a model-issued call. A refusal is an `Err(TinyAgentsError)`; a
tool that forwards it with `?` is answered to the model as
`TinyAgentsError::ToolFailed` (the message above reaches the model verbatim).

Outside the agent loop (no runner installed) `call_tool` returns
`ToolFailed("cannot call tool '<name>' ...: nested tool calls are only available
while the agent loop executes the calling tool")`.

## Budget

`LimitTracker` holds an atomic `nested_tool_calls` next to `tool_calls`. A
nested call reserves a slot with `try_reserve_nested_tool_call` (`&self`, so it
works from a tool future that holds only `&RunContext`); model-issued admission
counts `tool_calls + nested_tool_calls` against `max_tool_calls`. Concurrent
parents therefore cannot overspend the cap, and a cap spent by nested calls
trips the next model-issued call. A nested call that is refused at admission
releases its slot; one that ran keeps it. A spent cap is always an error for
the nested caller (`LimitExceeded`, plus `LimitReached { ToolCalls }`), whatever
`LimitBehavior` says: there is no loop boundary at which to stop with a partial
result.

## Depth

`RunLimits::max_nested_depth` (default `3`, `with_max_nested_depth`). A call the
model issued is level 0; what its tool calls is level 1; and so on. A call whose
level exceeds the cap fails with `nested call '<name>' exceeds
max_nested_depth (<n>)`. This is distinct from `max_depth`, the sub-agent
recursion cap.

## Ids and events

A nested call's id is `<parent call id>/<n>` (`n` counts from 1 per parent;
nested-of-nested ids extend the path: `p1/1/1`). `ToolStarted`, `ToolCompleted`
and `ToolFailed` carry `parent_call_id: Option<CallId>` (serde default, omitted
when `None`), so exporters can nest the span. Every nested `ToolStarted` has
exactly one terminal partner.

## Transcript and metadata

Nested calls get **no transcript rows**: a nested call is never answered to the
provider, so adding one would break tool-call/tool-result pairing. The parent's
result metadata (host-only; `ToolCompleted.metadata` and
`AgentRun::tool_metadata`) gains a capped summary:

```json
{
  "nested_calls": [
    {"id": "p1/1", "name": "leaf", "status": "ok", "duration_ms": 3,
     "args": "{\"n\":7}", "error": null}
  ],
  "nested_calls_truncated": 8
}
```

`status` is `ok`, `error` (the tool returned `is_error`) or `failed` (refused or
raised). `args` is the serialized arguments cut at 1 KiB; `error` is cut at 256
bytes and omitted when empty. At most 32 entries are kept; `nested_calls_truncated`
counts the rest. The summary is attached only when the metadata is absent or an
object.

## What a nested call does not do

- **Lifecycle hooks.** `Middleware::before_tool` / `after_tool` take
  `&mut RunContext`, which a tool future (holding `&RunContext`) cannot lend,
  so they do not run for nested calls. The wrap onion does. Policy that must
  bind nested calls belongs in a `ToolMiddleware::wrap_tool`. For the same
  reason the repeat-progress and no-progress guards (which key on
  `before_tool`/`after_tool`) never see nested calls: they cannot be counted as
  model repeats of the parent.
- **Progress gate.** A nested tool's `report_progress` is not forwarded: the
  parent's gate is keyed to the parent's call id, and a nested call does not open
  one of its own on the parent's stream. Use the nested call's result and the
  parent's own progress.
- **Effect ledger, host output screening, tool control.** The parent's ledger
  row and the screening of the parent's final output cover the call as a whole.
  A nested result's `ToolControl` (`terminate`, `return_direct`, `goto`) and a
  wrap middleware's control request are ignored.

## Shape (why it is a channel)

The harness installs a `NestedToolRunner` (a type-erased `Arc<dyn ...>`) in a
task-local while it drives the call, and `ToolExecutionContext::from_run_context`
captures it for exactly that call id. The runner is a channel: the loop that
drives the tool's future also services the nested requests it sends, borrowing
`&AgentHarness`, `&State` and `&RunContext` as the model-issued path does.
Servicing a nested call is a plain call-base execution, so it can nest again.
