# Command hooks

The `command_hooks` module loads Cursor-compatible `hooks.json` files and
dispatches their commands or prompt evaluations at agent lifecycle events.
The host provides a [`HookEnvironment`](super::environment::HookEnvironment)
with its shell, prompt evaluator, home directory, and product name.

## Configuration and trust

Configuration is read from system, user, workspace, and project layers. Lists
are concatenated in that order, so a more specific layer cannot remove a
broader policy. Gating decisions merge with denial taking precedence over
approval. Each hook receives the arguments produced by preceding hooks, so
policy hooks evaluate rewrites before later hooks can use them.

`loop_limit` omitted uses the engine default of five; `null` means unlimited;
an integer sets a per-hook cap. Follow-up counts are scoped to a session and
hook definition. A sessionless invocation receives at most one follow-up.

## Execution and operational constraints

Gating events run sequentially and wait for completion. Observing events run in
the background. Command and prompt hooks use the configured timeout, falling
back to the engine default. Failures are fail-open by default; set
`fail_closed` to turn a failure, including a timeout, into a denial. Command
hooks receive the event envelope on stdin and return a decision on stdout.

Only `sessionStart` output may contribute session environment variables. These
variables stay scoped to the session and are cleared when the host calls
`HookEngine::forget_session`.

See the module root documentation for the module map and the public types for
event, payload, and decision schemas.
