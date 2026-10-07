# Subagent policy, result policy and role

See also `crates/tinyagents-orchestration/src/subagent/README.md` (the section below is mirrored there).

One `SubAgentPolicy { timeout, retry, budget, retry_after_tool_calls }` governs
every path (it is defined in `tinyagents-graph` and re-exported here, because
orchestration depends on graph).

- **Timeout.** `SubagentDriver` runs the executor under its own child token; on
  elapse the child is cancelled (the caller's token is not) and the outcome is
  `Incomplete(kind: Timeout)`. `SubAgentTool` cancels the job token and marks
  the job `Incomplete` (`incomplete_kind: "timeout"`). A timeout is never retried.
- **Retry.** Only a failure the `RetryPolicy` deems retryable, and never after
  the attempt ran tools unless `retry_after_tool_calls`. The driver retries
  `SubagentError::Transient { tools_ran, .. }` (adapters classify; anything else
  is never retried) and needs `PreparedSubagent::retry_context`, a factory for a
  fresh run context per attempt; without it one attempt runs. `SubAgentTool`
  pre-mints one child per attempt and watches `ToolStarted` events to learn
  whether tools ran. A host-visible side effect of retry: the job's recorded
  `subagent_run_id` is the first attempt's.
- **Budget.** Call caps tighten the child `RunConfig` (stricter wins) so the
  harness enforces them during the run. Input/output token caps are checked on
  the reported usage after the run: the driver ends `Incomplete(BudgetExceeded)`
  (output kept), the tool path fails the job `Incomplete`. `max_cost` is **not
  enforced** here; `SubAgentBudget::to_budget_limits()` gives the harness
  `BudgetMiddleware` the same caps for in-run cost enforcement.
- **Incomplete status.** `SubagentStatus::Incomplete(SubagentIncomplete { reason,
  kind: IncompleteKind })` and `SubAgentJobStatus::Incomplete` replace the old
  `[SUBAGENT_INCOMPLETE]` text marker. The transcript view reads
  `"status": "incomplete"` in a spawn result and still reads the legacy marker;
  both project as `SubagentStatus::Incomplete` (previously `Failed`).
- **Result policy.** `ResultPolicy { max_chars, overflow: Truncate | Artifact,
  schema }` (builders; `artifact_store` is a host `ArtifactStore` callback, the
  orchestration layer owns no store). `Truncate` keeps head and tail around
  `[… N chars omitted …]`; `Artifact` stores the full output through the host
  store and returns the preview plus a path-free `ArtifactReference` (no store:
  falls back to truncate and reports `artifact_error`). `schema` is checked with
  the tool-call boundary's structural validator (`type`, `properties`,
  `required`, `additionalProperties`, `items`, `enum`; no `$ref`/combinators)
  and a mismatch is surfaced as `schema_error`, never a failure. Defaults: no
  cap, no schema.
- **Role and framing.** `SubagentRole::{Orchestrator (default), Leaf}`.
  `restrict_tools` drops `subagent_jobs`, `subagent_message` and the host's
  `delegation_tools` for a leaf and intersects with `tool_ceiling` so a child
  never widens what it inherited; the driver applies it to
  `PreparedSubagent::tools`. `SubAgentTool` cannot filter a shared harness, so a
  leaf tool refuses to spawn when the child harness exposes a delegation tool.
  `subagent_framing(role, depth, task)` is an optional neutral preamble hosts
  may prepend; nothing applies it by default.

**API note.** `PreparedSubagent` gained fields (use `PreparedSubagent::new`),
`SubagentOutcome` gained `schema_error`, `SubagentIncomplete` gained `kind`,
`SubagentError` gained `Transient`, `SubAgentBudget` gained token/cost fields
(use `..SubAgentBudget::unlimited()`), and the view `SubagentStatus` gained
`Incomplete`.
