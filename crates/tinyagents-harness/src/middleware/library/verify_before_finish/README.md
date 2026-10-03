# Verify before finish

`VerifyBeforeFinishMiddleware` can hold an eligible draft final answer for one
model check. It appends the check as a user turn, preserving the conversation's
prefix. The follow-up answer may call tools; the check fires at most once per
run. Hosts opt in and choose a tool-round threshold or a custom trigger.

`mod.rs` exposes the small public surface. `types.rs` defines activity, trigger,
and per-run state. `middleware.rs` contains configuration and lifecycle hooks;
`middleware_tests.rs` exercises those hooks and the run loop.

The check is skipped for tool-bearing, empty, truncated, or already continued
responses and when call or wall-clock budget is too small. Successful and
failed runs release activity through lifecycle hooks. Deferred resumes seed
activity from the resumed transcript before the next model response, so a new
context can still trigger the check. Interrupted runs are pruned once their
contexts have been dropped; active runs retain state beyond 1,024 entries.
