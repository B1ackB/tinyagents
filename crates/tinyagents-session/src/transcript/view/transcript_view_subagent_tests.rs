use super::*;

#[test]
fn subagent_anchors_to_parent_turn_by_spawn_timestamp() {
    let dir = TempDir::new().unwrap();
    let root_stem = "800_orchestrator";
    let thread_id = "thr_anchor";
    let t1 = chrono::DateTime::from_timestamp(1_000_000, 0)
        .unwrap()
        .to_rfc3339();
    let t2 = chrono::DateTime::from_timestamp(2_000_000, 0)
        .unwrap()
        .to_rfc3339();
    let root_body = [
        r#"{"role":"user","content":"one","request_id":"req-1"}"#.to_string(),
        format!(
            r#"{{"role":"assistant","content":"a1","provider":"anthropic","model":"m","usage":{{"input":1,"output":1,"cached_input":0,"cost_usd":0.0}},"ts":"{t1}","iteration":1,"request_id":"req-1"}}"#
        ),
        r#"{"role":"user","content":"two","request_id":"req-2"}"#.to_string(),
        format!(
            r#"{{"role":"assistant","content":"a2","provider":"anthropic","model":"m","usage":{{"input":1,"output":1,"cached_input":0,"cost_usd":0.0}},"ts":"{t2}","iteration":1,"request_id":"req-2"}}"#
        ),
    ];
    let root_refs: Vec<&str> = root_body.iter().map(String::as_str).collect();
    write_raw(dir.path(), root_stem, thread_id, &root_refs);
    write_raw(
        dir.path(),
        &format!("{root_stem}__999950_coder"),
        thread_id,
        &[r#"{"role":"assistant","content":"coder work"}"#],
    );
    write_raw(
        dir.path(),
        &format!("{root_stem}__1000050_planner"),
        thread_id,
        &[r#"{"role":"assistant","content":"planner work"}"#],
    );

    let projected = project_thread(dir.path(), thread_id).expect("project thread");
    let mut anchors: Vec<(String, Option<String>)> = projected
        .items
        .iter()
        .filter_map(|item| match item {
            DisplayItem::Subagent {
                request_id, items, ..
            } => {
                let marker = items.iter().find_map(|inner| match inner {
                    DisplayItem::AssistantMessage { content, .. } => Some(content.clone()),
                    _ => None,
                })?;
                Some((marker, request_id.clone()))
            }
            _ => None,
        })
        .collect();
    anchors.sort();
    assert_eq!(
        anchors,
        vec![
            ("coder work".to_string(), Some("req-1".to_string())),
            ("planner work".to_string(), Some("req-2".to_string())),
        ]
    );
}

/// Exact correlation (#C1): when the run ledger records the spawning
/// `parentCallId` for this task, it wins over the timestamp/target-argument
/// heuristic — which would otherwise pick the first unclaimed
/// delegation-shaped call, regardless of which one actually spawned this
/// child.
#[test]
fn subagent_correlates_by_ledger_parent_call_id_over_the_heuristic() {
    let dir = TempDir::new().unwrap();
    let root_stem = "800_orch_exact";
    let thread_id = "thr_exact";
    let commit_ts = chrono::DateTime::from_timestamp(1_900_000, 0)
        .unwrap()
        .to_rfc3339();
    let root_body = [
        r#"{"role":"user","content":"do research","request_id":"req-1"}"#.to_string(),
        format!(
            r#"{{"role":"assistant","content":"","provider":"test","model":"test","usage":{{"input":1,"output":1,"cached_input":0,"cost_usd":0.0}},"tool_calls":[{{"id":"call-decoy","name":"spawn_async_subagent","arguments":"{{}}"}},{{"id":"call-real","name":"spawn_async_subagent","arguments":"{{}}"}}],"iteration":1,"request_id":"req-1","ts":"{commit_ts}"}}"#
        ),
    ];
    let root_refs: Vec<&str> = root_body.iter().map(String::as_str).collect();
    write_raw(dir.path(), root_stem, thread_id, &root_refs);

    // Spawned at unix 2_000_000 — after the only turn's commit, so it
    // anchors to that turn either way; the heuristic would still pick the
    // first unclaimed `spawn_*`-shaped call (`call-decoy`) since neither
    // call names an agent. Only the exact ledger lookup can tell them apart.
    let child_stem = format!("{root_stem}__2000000_000000001_researcher");
    let child = transcript::resolve_keyed_transcript_path(dir.path(), &child_stem).unwrap();
    // Written by hand (not `write_raw_at`'s default `meta_line`) so the
    // `_meta` header carries `task_id`/`agent_id` — the ledger correlation
    // key `build_child` reads.
    let child_meta_line = format!(
        r#"{{"_meta":{{"version":1,"agent":"researcher","agent_id":"researcher","agent_type":"subagent","dispatcher":"native","created":"2026-07-21T00:00:00Z","updated":"2026-07-21T00:00:10Z","turn_count":1,"input_tokens":1,"output_tokens":1,"cached_input_tokens":0,"charged_amount_usd":0.0,"thread_id":"{thread_id}","task_id":"sub-exact-1"}}}}"#
    );
    std::fs::write(
        &child,
        format!("{child_meta_line}\n{{\"role\":\"assistant\",\"content\":\"Bali is great.\"}}\n"),
    )
    .unwrap();

    crate::run_ledger::upsert_agent_run(
        dir.path(),
        crate::run_ledger::AgentRunUpsert {
            id: "sub-exact-1".to_string(),
            kind: crate::run_ledger::AgentRunKind::Subagent,
            parent_run_id: None,
            parent_thread_id: Some(thread_id.to_string()),
            agent_id: Some("researcher".to_string()),
            status: crate::run_ledger::AgentRunStatus::Completed,
            prompt_ref: None,
            worker_thread_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: None,
            metadata: serde_json::json!({ "parentCallId": "call-real" }),
            started_at: None,
            completed_at: None,
        },
    )
    .expect("seed run ledger row");

    let projected = project_thread(dir.path(), thread_id).expect("project thread");
    let subagent_call_id = projected.items.iter().find_map(|item| match item {
        DisplayItem::Subagent { call_id, .. } => Some(call_id.clone()),
        _ => None,
    });
    assert_eq!(
        subagent_call_id,
        Some(Some("call-real".to_string())),
        "exact ledger correlation must win over the first-unclaimed heuristic; items={:#?}",
        projected.items
    );
}

/// Parent transcript with two identically-shaped spawn calls, the second of
/// which carries `result_for_real` as its tool result. Returns the child's
/// resolved `call_id` after projecting a child whose `_meta` says `task_id` /
/// `thread_id`.
/// The native replay envelope a host persists a tool row as; the reader
/// unwraps it to the tool's own output. (A raw JSON output that itself carries
/// a `tool_call_id` key is indistinguishable from an envelope, so a realistic
/// transcript always has the envelope.)
fn envelope(call_id: &str, output: &str) -> String {
    serde_json::json!({"tool_call_id": call_id, "content": output}).to_string()
}

fn project_two_spawns(
    result_for_decoy: &str,
    result_for_real: &str,
    child_task_id: &str,
    child_thread_id: &str,
) -> Option<String> {
    let dir = TempDir::new().unwrap();
    let root_stem = "800_orch_link";
    let thread_id = "thr_link";
    let commit_ts = chrono::DateTime::from_timestamp(1_900_000, 0)
        .unwrap()
        .to_rfc3339();
    let call = |id: &str| format!(r#"{{"id":"{id}","name":"worker","arguments":"{{}}"}}"#);
    let root_body = [
        r#"{"role":"user","content":"go","request_id":"req-1"}"#.to_string(),
        format!(
            r#"{{"role":"assistant","content":"","provider":"test","model":"test","usage":{{"input":1,"output":1,"cached_input":0,"cost_usd":0.0}},"tool_calls":[{},{}],"iteration":1,"request_id":"req-1","ts":"{commit_ts}"}}"#,
            call("call-decoy"),
            call("call-real")
        ),
        format!(
            r#"{{"role":"tool","content":{},"id":"call-decoy","request_id":"req-1"}}"#,
            serde_json::to_string(&envelope("call-decoy", result_for_decoy)).unwrap()
        ),
        format!(
            r#"{{"role":"tool","content":{},"id":"call-real","request_id":"req-1"}}"#,
            serde_json::to_string(&envelope("call-real", result_for_real)).unwrap()
        ),
    ];
    let root_refs: Vec<&str> = root_body.iter().map(String::as_str).collect();
    write_raw(dir.path(), root_stem, thread_id, &root_refs);

    let child_stem = format!("{root_stem}__2000000_000000001_worker");
    let child = transcript::resolve_keyed_transcript_path(dir.path(), &child_stem).unwrap();
    let meta = format!(
        r#"{{"_meta":{{"version":1,"agent":"worker","agent_id":"worker","agent_type":"subagent","dispatcher":"native","created":"2026-07-21T00:00:00Z","updated":"2026-07-21T00:00:10Z","turn_count":1,"input_tokens":1,"output_tokens":1,"cached_input_tokens":0,"charged_amount_usd":0.0,"thread_id":"{child_thread_id}","task_id":"{child_task_id}"}}}}"#
    );
    std::fs::write(
        &child,
        format!("{meta}\n{{\"role\":\"assistant\",\"content\":\"done\"}}\n"),
    )
    .unwrap();

    let projected = project_thread(dir.path(), thread_id).expect("project thread");
    projected.items.iter().find_map(|item| match item {
        DisplayItem::Subagent { call_id, .. } => call_id.clone(),
        _ => None,
    })
}

/// Explicit link: the parent's tool result names the child's run id
/// (`subagent_run_id`), which the child's `_meta.task_id` repeats. That must
/// beat the first-unclaimed heuristic, which would pick `call-decoy`.
#[test]
fn subagent_correlates_by_explicit_run_id_in_the_parent_tool_result() {
    let call_id = project_two_spawns(
        r#"{"job_id":"subagent-job-1","status":"queued","subagent_run_id":"worker-d1-parent-0"}"#,
        r#"{"job_id":"subagent-job-2","status":"queued","subagent_run_id":"worker-d1-parent-1"}"#,
        "worker-d1-parent-1",
        "thr_link",
    );
    assert_eq!(call_id.as_deref(), Some("call-real"));
}

/// The run id is also recoverable from the child thread id the harness derives
/// (`{parent_thread}-subagent-{run_id}`) when no task id was written.
#[test]
fn subagent_correlates_by_run_id_embedded_in_the_child_thread_id() {
    let call_id = project_two_spawns(
        r#"{"job_id":"subagent-job-1","status":"queued","subagent_run_id":"worker-d1-parent-0"}"#,
        r#"{"job_id":"subagent-job-2","status":"queued","subagent_run_id":"worker-d1-parent-1"}"#,
        "",
        "thr_link-subagent-worker-d1-parent-1",
    );
    assert_eq!(call_id.as_deref(), Some("call-real"));
}

/// Old transcripts (results carry no ids) keep the heuristic: the first
/// unclaimed delegation-shaped call that targets the child's agent.
#[test]
fn subagent_without_explicit_ids_falls_back_to_the_heuristic() {
    let call_id = project_two_spawns(
        "Accepted async sub-agent",
        "Accepted async sub-agent",
        "worker-d1-parent-1",
        "thr_link",
    );
    assert_eq!(call_id.as_deref(), Some("call-decoy"));
}

/// A grandchild's thread id nests the markers
/// (`{thr}-subagent-{child}-subagent-{grandchild}`); the run id is the part
/// after the last one.
#[test]
fn subagent_correlates_a_grandchild_thread_id_by_its_last_marker() {
    let call_id = project_two_spawns(
        r#"{"job_id":"subagent-job-1","status":"queued","subagent_run_id":"worker-d2-x-0"}"#,
        r#"{"job_id":"subagent-job-2","status":"queued","subagent_run_id":"worker-d2-x-1"}"#,
        "",
        "thr_link-subagent-worker-d1-parent-0-subagent-worker-d2-x-1",
    );
    assert_eq!(call_id.as_deref(), Some("call-real"));
}

/// A `subagent_jobs` query result names the same run id, but it is not the
/// spawn: it has no `job_id` spawn payload shape and must not steal the match.
#[test]
fn a_job_query_result_does_not_steal_the_spawning_call() {
    let call_id = project_two_spawns(
        r#"{"id":"subagent-job-2","agent":"worker","status":"completed","subagent_run_id":"worker-d1-parent-1","tool_call_id":"call-real"}"#,
        r#"{"job_id":"subagent-job-2","status":"queued","subagent_run_id":"worker-d1-parent-1","tool_call_id":"call-real"}"#,
        "worker-d1-parent-1",
        "thr_link",
    );
    assert_eq!(call_id.as_deref(), Some("call-real"));
}

/// A spawn payload that records a different `tool_call_id` than the call
/// carrying it is not trusted.
#[test]
fn a_spawn_payload_naming_another_call_is_not_trusted() {
    let call_id = project_two_spawns(
        "Accepted async sub-agent",
        r#"{"job_id":"subagent-job-2","status":"queued","subagent_run_id":"worker-d1-parent-1","tool_call_id":"call-elsewhere"}"#,
        "worker-d1-parent-1",
        "thr_link",
    );
    assert_eq!(
        call_id.as_deref(),
        Some("call-decoy"),
        "falls back to the heuristic"
    );
}
