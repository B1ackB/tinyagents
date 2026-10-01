//! Tests for [`MemoryProtocolMiddleware`].

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::{AgentRun, Middleware, ToolInvocationIdentity};
use tinyinference_llm::tool::ToolCall as TaToolCall;
use tinytools::ToolResult as TaToolResult;

use super::super::memory_protocol::{MEMORY_PROTOCOL_MARKER, ModeTool};

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("mw-test"), ())
}

fn result_text(result: &TaToolResult) -> String {
    result.output()
}

fn tool_result(_name: &str, content: &str) -> TaToolResult {
    TaToolResult::success(content)
}

fn spec() -> Arc<MemoryProtocolSpec> {
    Arc::new(MemoryProtocolSpec {
        index_update_tool: "update_memory_md".into(),
        index_file_arg: "file".into(),
        index_file: "MEMORY.md".into(),
        write_tools: [
            "memory_store",
            "memory_forget",
            "memory_tree_ingest_document",
        ]
        .map(String::from)
        .to_vec(),
        read_tools: ["memory_recall"].map(String::from).to_vec(),
        mode_tool: Some(ModeTool {
            name: "memory_tree".into(),
            mode_arg: "mode".into(),
            write_mode: "ingest_document".into(),
        }),
        recall_tool: "memory_recall".into(),
    })
}

/// Drive one full tool cycle through the middleware: `before_tool` (captures
/// the arguments the result won't carry) then `after_tool`, correlated by a
/// shared call id. Returns the (possibly annotated) result.
async fn run_cycle(
    mw: &MemoryProtocolMiddleware,
    cx: &mut RunContext,
    name: &str,
    args: serde_json::Value,
    content: &str,
    error: Option<&str>,
) -> TaToolResult {
    let mut call = TaToolCall {
        id: "c1".into(),
        name: name.into(),
        arguments: args,
        invalid: None,
    };
    mw.before_tool(cx, &(), &mut call).await.unwrap();
    let mut result = match error {
        Some(error) => TaToolResult::error(error),
        None => tool_result(name, content),
    };
    let invocation = ToolInvocationIdentity::new("c1", name);
    mw.after_tool(cx, &(), &invocation, &mut result)
        .await
        .unwrap();
    result
}

#[tokio::test]
async fn memory_write_without_index_read_gets_a_corrective_note() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    let result = run_cycle(
        &mw,
        &mut cx,
        "memory_store",
        json!({}),
        "stored entry 42",
        None,
    )
    .await;
    assert!(
        result_text(&result).contains(MEMORY_PROTOCOL_MARKER),
        "a write with no preceding dedupe read should be annotated: {}",
        result_text(&result)
    );
    assert!(result_text(&result).contains("without first reading the memory index"));
    assert!(result_text(&result).contains("update_memory_md"));
    // The original tool output is preserved, guidance is appended.
    assert!(result_text(&result).starts_with("stored entry 42"));
}

#[tokio::test]
async fn memory_write_without_index_tool_still_gets_dedupe_guidance() {
    let mw = MemoryProtocolMiddleware::with_index_update_tool(spec(), false);
    let mut cx = ctx();
    let result = run_cycle(
        &mw,
        &mut cx,
        "memory_store",
        json!({}),
        "stored entry",
        None,
    )
    .await;
    let text = result_text(&result);
    assert!(text.contains("without first reading the memory index"));
    assert!(!text.contains("update_memory_md"));

    run_cycle(
        &mw,
        &mut cx,
        "memory_recall",
        json!({}),
        "found entry",
        None,
    )
    .await;
    let after_read = run_cycle(
        &mw,
        &mut cx,
        "memory_store",
        json!({}),
        "stored another",
        None,
    )
    .await;
    assert!(!result_text(&after_read).contains("update_memory_md"));
}

#[tokio::test]
async fn full_cycle_read_then_write_then_update_only_reminds_on_the_write() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();

    let read = run_cycle(&mw, &mut cx, "memory_recall", json!({}), "no dupes", None).await;
    assert!(
        !result_text(&read).contains(MEMORY_PROTOCOL_MARKER),
        "a read is not annotated"
    );

    let write = run_cycle(&mw, &mut cx, "memory_store", json!({}), "stored", None).await;
    assert!(result_text(&write).contains(MEMORY_PROTOCOL_MARKER));
    // The read preceded the write, so no missing-read complaint — just the
    // forward "sync the index" reminder.
    assert!(!result_text(&write).contains("without first reading the memory index"));

    let update = run_cycle(
        &mw,
        &mut cx,
        "update_memory_md",
        json!({ "file": "MEMORY.md" }),
        "index updated",
        None,
    )
    .await;
    assert!(
        !result_text(&update).contains(MEMORY_PROTOCOL_MARKER),
        "closing the cycle needs no guidance"
    );
}

#[tokio::test]
async fn skill_md_update_does_not_close_the_memory_cycle() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    run_cycle(&mw, &mut cx, "memory_recall", json!({}), "checked", None).await;
    run_cycle(&mw, &mut cx, "memory_store", json!({}), "stored", None).await;
    // update_memory_md targeting SKILL.md must NOT reconcile the MEMORY.md
    // index, so the stale-index warning is still owed at run end.
    run_cycle(
        &mw,
        &mut cx,
        "update_memory_md",
        json!({ "file": "SKILL.md" }),
        "skill updated",
        None,
    )
    .await;
    // A following write reports drift, proving pending was not cleared.
    let next = run_cycle(&mw, &mut cx, "memory_store", json!({}), "again", None).await;
    assert!(
        result_text(&next).contains("drifting"),
        "SKILL.md update must not mask the stale MEMORY.md index: {}",
        result_text(&next)
    );
    let mut run = AgentRun::new();
    // Still pending → after_agent takes its warn path without erroring.
    mw.after_agent(&mut cx, &(), &mut run).await.unwrap();
}

#[tokio::test]
async fn consolidated_memory_tree_ingest_is_treated_as_a_write() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    let ingest = run_cycle(
        &mw,
        &mut cx,
        "memory_tree",
        json!({ "mode": "ingest_document" }),
        "ingested",
        None,
    )
    .await;
    assert!(
        result_text(&ingest).contains(MEMORY_PROTOCOL_MARKER),
        "memory_tree ingest_document is a write and must be annotated: {}",
        result_text(&ingest)
    );
}

#[tokio::test]
async fn failed_memory_write_does_not_advance_the_protocol() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    let failed = run_cycle(
        &mw,
        &mut cx,
        "memory_store",
        json!({}),
        "disk full",
        Some("disk full"),
    )
    .await;
    // A failed write is not annotated and leaves nothing pending, so a later
    // run-end sweep must not warn about a stale index.
    assert!(!result_text(&failed).contains(MEMORY_PROTOCOL_MARKER));
    let mut run = AgentRun::new();
    // after_agent is a no-op warn path; it must not error.
    mw.after_agent(&mut cx, &(), &mut run).await.unwrap();
}

#[tokio::test]
async fn second_write_without_an_update_flags_index_drift() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    run_cycle(&mw, &mut cx, "memory_recall", json!({}), "checked", None).await;
    let first = run_cycle(&mw, &mut cx, "memory_store", json!({}), "a", None).await;
    assert!(!result_text(&first).contains("drifting"));

    // No update_memory_md between the two writes → the index is drifting.
    let second = run_cycle(&mw, &mut cx, "memory_store", json!({}), "b", None).await;
    assert!(
        result_text(&second).contains("drifting"),
        "a second unsynced write should flag index drift: {}",
        result_text(&second)
    );
}

#[tokio::test]
async fn protocol_state_is_isolated_per_run_even_with_a_shared_run_id() {
    let mw = MemoryProtocolMiddleware::new(spec());
    // Two concurrent runs carrying the same caller-supplied run id.
    let mut run_a = ctx();
    let mut run_b = ctx();
    // Run A reads the index (satisfies its dedupe read) …
    run_cycle(
        &mw,
        &mut run_a,
        "memory_recall",
        json!({}),
        "recalled",
        None,
    )
    .await;
    // … which must not excuse run B's write from the missing-read note.
    let write = run_cycle(&mw, &mut run_b, "memory_store", json!({}), "stored", None).await;
    assert!(result_text(&write).contains(MEMORY_PROTOCOL_MARKER));
}

#[tokio::test]
async fn after_agent_releases_state_left_by_calls_that_never_executed() {
    let mw = MemoryProtocolMiddleware::new(spec());
    let mut cx = ctx();
    // A memory call admission stopped: before_tool ran, after_tool never did.
    let mut call = TaToolCall {
        id: "rejected".into(),
        name: "memory_store".into(),
        arguments: json!({}),
        invalid: None,
    };
    mw.before_tool(&mut cx, &(), &mut call).await.unwrap();
    assert_eq!(mw.tracked_runs(), 1);
    let mut run = AgentRun::new();
    mw.after_agent(&mut cx, &(), &mut run).await.unwrap();
    assert_eq!(mw.tracked_runs(), 0);
}
