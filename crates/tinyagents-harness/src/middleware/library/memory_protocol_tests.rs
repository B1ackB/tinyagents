use std::sync::Arc;

use serde_json::json;

use super::*;

/// The tool vocabulary the OpenHuman host passes; the state machine itself is
/// vocabulary-agnostic.
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
        read_tools: [
            "memory_recall",
            "memory_vector_search",
            "memory_chunk_context",
            "memory_hybrid_search",
            "memory_tree_query_source",
            "memory_tree_search_entities",
            "memory_tree_fetch_leaves",
            "memory_tree_drill_down",
            "memory_tree_cover_window",
        ]
        .map(String::from)
        .to_vec(),
        mode_tool: Some(ModeTool {
            name: "memory_tree".into(),
            mode_arg: "mode".into(),
            write_mode: "ingest_document".into(),
        }),
        recall_tool: "memory_recall".into(),
    })
}

/// Empty arguments — for the name-only tools where arguments are irrelevant.
fn no_args() -> serde_json::Value {
    json!({})
}

#[test]
fn classifies_the_memory_tool_surface() {
    let a = no_args();
    assert_eq!(spec().classify("memory_store", &a), MemoryOp::Write);
    assert_eq!(spec().classify("memory_forget", &a), MemoryOp::Write);
    assert_eq!(
        spec().classify("memory_tree_ingest_document", &a),
        MemoryOp::Write
    );
    assert_eq!(spec().classify("memory_recall", &a), MemoryOp::IndexRead);
    assert_eq!(
        spec().classify("memory_vector_search", &a),
        MemoryOp::IndexRead
    );
    assert_eq!(
        spec().classify("memory_tree_search_entities", &a),
        MemoryOp::IndexRead
    );
    assert_eq!(spec().classify("send_message", &a), MemoryOp::Other);
    assert_eq!(spec().classify("file_write", &a), MemoryOp::Other);
}

#[test]
fn update_memory_md_only_closes_the_cycle_for_the_memory_index() {
    // A MEMORY.md edit reconciles the index; a SKILL.md edit does not and so
    // must not close the cycle or clear a pending write.
    assert_eq!(
        spec().classify("update_memory_md", &json!({ "file": "MEMORY.md" })),
        MemoryOp::IndexUpdate
    );
    assert_eq!(
        spec().classify("update_memory_md", &json!({ "file": "SKILL.md" })),
        MemoryOp::Other
    );

    // A SKILL.md update after a memory write leaves the index still owed.
    let mut t = MemoryProtocolTracker::new(spec());
    t.observe_tool("memory_recall", &no_args());
    t.observe_tool("memory_store", &no_args());
    t.observe_tool("update_memory_md", &json!({ "file": "SKILL.md" }));
    assert!(
        t.pending_index_update(),
        "a SKILL.md edit must not mask the stale MEMORY.md index"
    );
}

#[test]
fn consolidated_memory_tree_ingest_is_a_write() {
    // Every mode is a read except `ingest_document`, which writes.
    assert_eq!(
        spec().classify("memory_tree", &json!({ "mode": "ingest_document" })),
        MemoryOp::Write
    );
    assert_eq!(
        spec().classify("memory_tree", &json!({ "mode": "search_entities" })),
        MemoryOp::IndexRead
    );

    // An ingest via the consolidated tool obliges an index update.
    let mut t = MemoryProtocolTracker::new(spec());
    let obs = t.observe_tool("memory_tree", &json!({ "mode": "ingest_document" }));
    assert!(obs.was_write, "ingest_document mode is a durable write");
    assert!(t.pending_index_update());
}

#[test]
fn full_cycle_reports_no_violation() {
    let mut t = MemoryProtocolTracker::new(spec());
    assert_eq!(
        t.observe_tool("memory_recall", &no_args()),
        Default::default()
    );

    let write = t.observe_tool("memory_store", &no_args());
    assert!(write.was_write);
    assert!(!write.missing_index_read, "read preceded the write");
    assert!(!write.index_drift);
    assert!(t.pending_index_update());

    // Closing the cycle clears the pending index update.
    assert_eq!(
        t.observe_tool("update_memory_md", &json!({ "file": "MEMORY.md" })),
        Default::default()
    );
    assert!(!t.pending_index_update());
}

#[test]
fn write_without_index_read_is_flagged() {
    let mut t = MemoryProtocolTracker::new(spec());
    let obs = t.observe_tool("memory_store", &no_args());
    assert!(obs.was_write);
    assert!(obs.missing_index_read, "no dedupe read preceded the write");
    assert!(obs.needs_guidance());
    let note = obs
        .guidance(&spec(), "memory_store")
        .expect("guidance for a write");
    assert!(note.starts_with(MEMORY_PROTOCOL_MARKER));
    assert!(note.contains("without first reading the memory index"));
    assert!(note.contains("update_memory_md"));
}

#[test]
fn write_not_followed_by_update_is_detected_at_next_write() {
    let mut t = MemoryProtocolTracker::new(spec());
    t.observe_tool("memory_recall", &no_args());
    let first = t.observe_tool("memory_store", &no_args());
    assert!(!first.index_drift);
    assert!(t.pending_index_update());

    // A second write with no intervening update_memory_md: the index is
    // drifting from the store.
    let second = t.observe_tool("memory_store", &no_args());
    assert!(second.index_drift, "prior write never synced the index");
    let note = second.guidance(&spec(), "memory_store").unwrap();
    assert!(note.contains("drifting"));
}

#[test]
fn pending_index_update_survives_until_update_at_run_end() {
    let mut t = MemoryProtocolTracker::new(spec());
    t.observe_tool("memory_recall", &no_args());
    t.observe_tool("memory_store", &no_args());
    // Intervening non-memory tool calls don't clear the obligation.
    t.observe_tool("send_message", &no_args());
    assert!(
        t.pending_index_update(),
        "index update still owed at run end"
    );
}

#[test]
fn update_arms_a_fresh_cycle_that_expects_a_new_read() {
    let mut t = MemoryProtocolTracker::new(spec());
    t.observe_tool("memory_recall", &no_args());
    t.observe_tool("memory_store", &no_args());
    t.observe_tool("update_memory_md", &json!({ "file": "MEMORY.md" }));

    // Next cycle: a write with no fresh read is flagged again.
    let obs = t.observe_tool("memory_store", &no_args());
    assert!(
        obs.missing_index_read,
        "each cycle needs its own dedupe read"
    );
}

#[test]
fn reads_and_other_ops_need_no_guidance() {
    let mut t = MemoryProtocolTracker::new(spec());
    assert!(!t.observe_tool("memory_recall", &no_args()).needs_guidance());
    assert!(!t.observe_tool("send_message", &no_args()).needs_guidance());
    assert!(
        t.observe_tool("memory_recall", &no_args())
            .guidance(&spec(), "memory_recall")
            .is_none()
    );
}

/// The corrective notes are prompt text the model reads: pin the exact bytes.
#[test]
fn guidance_text_is_stable() {
    let spec = spec();
    let mut t = MemoryProtocolTracker::new(spec.clone());
    let first = t.observe_tool("memory_store", &no_args());
    assert_eq!(
        first.guidance(&spec, "memory_store").unwrap(),
        "[memory-protocol] `memory_store` wrote to memory without first reading the memory index to check for duplicates. Before creating entries, recall existing memory (e.g. `memory_recall`) so you don't store a near-duplicate. After mutating memory, call `update_memory_md` to keep the MEMORY.md index in sync."
    );
    let second = t.observe_tool("memory_store", &no_args());
    assert_eq!(
        second.guidance(&spec, "memory_store").unwrap(),
        "[memory-protocol] `memory_store` wrote to memory without first reading the memory index to check for duplicates. Before creating entries, recall existing memory (e.g. `memory_recall`) so you don't store a near-duplicate. A previous memory write was never followed by `update_memory_md`, so the MEMORY.md index is drifting from stored memory. Reconcile it now. After mutating memory, call `update_memory_md` to keep the MEMORY.md index in sync."
    );
    assert_eq!(
        second
            .guidance_with_index_update(&spec, "memory_store", false)
            .unwrap(),
        "[memory-protocol] `memory_store` wrote to memory without first reading the memory index to check for duplicates. Before creating entries, recall existing memory (e.g. `memory_recall`) so you don't store a near-duplicate."
    );
}
