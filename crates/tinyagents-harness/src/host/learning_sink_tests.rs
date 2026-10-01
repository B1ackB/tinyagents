use super::*;

fn sample() -> TurnSummary {
    TurnSummary::new("thread-1", "agent-a")
        .with_text("hello", "hi")
        .with_tool("search")
}

#[test]
fn new_starts_empty_except_identity() {
    let summary = TurnSummary::new("thread-1", "agent-a");
    assert_eq!(summary.thread_id.as_str(), "thread-1");
    assert_eq!(summary.agent_id, "agent-a");
    assert!(summary.input.is_empty());
    assert!(summary.output.is_empty());
    assert!(!summary.used_tools());
    assert_eq!(summary.usage, Usage::default());
}

#[test]
fn record_tool_trims_and_drops_blanks() {
    let mut summary = TurnSummary::new("t", "a");
    summary.record_tool("  search  ");
    summary.record_tool("   ");
    summary.record_tool("");
    assert_eq!(summary.tools_invoked, vec!["search".to_string()]);
}

#[test]
fn record_tool_keeps_first_invocation_order_without_duplicates() {
    let mut summary = TurnSummary::new("t", "a");
    for name in ["search", "read", "search", " read ", "write"] {
        summary.record_tool(name);
    }
    assert_eq!(summary.tools_invoked, vec!["search", "read", "write"]);
    assert!(summary.used_tools());
}

#[test]
fn summary_round_trips_through_json() {
    let summary = sample().with_usage(Usage {
        input_tokens: 12,
        output_tokens: 3,
        total_tokens: 15,
        ..Usage::default()
    });
    let json = serde_json::to_string(&summary).expect("serializable");
    let decoded: TurnSummary = serde_json::from_str(&json).expect("deserializable");
    assert_eq!(decoded, summary);
}

#[test]
fn summary_deserializes_with_optional_fields_absent() {
    let decoded: TurnSummary =
        serde_json::from_str(r#"{"thread_id":"t","agent_id":"a","input":"","output":""}"#)
            .expect("defaults fill in");
    assert!(decoded.tools_invoked.is_empty());
    assert_eq!(decoded.usage, Usage::default());
}

#[tokio::test]
async fn noop_sink_accepts_a_summary() {
    let sink = NoopLearningSink::new();
    assert!(sink.on_turn_complete(&sample()).await.is_ok());
}

#[tokio::test]
async fn noop_sink_never_fails_and_keeps_no_state() {
    let sink = NoopLearningSink;
    for i in 0..3 {
        let summary = TurnSummary::new(format!("thread-{i}"), "agent-a");
        assert!(sink.on_turn_complete(&summary).await.is_ok());
    }
    // Copy semantics: the sink holds nothing, so a clone is interchangeable.
    let clone = sink;
    assert!(clone.on_turn_complete(&sample()).await.is_ok());
}

#[tokio::test]
async fn noop_sink_is_usable_as_a_trait_object() {
    // The runtime stores this capability as `Option<Arc<dyn LearningSink>>`;
    // this pins that the default impl is object-safe in that position.
    let sink: std::sync::Arc<dyn LearningSink> = std::sync::Arc::new(NoopLearningSink);
    assert!(sink.on_turn_complete(&sample()).await.is_ok());
}
