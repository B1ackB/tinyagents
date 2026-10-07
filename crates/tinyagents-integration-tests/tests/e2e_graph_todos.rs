//! End-to-end coverage for the per-thread todo list (`graph::todos`) exercised
//! through the public crate surface: a `MockModel`-driven agent loop calls the
//! `todo` tool, and the list persists on a shared `Store` addressed by the
//! run's thread id.

use std::sync::Arc;

use serde_json::json;

use tinyagents_graph::*;
use tinyagents_harness::context::RunConfig;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::store::{InMemoryStore, Store};
use tinyagents_harness::*;
use tinyagents_registry::*;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    tinyagents_harness::testkit::tool_call_response(ToolCall::new(id, name, arguments))
        .with_usage(Usage::new(7, 3))
}

fn text_response(text: &str) -> ModelResponse {
    tinyagents_harness::testkit::text_response(text).with_usage(Usage::new(4, 2))
}

#[tokio::test]
async fn model_drives_the_todo_tool_and_list_persists_to_the_thread() {
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::with_responses(vec![
                // First turn: write the plan via the `todo` tool.
                tool_call_response(
                    "call-1",
                    "todo",
                    json!({ "todos": [
                        { "content": "Write the integration test", "status": "in_progress" },
                        { "content": "Run it", "status": "pending" }
                    ] }),
                ),
                // Second turn: tick the first one off and start the next.
                tool_call_response(
                    "call-2",
                    "todo",
                    json!({ "todos": [
                        { "content": "Write the integration test", "status": "completed" },
                        { "content": "Run it", "status": "in_progress" }
                    ] }),
                ),
                text_response("done"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(TodoTool::new(store.clone())));

    let run = harness
        .invoke(
            &(),
            (),
            RunConfig::new("run").with_thread("thread-e2e"),
            vec![Message::user("track this work")],
        )
        .await
        .expect("agent run succeeds");

    assert_eq!(run.tool_calls, 2, "both todo tool calls executed");

    // The list persisted under the run's thread id, reachable via the public
    // programmatic surface, and reflects the last write.
    let snapshot = todo_store::list(&store, "thread-e2e")
        .await
        .expect("list reads");
    assert_eq!(snapshot.items.len(), 2);
    assert_eq!(snapshot.items[0].content, "Write the integration test");
    assert_eq!(snapshot.items[0].status, TodoStatus::Completed);
    assert_eq!(snapshot.items[1].status, TodoStatus::InProgress);
    assert_eq!(
        snapshot.markdown,
        "- [x] Write the integration test\n- [~] Run it"
    );

    // A different thread has its own (empty) list.
    let other = todo_store::list(&store, "other-thread")
        .await
        .expect("other list reads");
    assert!(other.items.is_empty());
}
