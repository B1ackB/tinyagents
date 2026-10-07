use super::*;
use std::sync::Arc;
use tinyagents_harness::testkit::FakeTool;

#[test]
fn declares_every_direct_tool() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(FakeTool::returning("get_time", "noon")));
    harness.register_tool(Arc::new(FakeTool::new("lookup")));
    let declarations = tool_declarations(&harness, false);
    let names: Vec<_> = declarations.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, vec!["get_time", "lookup"]);
    assert_eq!(declarations[0].description, "Fake tool: get_time");
    assert_eq!(declarations[0].parameters["type"], "object");
    // No deferred tools registered: including them changes nothing.
    assert_eq!(tool_declarations(&harness, true).len(), 2);
}

#[test]
fn an_empty_harness_declares_nothing() {
    let harness: AgentHarness<()> = AgentHarness::new();
    assert!(tool_declarations(&harness, true).is_empty());
}
