use super::*;

fn spec(name: &str) -> tinytools::ToolSpec {
    tinytools::ToolSpec::new(name, "d", serde_json::json!({}))
}

fn snapshot(names: &[&str]) -> ToolSnapshot {
    ToolSnapshot::new(names.iter().map(|n| spec(n)).collect()).unwrap()
}

fn names(snapshot: &ToolSnapshot) -> Vec<String> {
    snapshot.specs().iter().map(|s| s.name.clone()).collect()
}

#[test]
fn orchestrator_keeps_every_tool() {
    let tools = snapshot(&["read", SUBAGENT_JOBS_TOOL, "delegate_coder"]);
    let out = restrict_tools(&tools, SubagentRole::Orchestrator, None, &[]);
    assert_eq!(names(&out), names(&tools));
}

#[test]
fn leaf_loses_the_job_tools_and_host_named_delegation_tools() {
    let tools = snapshot(&[
        "read",
        SUBAGENT_JOBS_TOOL,
        SUBAGENT_MESSAGE_TOOL,
        "delegate_coder",
    ]);
    let extra = vec!["delegate_coder".to_owned()];
    let out = restrict_tools(&tools, SubagentRole::Leaf, None, &extra);
    assert_eq!(names(&out), ["read"]);
}

#[test]
fn a_child_can_never_widen_the_tool_set_it_inherits() {
    let child = snapshot(&["read", "write", "shell"]);
    let ceiling = snapshot(&["read", "grep"]);
    let out = restrict_tools(&child, SubagentRole::Orchestrator, Some(&ceiling), &[]);
    assert_eq!(names(&out), ["read"], "only the intersection survives");
}

#[test]
fn restriction_preserves_a_one_off_snapshot() {
    let tools = snapshot(&["read", SUBAGENT_JOBS_TOOL]).exact();
    let out = restrict_tools(&tools, SubagentRole::Leaf, None, &[]);
    assert_eq!(names(&out), ["read"]);
}

#[test]
fn framing_tells_a_leaf_it_cannot_delegate_and_who_it_reports_to() {
    let text = subagent_framing(SubagentRole::Leaf, 2, "summarise the diff");
    assert!(text.contains("subagent"));
    assert!(text.contains("depth 2"));
    assert!(text.contains("summarise the diff"));
    assert!(text.contains("parent"));
    assert!(text.contains("not the end user"));
    assert!(text.contains("background"));
    assert!(text.contains("cannot spawn"));
}

#[test]
fn framing_lets_an_orchestrator_delegate() {
    let text = subagent_framing(SubagentRole::Orchestrator, 1, "plan the work");
    assert!(text.contains("may delegate"));
    assert!(!text.contains("cannot spawn"));
}

#[test]
fn default_role_is_the_unrestricted_orchestrator() {
    assert_eq!(SubagentRole::default(), SubagentRole::Orchestrator);
}
