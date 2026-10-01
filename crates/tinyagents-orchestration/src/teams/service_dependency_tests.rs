use chrono::Utc;
use tinyagents_session::run_ledger::AgentTeamTaskStatus;

use super::*;

fn task(id: &str, depends_on: &[&str]) -> AgentTeamTask {
    let now = Utc::now();
    AgentTeamTask {
        id: id.to_string(),
        team_id: "team".to_string(),
        title: id.to_string(),
        objective: None,
        status: AgentTeamTaskStatus::Todo,
        owner_member_id: None,
        claimed_by_member_id: None,
        claim_token: None,
        depends_on: depends_on
            .iter()
            .map(|dependency| (*dependency).to_string())
            .collect(),
        gate_status: "pending".to_string(),
        gate_reason: None,
        evidence: Vec::new(),
        source_run_id: None,
        order_index: 0,
        created_at: now,
        updated_at: now,
    }
}

#[test]
fn rejects_self_dependency() {
    let error = validate_dependencies("task-self", &["task-self".to_string()], &[]).unwrap_err();
    assert_eq!(
        error.downcast::<TeamError>().unwrap(),
        TeamError::SelfDependency {
            task_id: "task-self".to_string()
        }
    );
}

#[test]
fn rejects_dependency_cycle() {
    let existing = vec![task("task-a", &["task-new"])];
    let error = validate_dependencies("task-new", &["task-a".to_string()], &existing).unwrap_err();
    assert_eq!(
        error.downcast::<TeamError>().unwrap(),
        TeamError::CyclicDependency
    );
}
