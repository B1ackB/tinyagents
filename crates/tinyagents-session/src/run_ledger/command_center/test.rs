use std::path::Path;

use chrono::{TimeZone, Utc};
use serde_json::json;
use tempfile::TempDir;

use super::control::plan_transition;
use super::*;
use crate::run_ledger::{
    AgentRun, AgentRunKind, AgentRunStatus, AgentRunUpsert, RunEventListRequest, get_agent_run,
    list_recent_run_events, upsert_agent_run,
};

fn no_names(_: &str) -> Option<String> {
    None
}


fn run_with(id: &str, status: AgentRunStatus, updated_secs: i64) -> AgentRun {
    AgentRun {
        id: id.to_string(),
        kind: AgentRunKind::Subagent,
        parent_run_id: None,
        parent_thread_id: Some("thread-1".to_string()),
        agent_id: Some("researcher".to_string()),
        status,
        prompt_ref: None,
        worker_thread_id: None,
        checkpoint_path: None,
        checkpoint: None,
        summary: None,
        error: None,
        metadata: json!({}),
        telemetry: None,
        started_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        updated_at: Utc.timestamp_opt(1_700_000_000 + updated_secs, 0).unwrap(),
        completed_at: None,
    }
}

#[test]
fn bucket_for_maps_every_status_to_its_group() {
    assert_eq!(
        bucket_for(AgentRunStatus::AwaitingUser),
        AgentWorkBucket::NeedsInput
    );
    assert_eq!(
        bucket_for(AgentRunStatus::Pending),
        AgentWorkBucket::Working
    );
    assert_eq!(
        bucket_for(AgentRunStatus::Running),
        AgentWorkBucket::Working
    );
    assert_eq!(bucket_for(AgentRunStatus::Paused), AgentWorkBucket::Working);
    assert_eq!(
        bucket_for(AgentRunStatus::Completed),
        AgentWorkBucket::Completed
    );
    assert_eq!(bucket_for(AgentRunStatus::Failed), AgentWorkBucket::Failed);
    assert_eq!(
        bucket_for(AgentRunStatus::Cancelled),
        AgentWorkBucket::Stopped
    );
    assert_eq!(
        bucket_for(AgentRunStatus::Interrupted),
        AgentWorkBucket::Stopped
    );
}

#[test]
fn build_view_always_emits_five_buckets_in_display_order() {
    let view = build_view(vec![], &no_names);
    assert_eq!(view.total, 0);
    let order: Vec<AgentWorkBucket> = view.groups.iter().map(|g| g.bucket).collect();
    assert_eq!(order, AgentWorkBucket::ALL.to_vec());
    assert!(view
        .groups
        .iter()
        .all(|g| g.rows.is_empty() && g.count == 0));
}

#[test]
fn build_view_groups_runs_into_correct_buckets() {
    let runs = vec![
        run_with("a", AgentRunStatus::Running, 1),
        run_with("b", AgentRunStatus::AwaitingUser, 2),
        run_with("c", AgentRunStatus::Completed, 3),
        run_with("d", AgentRunStatus::Failed, 4),
        run_with("e", AgentRunStatus::Cancelled, 5),
        run_with("f", AgentRunStatus::Pending, 6),
    ];
    let view = build_view(runs, &no_names);
    assert_eq!(view.total, 6);

    let group = |bucket: AgentWorkBucket| {
        view.groups
            .iter()
            .find(|g| g.bucket == bucket)
            .expect("bucket present")
    };
    assert_eq!(group(AgentWorkBucket::NeedsInput).count, 1);
    assert_eq!(group(AgentWorkBucket::Working).count, 2); // running + pending
    assert_eq!(group(AgentWorkBucket::Completed).count, 1);
    assert_eq!(group(AgentWorkBucket::Failed).count, 1);
    assert_eq!(group(AgentWorkBucket::Stopped).count, 1);
}

#[test]
fn build_view_preserves_input_order_within_a_bucket() {
    // Caller passes recent-first; projection must not reorder.
    let runs = vec![
        run_with("newest", AgentRunStatus::Running, 30),
        run_with("middle", AgentRunStatus::Running, 20),
        run_with("oldest", AgentRunStatus::Running, 10),
    ];
    let view = build_view(runs, &no_names);
    let working = view
        .groups
        .iter()
        .find(|g| g.bucket == AgentWorkBucket::Working)
        .unwrap();
    let ids: Vec<&str> = working.rows.iter().map(|r| r.run_id.as_str()).collect();
    assert_eq!(ids, vec!["newest", "middle", "oldest"]);
}

#[test]
fn project_row_defaults_telemetry_to_zero_when_absent() {
    let view = build_view(vec![run_with("x", AgentRunStatus::Completed, 1)], &no_names);
    let row = view
        .groups
        .iter()
        .flat_map(|g| &g.rows)
        .find(|r| r.run_id == "x")
        .unwrap();
    assert_eq!(row.input_tokens, 0);
    assert_eq!(row.output_tokens, 0);
    assert_eq!(row.cost_usd, 0.0);
    assert_eq!(row.tool_count, 0);
    assert_eq!(row.elapsed_ms, None);
    assert_eq!(row.status, "completed");
    assert_eq!(row.kind, "subagent");
}


fn seed_run(workspace_dir: &Path, id: &str, status: AgentRunStatus) {
    upsert_agent_run(
        workspace_dir,
        AgentRunUpsert {
            id: id.to_string(),
            kind: AgentRunKind::Subagent,
            parent_run_id: None,
            parent_thread_id: Some("thread-1".into()),
            agent_id: Some("researcher".into()),
            status,
            prompt_ref: None,
            worker_thread_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: if status == AgentRunStatus::Failed {
                Some("boom".into())
            } else {
                None
            },
            metadata: json!({}),
            started_at: None,
            completed_at: if status.is_terminal() {
                Some(Utc::now())
            } else {
                None
            },
        },
    )
    .unwrap();
}

// ---- pure planner -----------------------------------------------------

#[test]
fn parse_round_trips_known_verbs_and_rejects_unknown() {
    for verb in [
        ControlVerb::Stop,
        ControlVerb::Retry,
        ControlVerb::Continue,
        ControlVerb::FollowUp,
    ] {
        assert_eq!(ControlVerb::parse(verb.as_str()), Some(verb));
    }
    assert_eq!(ControlVerb::parse("nonsense"), None);
    assert_eq!(ControlVerb::parse(" stop "), Some(ControlVerb::Stop));
}

#[test]
fn message_requirement_is_verb_specific() {
    assert!(ControlVerb::Continue.requires_message());
    assert!(ControlVerb::FollowUp.requires_message());
    assert!(!ControlVerb::Stop.requires_message());
    assert!(!ControlVerb::Retry.requires_message());
}

#[test]
fn stop_allowed_only_while_non_terminal() {
    for status in [
        AgentRunStatus::Pending,
        AgentRunStatus::Running,
        AgentRunStatus::AwaitingUser,
        AgentRunStatus::Paused,
    ] {
        let plan = plan_transition(status, ControlVerb::Stop).unwrap();
        assert_eq!(plan.target_status, AgentRunStatus::Cancelled);
        assert_eq!(plan.event_type, "control_stopped");
    }
    for status in [
        AgentRunStatus::Completed,
        AgentRunStatus::Failed,
        AgentRunStatus::Cancelled,
        AgentRunStatus::Interrupted,
    ] {
        assert!(matches!(
            plan_transition(status, ControlVerb::Stop),
            Err(ControlError::InvalidTransition { .. })
        ));
    }
}

#[test]
fn retry_allowed_only_from_error_terminals() {
    for status in [
        AgentRunStatus::Failed,
        AgentRunStatus::Cancelled,
        AgentRunStatus::Interrupted,
    ] {
        let plan = plan_transition(status, ControlVerb::Retry).unwrap();
        assert_eq!(plan.target_status, AgentRunStatus::Pending);
        assert_eq!(plan.event_type, "control_retry");
    }
    for status in [
        AgentRunStatus::Pending,
        AgentRunStatus::Running,
        AgentRunStatus::AwaitingUser,
        AgentRunStatus::Paused,
        AgentRunStatus::Completed,
    ] {
        assert!(matches!(
            plan_transition(status, ControlVerb::Retry),
            Err(ControlError::InvalidTransition { .. })
        ));
    }
}

#[test]
fn continue_allowed_only_from_awaiting_user() {
    let plan = plan_transition(AgentRunStatus::AwaitingUser, ControlVerb::Continue).unwrap();
    assert_eq!(plan.target_status, AgentRunStatus::Running);
    assert_eq!(plan.event_type, "control_continued");
    for status in [
        AgentRunStatus::Pending,
        AgentRunStatus::Running,
        AgentRunStatus::Paused,
        AgentRunStatus::Completed,
        AgentRunStatus::Failed,
    ] {
        assert!(matches!(
            plan_transition(status, ControlVerb::Continue),
            Err(ControlError::InvalidTransition { .. })
        ));
    }
}

#[test]
fn follow_up_keeps_status_from_any_state() {
    for status in [
        AgentRunStatus::Running,
        AgentRunStatus::Completed,
        AgentRunStatus::Failed,
    ] {
        let plan = plan_transition(status, ControlVerb::FollowUp).unwrap();
        assert_eq!(plan.target_status, status);
        assert_eq!(plan.event_type, "control_follow_up");
    }
}

// ---- ledger-backed apply ---------------------------------------------

#[test]
fn stop_cancels_a_running_run_and_records_an_event() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::Running);

    let row = apply_control(&config, "run-1", ControlVerb::Stop, None, Some("manual", &no_names)).unwrap();
    assert_eq!(row.status, "cancelled");
    assert_eq!(row.bucket.as_str(), "stopped");
    assert_eq!(row.error.as_deref(), Some("manual"));

    let events = list_recent_run_events(
        &config,
        &RunEventListRequest {
            run_id: "run-1".into(),
            after_sequence: None,
            limit: None,
        },
    )
    .unwrap();
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].event_type, "control_stopped");
    assert_eq!(events.events[0].payload["toStatus"], "cancelled");
}

#[test]
fn retry_requeues_a_failed_run_and_clears_error() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::Failed);

    let row = apply_control(&config, "run-1", ControlVerb::Retry, None, None, &no_names).unwrap();
    assert_eq!(row.status, "pending");
    assert_eq!(row.bucket.as_str(), "working");
    // The stale failure reason is dropped (upsert COALESCE could not do this).
    assert_eq!(row.error, None);
}

#[test]
fn continue_resumes_an_awaiting_user_run() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::AwaitingUser);

    let row = apply_control(&config, "run-1", ControlVerb::Continue, Some("use the staging bucket"), None,, &no_names)
    .unwrap();
    assert_eq!(row.status, "running");
    assert_eq!(row.bucket.as_str(), "working");
}

#[test]
fn continue_without_message_is_rejected_before_touching_the_ledger() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::AwaitingUser);

    let err =
        apply_control(&config, "run-1", ControlVerb::Continue, Some("   "), None, &no_names).unwrap_err();
    assert!(matches!(err, ControlError::MessageRequired("continue")));
    // Status untouched.
    let run = get_agent_run(&config, "run-1")
        .unwrap()
        .unwrap();
    assert_eq!(run.status, AgentRunStatus::AwaitingUser);
}

#[test]
fn follow_up_records_an_event_without_changing_status() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::Completed);

    let row = apply_control(&config, "run-1", ControlVerb::FollowUp, Some("now summarize it"), None,, &no_names)
    .unwrap();
    assert_eq!(row.status, "completed");

    let events = list_recent_run_events(
        &config,
        &RunEventListRequest {
            run_id: "run-1".into(),
            after_sequence: None,
            limit: None,
        },
    )
    .unwrap();
    assert_eq!(events.events[0].event_type, "control_follow_up");
    assert_eq!(events.events[0].payload["message"], "now summarize it");
}

#[test]
fn invalid_transition_is_rejected() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    seed_run(&config, "run-1", AgentRunStatus::Completed);

    let err = apply_control(&config, "run-1", ControlVerb::Stop, None, None, &no_names).unwrap_err();
    assert!(matches!(
        err,
        ControlError::InvalidTransition {
            verb: "stop",
            status: "completed"
        }
    ));
}

#[test]
fn unknown_run_is_not_found() {
    let dir = TempDir::new().unwrap();
    let config = dir.path().to_path_buf();
    let err = apply_control(&config, "ghost", ControlVerb::Stop, None, None, &no_names).unwrap_err();
    assert!(matches!(err, ControlError::RunNotFound(id) if id == "ghost"));
}
