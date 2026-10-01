use super::*;

fn exp(agent: &str, task: &str, outcome: &str, success: bool) -> Experience {
    let e = Experience::new(agent, task, outcome);
    if success { e.succeeded() } else { e }
}

#[test]
fn new_experience_defaults_to_failure() {
    let e = Experience::new("planner", "migrate schema", "permission denied");
    assert!(!e.success);
    assert!(e.succeeded().success);
}

#[test]
fn recallable_requires_agent_and_task() {
    assert!(Experience::new("planner", "migrate", "").is_recallable());
    assert!(!Experience::new("", "migrate", "ok").is_recallable());
    assert!(!Experience::new("planner", "   ", "ok").is_recallable());
}

#[test]
fn experience_round_trips_through_serde() {
    let e = exp("planner", "migrate schema", "worked on retry", true);
    let json = serde_json::to_string(&e).expect("serialize");
    let back: Experience = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(e, back);
}

#[tokio::test]
async fn empty_store_recalls_nothing() {
    let store = InMemoryExperienceStore::new();
    assert!(store.is_empty());
    let found = store
        .recall_for("planner", "anything")
        .await
        .expect("recall");
    assert!(found.is_empty());
}

#[tokio::test]
async fn records_and_recalls_by_agent_and_related_task() {
    let store = InMemoryExperienceStore::new();
    store
        .record(&exp("planner", "migrate the schema", "ok", true))
        .await
        .expect("record");
    store
        .record(&exp("planner", "brew coffee", "no kettle", false))
        .await
        .expect("record");
    store
        .record(&exp("writer", "migrate the schema", "n/a", true))
        .await
        .expect("record");

    let found = store
        .recall_for("planner", "schema migration")
        .await
        .expect("recall");
    assert_eq!(found.len(), 1, "other agent and unrelated task excluded");
    assert_eq!(found[0].task, "migrate the schema");
    assert!(found[0].success);
}

#[tokio::test]
async fn recall_returns_newest_first() {
    let store = InMemoryExperienceStore::new();
    store
        .record(&exp("planner", "deploy service", "first", false))
        .await
        .expect("record");
    store
        .record(&exp("planner", "deploy service", "second", true))
        .await
        .expect("record");

    let found = store.recall_for("planner", "deploy").await.expect("recall");
    let outcomes: Vec<&str> = found.iter().map(|e| e.outcome.as_str()).collect();
    assert_eq!(outcomes, vec!["second", "first"]);
}

#[tokio::test]
async fn punctuation_and_case_do_not_defeat_matching() {
    let store = InMemoryExperienceStore::new();
    store
        .record(&exp("planner", "Deploy the service.", "ok", true))
        .await
        .expect("record");
    let found = store
        .recall_for("planner", "deploy?")
        .await
        .expect("recall");
    assert_eq!(found.len(), 1);
}

#[tokio::test]
async fn unrecallable_records_are_dropped_without_error() {
    let store = InMemoryExperienceStore::new();
    store
        .record(&exp("", "migrate", "ok", true))
        .await
        .expect("record must not fail");
    assert_eq!(store.len(), 0);
}

#[tokio::test]
async fn oldest_records_are_evicted_at_capacity() {
    let store = InMemoryExperienceStore::with_capacity(2);
    for outcome in ["a", "b", "c"] {
        store
            .record(&exp("planner", "deploy service", outcome, false))
            .await
            .expect("record");
    }
    assert_eq!(store.len(), 2);
    let found = store.recall_for("planner", "deploy").await.expect("recall");
    let outcomes: Vec<&str> = found.iter().map(|e| e.outcome.as_str()).collect();
    assert_eq!(outcomes, vec!["c", "b"], "oldest record evicted");
}

#[tokio::test]
async fn zero_capacity_is_clamped_to_one() {
    let store = InMemoryExperienceStore::with_capacity(0);
    store
        .record(&exp("planner", "deploy service", "only", true))
        .await
        .expect("record");
    assert_eq!(store.len(), 1);
}
