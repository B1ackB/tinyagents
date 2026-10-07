use std::sync::Mutex;

use serde_json::json;

use super::*;

#[test]
fn truncate_keeps_head_and_tail_with_an_explicit_marker() {
    let text: String = ('a'..='z').collect();
    let (out, omitted) = truncate_head_tail(&text, 10);
    assert_eq!(omitted, 16);
    assert!(out.starts_with("abcde"));
    assert!(out.ends_with("vwxyz"));
    assert!(out.contains("[… 16 chars omitted …]"), "{out}");
}

#[test]
fn truncate_is_a_noop_within_the_cap_and_counts_chars_not_bytes() {
    assert_eq!(truncate_head_tail("héllo", 5), ("héllo".to_owned(), 0));
    let (out, omitted) = truncate_head_tail("ééééééé", 4);
    assert_eq!(omitted, 3);
    assert!(out.starts_with("éé") && out.ends_with("éé"));
}

#[tokio::test]
async fn the_default_policy_changes_nothing() {
    let applied = ResultPolicy::default().apply("t", "x".repeat(10_000).as_str(), None).await;
    assert_eq!(applied.text.len(), 10_000);
    assert_eq!(applied.omitted_chars, 0);
    assert!(applied.artifact.is_none() && applied.schema_error.is_none());
}

#[tokio::test]
async fn truncate_overflow_caps_the_visible_text() {
    let policy = ResultPolicy::new().with_max_chars(8);
    let applied = policy.apply("t", "0123456789abcdef", None).await;
    assert_eq!(applied.omitted_chars, 8);
    assert!(applied.text.contains("omitted"));
    assert!(applied.artifact.is_none());
}

struct MemoryStore(Mutex<Vec<(String, String)>>);

#[async_trait::async_trait]
impl ArtifactStore for MemoryStore {
    async fn store(&self, task_id: &str, content: &str) -> Result<ArtifactReference, String> {
        let mut stored = self.0.lock().unwrap();
        stored.push((task_id.to_owned(), content.to_owned()));
        Ok(ArtifactReference {
            id: format!("artifact-{}", stored.len()),
            media_type: Some("text/plain".into()),
            ..ArtifactReference::default()
        })
    }
}

#[tokio::test]
async fn artifact_overflow_stores_the_full_text_and_returns_a_preview() {
    let store = Arc::new(MemoryStore(Mutex::new(Vec::new())));
    let policy = ResultPolicy::new()
        .with_max_chars(6)
        .with_overflow(ResultOverflow::Artifact)
        .with_artifact_store(store.clone());
    let applied = policy.apply("task-1", "0123456789", None).await;
    assert_eq!(applied.artifact.as_ref().unwrap().id, "artifact-1");
    assert!(applied.text.contains("omitted"));
    assert_eq!(
        store.0.lock().unwrap()[0],
        ("task-1".to_owned(), "0123456789".to_owned()),
        "the full, untruncated output is what is stored"
    );
}

#[tokio::test]
async fn artifact_within_the_cap_is_not_stored() {
    let store = Arc::new(MemoryStore(Mutex::new(Vec::new())));
    let policy = ResultPolicy::new()
        .with_max_chars(100)
        .with_overflow(ResultOverflow::Artifact)
        .with_artifact_store(store.clone());
    let applied = policy.apply("t", "short", None).await;
    assert!(applied.artifact.is_none());
    assert!(store.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn artifact_without_a_store_falls_back_to_truncation_and_says_so() {
    let policy = ResultPolicy::new()
        .with_max_chars(4)
        .with_overflow(ResultOverflow::Artifact);
    let applied = policy.apply("t", "0123456789", None).await;
    assert!(applied.artifact.is_none());
    assert!(applied.text.contains("omitted"));
    assert!(applied.artifact_error.unwrap().contains("no artifact store"));
}

#[tokio::test]
async fn schema_validates_the_final_output_as_json() {
    let schema = json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]});
    let policy = ResultPolicy::new().with_schema(schema);
    assert!(policy.apply("t", r#"{"n": 3}"#, None).await.schema_error.is_none());
    let wrong = policy.apply("t", r#"{"n": "x"}"#, None).await;
    assert!(wrong.schema_error.unwrap().contains("integer"));
    let not_json = policy.apply("t", "plain prose", None).await;
    assert!(not_json.schema_error.unwrap().contains("not valid JSON"));
}

#[tokio::test]
async fn schema_prefers_the_runs_structured_value() {
    let policy = ResultPolicy::new().with_schema(json!({"type":"object","required":["k"]}));
    let applied = policy.apply("t", "prose", Some(&json!({"k": 1}))).await;
    assert!(applied.schema_error.is_none());
}
