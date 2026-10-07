use std::time::Duration;

use super::*;
use crate::testkit::conformance::{
    checkpointer_concurrent_contract, checkpointer_contract, checkpointer_lineage_contract,
    checkpointer_writes_contract,
};
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};

fn docs(storage: &MemoryStorage, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn sample(thread: &str, id: &str, _parent: Option<&str>, step: i32) -> Checkpoint<i32> {
    Checkpoint::new(step, Vec::new())
        .with_thread_id(thread.to_string())
        .with_checkpoint_id(id.to_string())
}

fn checkpointer() -> DriverCheckpointer<i32> {
    DriverCheckpointer::new(docs(&MemoryStorage::new(), "local"))
}

#[tokio::test]
async fn passes_the_checkpointer_contract() {
    checkpointer_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_writes_contract() {
    checkpointer_writes_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_lineage_contract() {
    checkpointer_lineage_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_concurrent_contract() {
    checkpointer_concurrent_contract(Arc::new(checkpointer())).await;
}

#[tokio::test]
async fn scopes_and_prefixes_keep_threads_apart() {
    let storage = MemoryStorage::new();
    let alice = DriverCheckpointer::<i32>::new(docs(&storage, "alice"));
    let bob = DriverCheckpointer::<i32>::new(docs(&storage, "bob"));
    let other = DriverCheckpointer::<i32>::with_prefix(docs(&storage, "alice"), "other");
    let checkpoint = sample("t", "c1", None, 1);
    alice.put(checkpoint).await.unwrap();
    assert!(bob.get("t", None).await.unwrap().is_none());
    assert!(other.get("t", None).await.unwrap().is_none());
    assert_eq!(alice.list_threads().await.unwrap(), vec!["t".to_owned()]);
    assert!(bob.list_threads().await.unwrap().is_empty());
    assert!(format!("{alice:?}").contains("graph_checkpoints"));
    let _ = alice.clone();
}

#[tokio::test]
async fn leases_follow_the_claim_protocol() {
    let cp = checkpointer();
    let minute = Duration::from_secs(60);
    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "live lease is refused"
    );
    assert!(
        cp.try_claim("t", "a", minute).await.unwrap(),
        "same owner re-claims"
    );
    assert!(cp.renew("t", "a", minute).await.unwrap());
    assert!(
        !cp.renew("t", "b", minute).await.unwrap(),
        "only the owner renews"
    );
    assert!(!cp.renew("missing", "a", minute).await.unwrap());

    cp.release("t", "b").await.unwrap();
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "foreign release is a no-op"
    );
    cp.release("t", "a").await.unwrap();
    cp.release("t", "a").await.unwrap();
    assert!(
        cp.try_claim("t", "b", minute).await.unwrap(),
        "released lease is free"
    );

    assert!(cp.try_claim("z", "dead", Duration::ZERO).await.unwrap());
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(
        !cp.renew("z", "dead", minute).await.unwrap(),
        "expired lease cannot renew"
    );
    assert!(
        cp.try_claim("z", "new", minute).await.unwrap(),
        "expired lease is reclaimable"
    );
}

#[test]
fn long_keys_are_hashed_and_tuples_never_collide() {
    assert_ne!(key(&["a/b", "c"]), key(&["a", "b/c"]));
    let long = "t".repeat(500);
    let hashed = key(&[&long]);
    assert!(hashed.starts_with("h:") && hashed.len() < 64, "{hashed}");
    assert_ne!(hashed, key(&[&"u".repeat(500)]));
}

#[test]
fn driver_failures_are_checkpoint_errors() {
    let error = map_error(StorageError::unavailable("busy"));
    assert!(matches!(error, TinyAgentsError::Checkpoint(ref m) if m.contains("busy")));
}

#[tokio::test]
async fn a_corrupt_record_is_a_checkpoint_error() {
    let storage = MemoryStorage::new();
    let docs = docs(&storage, "local");
    let cp = DriverCheckpointer::<i32>::new(Arc::clone(&docs));
    cp.put(sample("t", "c1", None, 1)).await.unwrap();
    let stored = cp.thread_docs("t").await.unwrap().remove(0);
    docs.put(
        "graph_checkpoints",
        &stored.id,
        json!({"thread": "t", "seq": 0, "checkpoint_id": "c1", "record": "nope"}),
        Precondition::None,
    )
    .await
    .unwrap();
    let error = cp.get("t", None).await.unwrap_err();
    assert!(matches!(error, TinyAgentsError::Checkpoint(_)), "{error:?}");
}
