use super::*;

fn thread(id: &str) -> ThreadId {
    ThreadId::new(id)
}

#[test]
fn memory_id_round_trips_through_string_conversions() {
    let id = MemoryId::from("mem-1");
    assert_eq!(id.as_str(), "mem-1");
    assert_eq!(id.to_string(), "mem-1");
    assert_eq!(MemoryId::new("mem-1"), id);
    assert_eq!(MemoryId::from(String::from("mem-1")), id);
}

#[test]
fn recall_request_builders_compose() {
    let req = RecallRequest::new("hello")
        .with_thread(thread("t1"))
        .with_agent("planner")
        .with_limit(3);

    assert_eq!(req.query, "hello");
    assert_eq!(req.thread_id, Some(thread("t1")));
    assert_eq!(req.agent_id.as_deref(), Some("planner"));
    assert_eq!(req.limit, Some(3));
}

#[test]
fn new_memory_defaults_carry_no_provenance_fields() {
    let item = NewMemory::new("a fact");
    assert_eq!(item.text, "a fact");
    assert!(item.agent_id.is_none());
    assert!(item.thread_id.is_none());
    assert!(item.tags.is_empty());

    let tagged = NewMemory::new("a fact").with_tag("x").with_tag("y");
    assert_eq!(tagged.tags, vec!["x".to_string(), "y".to_string()]);
}

#[test]
fn memory_item_defaults_to_no_score_and_no_citation() {
    let item = MemoryItem::new("mem-1", "text");
    assert!(item.score.is_none());
    assert!(item.citation.is_none());

    let decorated = MemoryItem::new("mem-1", "text")
        .with_score(0.5)
        .with_citation("opaque://host/1");
    assert_eq!(decorated.score, Some(0.5));
    assert_eq!(decorated.citation.as_deref(), Some("opaque://host/1"));
}

#[test]
fn memory_item_serde_round_trips() {
    let item = MemoryItem::new("mem-1", "text").with_citation("opaque://host/1");
    let json = serde_json::to_string(&item).expect("serialize");
    let back: MemoryItem = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, item);
}

#[tokio::test]
async fn empty_store_recalls_nothing_without_erroring() {
    let store = InMemoryAgentMemory::new();
    assert!(store.is_empty());

    let items = store.recall(RecallRequest::new("anything")).await.unwrap();
    assert!(items.is_empty());
}

#[tokio::test]
async fn remember_then_recall_matches_case_insensitive_substrings() {
    let store = InMemoryAgentMemory::new();
    let id = store
        .remember(NewMemory::new("The Sky Is Blue"))
        .await
        .unwrap();

    let hit = store.recall(RecallRequest::new("sky is")).await.unwrap();
    assert_eq!(hit.len(), 1);
    assert_eq!(hit[0].id, id);
    assert_eq!(hit[0].text, "The Sky Is Blue");
    assert!(hit[0].score.is_none(), "test double must not invent scores");
    assert!(hit[0].citation.is_none());

    let miss = store.recall(RecallRequest::new("green")).await.unwrap();
    assert!(miss.is_empty());
}

#[tokio::test]
async fn remember_mints_a_distinct_id_per_write() {
    let store = InMemoryAgentMemory::new();
    let first = store.remember(NewMemory::new("same text")).await.unwrap();
    let second = store.remember(NewMemory::new("same text")).await.unwrap();

    assert_ne!(first, second);
    assert_eq!(store.len(), 2);
}

#[tokio::test]
async fn recall_preserves_insertion_order_and_honours_the_limit() {
    let store = InMemoryAgentMemory::new();
    for text in ["note one", "note two", "note three"] {
        store.remember(NewMemory::new(text)).await.unwrap();
    }

    let all = store.recall(RecallRequest::new("note")).await.unwrap();
    let texts: Vec<&str> = all.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(texts, vec!["note one", "note two", "note three"]);

    let capped = store
        .recall(RecallRequest::new("note").with_limit(2))
        .await
        .unwrap();
    assert_eq!(capped.len(), 2);
    assert_eq!(capped[0].text, "note one");
}

#[tokio::test]
async fn blank_query_returns_every_in_scope_record() {
    let store = InMemoryAgentMemory::new();
    store.remember(NewMemory::new("alpha")).await.unwrap();
    store.remember(NewMemory::new("beta")).await.unwrap();

    let all = store.recall(RecallRequest::new("")).await.unwrap();
    assert_eq!(all.len(), 2);
}

#[tokio::test]
async fn thread_scoping_hides_other_threads_but_keeps_unscoped_records() {
    let store = InMemoryAgentMemory::new();
    store
        .remember(NewMemory::new("scoped one").with_thread(thread("t1")))
        .await
        .unwrap();
    store
        .remember(NewMemory::new("scoped two").with_thread(thread("t2")))
        .await
        .unwrap();
    store.remember(NewMemory::new("scoped none")).await.unwrap();

    let scoped = store
        .recall(RecallRequest::new("scoped").with_thread(thread("t1")))
        .await
        .unwrap();
    let texts: Vec<&str> = scoped.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(texts, vec!["scoped one", "scoped none"]);

    let unscoped = store.recall(RecallRequest::new("scoped")).await.unwrap();
    assert_eq!(unscoped.len(), 3);
}

#[tokio::test]
async fn agent_scoping_hides_other_agents_but_keeps_unscoped_records() {
    let store = InMemoryAgentMemory::new();
    store
        .remember(NewMemory::new("fact a").with_agent("planner"))
        .await
        .unwrap();
    store
        .remember(NewMemory::new("fact b").with_agent("researcher"))
        .await
        .unwrap();
    store.remember(NewMemory::new("fact c")).await.unwrap();

    let scoped = store
        .recall(RecallRequest::new("fact").with_agent("planner"))
        .await
        .unwrap();
    let texts: Vec<&str> = scoped.iter().map(|i| i.text.as_str()).collect();
    assert_eq!(texts, vec!["fact a", "fact c"]);
}

#[tokio::test]
async fn thread_summary_is_none_even_after_remembering_into_that_thread() {
    let store = InMemoryAgentMemory::new();
    store
        .remember(NewMemory::new("something").with_thread(thread("t1")))
        .await
        .unwrap();

    assert!(store.thread_summary(&thread("t1")).await.unwrap().is_none());
}

#[tokio::test]
async fn clear_drops_every_record() {
    let store = InMemoryAgentMemory::new();
    store.remember(NewMemory::new("gone soon")).await.unwrap();
    assert_eq!(store.len(), 1);

    store.clear();
    assert!(store.is_empty());
    assert!(
        store
            .recall(RecallRequest::new(""))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn usable_behind_a_trait_object() {
    let store: Box<dyn AgentMemory> = Box::new(InMemoryAgentMemory::new());
    store.remember(NewMemory::new("dyn safe")).await.unwrap();
    assert_eq!(
        store.recall(RecallRequest::new("dyn")).await.unwrap().len(),
        1
    );
}
