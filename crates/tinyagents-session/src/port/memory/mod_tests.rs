use super::*;
use crate::testkit::conformance::{session_store_conformance, session_store_isolation_conformance};
use crate::transcript::{SessionRef, TranscriptTurn};

#[tokio::test]
async fn in_memory_stores_meet_the_session_store_contract() {
    session_store_conformance(&InMemorySessionStores::new()).await;
}

#[tokio::test]
async fn in_memory_stores_keep_agents_apart() {
    session_store_isolation_conformance(&InMemorySessionStores::new()).await;
}

#[test]
fn an_agent_gets_the_same_stores_every_time() {
    let provider = InMemorySessionStores::new();
    let first = provider.for_agent("a");
    let again = provider.for_agent("a");
    assert!(Arc::ptr_eq(&first.turn_states, &again.turn_states));
    assert_eq!(
        first.transcripts.destination_key(),
        again.transcripts.destination_key()
    );
    assert!(format!("{provider:?}").contains("agents: 1"));
    assert!(format!("{first:?}").contains("AgentStores"));
    assert!(
        provider
            .destination_key()
            .is_some_and(|key| key.starts_with("memory://"))
    );
}

#[test]
fn recovery_interrupts_every_agents_turns_in_flight() {
    let provider = InMemorySessionStores::new();
    for agent in ["a", "b"] {
        provider
            .for_agent(agent)
            .turn_states
            .put(&TurnState::started("t", "r", 4, "2026-01-01T00:00:00Z"))
            .unwrap();
    }
    provider.recover().unwrap();
    for agent in ["a", "b"] {
        let turn = provider.for_agent(agent).turn_states.get("t").unwrap();
        assert_eq!(turn.map(|t| t.lifecycle), Some(TurnLifecycle::Interrupted));
    }
    // Shared behind an Arc, it is still the same provider.
    let shared: Arc<dyn SessionStoreProvider> = Arc::new(provider);
    assert!(shared.recover().is_ok());
    assert!(shared.destination_key().is_some());
    assert!(
        Arc::new(shared.clone())
            .for_agent("a")
            .turn_states
            .get("t")
            .unwrap()
            .is_some()
    );
}

#[test]
fn completed_turns_are_kept_up_to_the_retention_limit() {
    let turns = InMemoryTurnStates::default();
    for minute in 0..(COMPLETED_RETENTION + 3) {
        let mut turn = TurnState::started(
            "t",
            format!("r{minute}"),
            4,
            format!("2026-01-01T00:{minute:02}:00Z"),
        );
        turn.lifecycle = TurnLifecycle::Completed;
        turns.put(&turn).unwrap();
    }
    let kept = turns.list_thread("t").unwrap();
    assert_eq!(kept.len(), COMPLETED_RETENTION);
    assert_eq!(kept[0].request_id, format!("r{}", COMPLETED_RETENTION + 2));
    // A conditional completed write prunes too, and a missing turn does not
    // settle.
    let mut late = TurnState::started("t", "late", 4, "2026-01-01T01:00:00Z");
    late.lifecycle = TurnLifecycle::Completed;
    assert!(turns.put_unless_completed(&late).unwrap());
    assert_eq!(turns.list_thread("t").unwrap().len(), COMPLETED_RETENTION);
    assert!(
        !turns
            .settle_turn("t", "missing", TurnLifecycle::Completed, "now")
            .unwrap()
    );
    assert!(
        !turns
            .settle_turn("t", "late", TurnLifecycle::Interrupted, "now")
            .unwrap(),
        "a terminal turn does not settle again"
    );
    assert!(turns.delete("t").unwrap());
    assert!(!turns.delete("t").unwrap());
}

#[test]
fn the_newest_root_wins_and_sub_agents_are_never_roots() {
    let locator = InMemoryTranscriptLocator::new("test");
    let meta = |agent: &str| TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: agent.to_string(),
        agent_id: Some(agent.to_string()),
        agent_type: None,
        dispatcher: "native".to_string(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".to_string(),
        updated: "2026-01-01T00:00:00Z".to_string(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("t".to_string()),
        task_id: None,
    };
    let write = |stem: &str, meta: TranscriptMeta, text: &str| {
        let message = crate::transcript::TranscriptMessage::user(text);
        locator
            .open_stem(stem, meta.clone())
            .unwrap()
            .append_turn(TranscriptTurn {
                prev: &[],
                next: &[message],
                meta: &meta,
                turn_usage: None,
                request_id: None,
                tools: None,
            })
            .unwrap();
    };
    write("old", meta("planner"), "old");
    write("new", meta("planner"), "new");
    write("new__child", meta("planner"), "child");
    let content = |found: Option<Arc<dyn TranscriptRead>>| {
        found.unwrap().read_session().unwrap().unwrap().messages[0]
            .content
            .clone()
    };
    assert_eq!(content(locator.latest_for_agent("planner")), "new");
    assert_eq!(content(locator.root_for_thread("t")), "new");
    assert!(locator.root_for_thread("  ").is_none());
    assert!(locator.latest_for_agent("nobody").is_none());
    // An unwritten probe is never a root.
    let _probe = locator
        .open_stem(
            &crate::transcript::session_stem(&SessionRef::root("p")),
            meta("planner"),
        )
        .unwrap();
    assert_eq!(content(locator.root_for_thread("t")), "new");
    // A partial goes to the newest root and stays out of the replay.
    assert!(
        locator
            .append_interrupted_partial("t", Some("planner"), &TranscriptPartial::new("par"), None)
            .unwrap()
    );
    assert!(
        !locator
            .append_interrupted_partial("t", Some("other"), &TranscriptPartial::new("x"), None)
            .unwrap()
    );
}
