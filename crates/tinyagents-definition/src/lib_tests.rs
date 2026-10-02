use super::*;

#[tokio::test]
async fn catalogue_is_stable_first_wins_and_absence_is_not_an_error() {
    let registry = InMemoryDefinitionRegistry::new(vec![
        AgentDefinition::new("planner", "Planner", "first").with_subagents(["research"]),
        AgentDefinition::new("planner", "Other", "ignored"),
        AgentDefinition::new("research", "Research", "second"),
    ]);
    assert_eq!(
        registry
            .resolve("planner")
            .await
            .unwrap()
            .unwrap()
            .description,
        "first"
    );
    assert_eq!(registry.list().await.unwrap().len(), 2);
    assert_eq!(
        registry.delegates_for("planner").await.unwrap(),
        ["research"]
    );
    assert!(registry.resolve("missing").await.unwrap().is_none());
    assert!(registry.delegates_for("missing").await.unwrap().is_empty());
}
