#[test]
fn dependency_direction_stays_one_way_and_host_free() {
    let manifest = include_str!("../Cargo.toml");
    assert!(!manifest.contains("openhuman"));

    for lower_layer in [
        include_str!("../../tinyagents-harness/Cargo.toml"),
        include_str!("../../tinyagents-runtime/Cargo.toml"),
    ] {
        assert!(
            !lower_layer.contains("tinyagents-orchestration"),
            "lower TinyAgents layers must not depend on orchestration"
        );
    }
}

#[test]
fn public_subagent_surface_compiles() {
    fn assert_executor<E: crate::subagent::SubagentExecutor>() {}
    let _ = assert_executor::<NeverExecutor>;
}

struct NeverExecutor;

#[async_trait::async_trait]
impl crate::subagent::SubagentExecutor for NeverExecutor {
    async fn execute(
        &self,
        _execution: crate::subagent::SubagentExecution,
    ) -> Result<crate::subagent::SubagentOutcome, crate::subagent::SubagentError> {
        unreachable!()
    }
}
