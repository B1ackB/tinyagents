//! Tool declarations for a live session, taken from a harness's registry.

use tinyagents_harness::AgentHarness;
use tinyliveagents::ToolDeclaration;

/// The declarations a live model should see for `harness`'s tools: every
/// directly exposed tool, plus the deferred ones when `include_deferred`.
/// Providers clean the schemas for their own dialect (Gemini, for one).
pub fn tool_declarations<State: Send + Sync, Ctx: Send + Sync>(
    harness: &AgentHarness<State, Ctx>,
    include_deferred: bool,
) -> Vec<ToolDeclaration> {
    let registry = harness.tools();
    let mut schemas = registry.schemas();
    if include_deferred {
        schemas.extend(registry.deferred_schemas());
    }
    schemas
        .into_iter()
        .map(|schema| ToolDeclaration::new(schema.name, schema.description, schema.parameters))
        .collect()
}

#[cfg(test)]
#[path = "declarations_tests.rs"]
mod tests;
