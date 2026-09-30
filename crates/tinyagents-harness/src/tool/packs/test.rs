//! Mechanism tests over a two-pack test catalog. The product table's own
//! invariants (membership, owners, guides) are tested by the host that owns it.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinytools::{PermissionLevel, Tool, ToolResult, ToolSpec, ToolTimeout};

use super::*;

const MARKER: &str = "[not_found]";

static PACKS: &[ToolPack] = &[
    ToolPack {
        id: "alpha",
        summary: "Alpha things.",
        tools: &["a_one", "a_two"],
        owners: &["alpha_agent"],
        guide: "Use alpha carefully.",
    },
    ToolPack {
        id: "beta",
        summary: "Beta things.",
        tools: &["b_one"],
        owners: &[],
        guide: "",
    },
];

const CATALOG: PackCatalog = PackCatalog::new(PACKS, MARKER);

struct FakeTool {
    name: &'static str,
    level: PermissionLevel,
    external: bool,
    timeout: ToolTimeout,
}

impl FakeTool {
    fn plain(name: &'static str) -> Self {
        Self {
            name,
            level: PermissionLevel::ReadOnly,
            external: false,
            timeout: ToolTimeout::Inherit,
        }
    }
}

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fake"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"marker": {"type": "string"}}})
    }
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(format!("{}:{}", self.name, args)))
    }
    fn permission_level(&self) -> PermissionLevel {
        self.level
    }
    fn external_effect_with_args(&self, _args: &Value) -> bool {
        self.external
    }
    fn timeout_policy(&self, _args: &Value) -> ToolTimeout {
        self.timeout
    }
}

/// A durable registry holding `tools` plus a bound `use_skill`.
fn bound(tools: Vec<FakeTool>) -> Arc<Vec<Box<dyn Tool>>> {
    let handle = PackRegistryHandle::default();
    let mut all: Vec<Box<dyn Tool>> = tools.into_iter().map(|t| Box::new(t) as _).collect();
    all.push(Box::new(UseSkillTool::new(handle.clone(), CATALOG)));
    let all = Arc::new(all);
    handle.bind(Arc::downgrade(&all));
    all
}

fn handle_of(tools: &[Box<dyn Tool>]) -> &PackRegistryHandle {
    find(tools, USE_SKILL)
        .host_extension()
        .and_then(|any| any.downcast_ref::<PackRegistryHandle>())
        .expect("use_skill carries its handle as a host extension")
}

fn find<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> &'a dyn Tool {
    tools
        .iter()
        .find(|t| t.name() == name)
        .map(AsRef::as_ref)
        .unwrap_or_else(|| panic!("{name} missing"))
}

fn text(result: &ToolResult) -> String {
    format!("{:?}", result.content)
}

#[test]
fn the_use_skill_declaration_is_byte_stable() {
    let tools = bound(vec![]);
    let tool = find(&tools, USE_SKILL);
    assert_eq!(tool.name(), "use_skill");
    assert_eq!(
        tool.description(),
        "Reach a skill's tools. Their names, descriptions and argument schemas are NOT in \
         your context until you ask for them: call this with `skill` alone to see them, then \
         again with `skill` + `tool` + `args` to run one.\n\nSkills:\n\
         - `alpha` — Alpha things.\n- `beta` — Beta things.\n"
    );
    assert_eq!(
        tool.parameters_schema(),
        json!({
            "type": "object",
            "properties": {
                "skill": { "type": "string", "enum": ["alpha", "beta"], "description": "Skill to read or run a tool from." },
                "tool": { "type": "string", "description": "Tool to run. Omit to list the skill's tools and their arguments instead." },
                "args": {
                    "type": "object",
                    "description": "The tool's own arguments, as documented in the listing.",
                    "additionalProperties": true
                }
            },
            "required": ["skill"]
        })
    );
}

#[tokio::test]
async fn a_skill_alone_renders_guide_and_the_schema_of_a_present_tool() {
    let tools = bound(vec![FakeTool::plain("a_one")]);
    let result = find(&tools, USE_SKILL)
        .execute(json!({"skill": "alpha"}))
        .await
        .unwrap();
    assert!(!result.is_error);
    let rendered = result.text();
    assert!(rendered.starts_with("# Skill `alpha`\n\nAlpha things.\n\nUse alpha carefully.\n\n"));
    assert!(rendered.contains("## `a_one`\n\nfake\n\n```json\n"));
    assert!(
        rendered.contains(r#"{"properties":{"marker":{"type":"string"}},"type":"object"}"#)
            || rendered.contains(r#""marker""#)
    );
    // A pack tool this session lacks is skipped, not fatal.
    assert!(!rendered.contains("a_two"));
}

#[tokio::test]
async fn an_unknown_skill_is_a_marked_error_listing_the_alternatives() {
    let tools = bound(vec![FakeTool::plain("a_one")]);
    let result = find(&tools, USE_SKILL)
        .execute(json!({"skill": "nope"}))
        .await
        .unwrap();
    assert!(result.is_error);
    let message = result.text();
    assert!(message.starts_with(&format!("{MARKER} Unknown skill `nope`. Available:\n")));
    assert!(message.contains("- `alpha` — Alpha things."));
}

#[tokio::test]
async fn dispatch_forwards_args_to_the_packed_tool() {
    let tools = bound(vec![FakeTool::plain("a_one")]);
    let result = find(&tools, USE_SKILL)
        .execute(json!({"skill": "alpha", "tool": "a_one", "args": {"marker": "x"}}))
        .await
        .unwrap();
    assert!(!result.is_error);
    assert!(
        text(&result).contains(r#"a_one:{"marker":"x"}"#.replace('"', "\\\"").as_str())
            || text(&result).contains("marker")
    );
}

#[tokio::test]
async fn a_tool_from_another_skill_is_refused() {
    // Cross-skill dispatch would make `skill` decoration and let a harmless
    // skill reach a dangerous tool.
    let tools = bound(vec![FakeTool {
        level: PermissionLevel::Dangerous,
        ..FakeTool::plain("b_one")
    }]);
    let result = find(&tools, USE_SKILL)
        .execute(json!({"skill": "alpha", "tool": "b_one", "args": {}}))
        .await
        .unwrap();
    assert!(result.is_error, "cross-skill dispatch was admitted");
    assert!(result.text().starts_with(&format!(
        "{MARKER} No tool `b_one` in skill `alpha`. Call `use_skill {{ \"skill\": \"alpha\" }}` to see what it contains."
    )));
}

#[test]
fn permission_level_is_the_inner_tools_not_the_proxys() {
    let tools = bound(vec![FakeTool {
        level: PermissionLevel::Dangerous,
        ..FakeTool::plain("a_one")
    }]);
    let use_skill = find(&tools, USE_SKILL);
    assert_eq!(
        use_skill.permission_level_with_args(&json!({"skill": "alpha", "tool": "a_one"})),
        PermissionLevel::Dangerous
    );
    // Naming no tool (or an empty one) only reads a schema.
    for args in [
        json!({"skill": "alpha"}),
        json!({"skill": "alpha", "tool": ""}),
    ] {
        assert_eq!(
            use_skill.permission_level_with_args(&args),
            PermissionLevel::ReadOnly
        );
    }
    // Unresolvable: the ceiling, never a permissive default.
    assert_eq!(
        use_skill.permission_level_with_args(&json!({"skill": "alpha", "tool": "ghost"})),
        PermissionLevel::Dangerous
    );
}

#[test]
fn external_effect_and_timeout_are_forwarded() {
    let tools = bound(vec![FakeTool {
        external: true,
        timeout: ToolTimeout::Unbounded,
        ..FakeTool::plain("a_one")
    }]);
    let use_skill = find(&tools, USE_SKILL);
    let call = json!({"skill": "alpha", "tool": "a_one"});
    assert!(use_skill.external_effect_with_args(&call));
    assert_eq!(use_skill.timeout_policy(&call), ToolTimeout::Unbounded);
    let ghost = json!({"skill": "alpha", "tool": "ghost"});
    assert!(!use_skill.external_effect_with_args(&ghost));
    assert_eq!(use_skill.timeout_policy(&ghost), ToolTimeout::Inherit);
}

#[test]
fn an_unbound_handle_degrades_closed() {
    let tool = UseSkillTool::new(PackRegistryHandle::default(), CATALOG);
    assert_eq!(tool.permission_level(), PermissionLevel::Dangerous);
}

#[tokio::test]
async fn an_unbound_handle_reports_the_registry_unavailable() {
    let tool = UseSkillTool::new(PackRegistryHandle::default(), CATALOG);
    let result = tool.execute(json!({"skill": "alpha"})).await.unwrap();
    assert!(result.is_error);
    assert!(result.text().contains("registry is not available"));
}

/// The synthesised set is a second registry; a packed tool living only there
/// must be reachable, and rebinding it must repoint the handle.
#[tokio::test]
async fn a_tool_in_the_synthesised_set_is_reachable_and_rebinding_repoints() {
    let durable = bound(vec![]);
    let first: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(FakeTool::plain("a_one"))]);
    handle_of(&durable).bind_synthesized(Arc::downgrade(&first));
    let use_skill = find(&durable, USE_SKILL);

    let ran = use_skill
        .execute(json!({"skill": "alpha", "tool": "a_one", "args": {}}))
        .await
        .unwrap();
    assert!(!ran.is_error, "{}", ran.text());

    let second: Arc<Vec<Box<dyn Tool>>> = Arc::new(vec![Box::new(FakeTool::plain("a_two"))]);
    handle_of(&durable).bind_synthesized(Arc::downgrade(&second));
    drop(first);
    let ran = use_skill
        .execute(json!({"skill": "alpha", "tool": "a_two", "args": {}}))
        .await
        .unwrap();
    assert!(!ran.is_error, "handle did not follow the rebind");
    let stale = use_skill
        .execute(json!({"skill": "alpha", "tool": "a_one", "args": {}}))
        .await
        .unwrap();
    assert!(stale.is_error, "a dropped tool stayed reachable");
}

#[test]
fn resolve_registry_for_enforces_pack_membership() {
    let durable = bound(vec![FakeTool::plain("a_one")]);
    let handle = handle_of(&durable);
    assert!(
        handle
            .resolve_registry_for(&CATALOG, "alpha", "a_one")
            .is_some()
    );
    assert!(
        handle
            .resolve_registry_for(&CATALOG, "beta", "a_one")
            .is_none()
    );
    assert!(
        handle
            .resolve_registry_for(&CATALOG, "alpha", "ghost")
            .is_none()
    );
}

#[tokio::test]
async fn the_filtered_listing_omits_tools_the_session_cannot_call() {
    let tools = bound(vec![FakeTool::plain("a_one"), FakeTool::plain("a_two")]);
    let handle = handle_of(&tools);
    let listing = render_pack_filtered(&CATALOG, "alpha", handle, &|name| name == "a_one", "")
        .expect("one callable tool");
    assert!(listing.contains("## `a_one`"));
    assert!(!listing.contains("## `a_two`"));

    let denied = render_pack_filtered(&CATALOG, "alpha", handle, &|_| false, "Hand off instead.")
        .unwrap_err();
    assert_eq!(
        denied,
        "Skill `alpha` has no tools available in this session. Hand off instead."
    );
    let bare = render_pack_filtered(&CATALOG, "alpha", handle, &|_| false, "").unwrap_err();
    assert_eq!(
        bare,
        "Skill `alpha` has no tools available in this session."
    );
}

#[test]
fn no_such_pack_tool_renders_each_shape() {
    let base = |callable: Vec<&'static str>, route: &str| NoSuchPackTool {
        not_found_marker: MARKER,
        skill: "alpha",
        tool: "x",
        callable,
        route: route.to_string(),
    };
    assert_eq!(
        base(vec!["a_one", "a_two"], "ignored").render(),
        "[not_found] There is no tool `x` in skill `alpha`. The tools in it you can call: `a_one`, `a_two`."
    );
    assert_eq!(
        base(vec![], "Go elsewhere.").render(),
        "[not_found] There is no tool `x` in skill `alpha`. Go elsewhere."
    );
    assert_eq!(
        base(vec![], "").render(),
        "[not_found] There is no tool `x` in skill `alpha`. Nothing in it is available in this session."
    );
}

#[test]
fn route_sentence_prefers_a_callable_tool_and_falls_back_to_owners() {
    assert_eq!(
        route_sentence(&["a".to_string(), "b".to_string()], &["o"]),
        "Call `a` or `b` instead — that agent owns these tools and runs them directly. Do not retry this skill."
    );
    assert_eq!(
        route_sentence(&[], &["x", "y"]),
        "These tools belong to `x` or `y`; hand the task to one of them rather than calling directly."
    );
    assert!(route_sentence(&[], &[]).is_empty());
}

#[test]
fn scoping_narrows_the_enum_and_the_index_or_drops_the_tool() {
    let tools = bound(vec![]);
    let mut spec = ToolSpec {
        name: USE_SKILL.into(),
        description: find(&tools, USE_SKILL).description().to_string(),
        parameters: find(&tools, USE_SKILL).parameters_schema(),
    };
    assert!(scope_use_skill_spec(&CATALOG, &mut spec, &|name| name == "b_one"));
    assert!(
        spec.description
            .ends_with("Skills:\n- `beta` — Beta things.\n")
    );
    assert_eq!(
        spec.parameters.pointer("/properties/skill/enum"),
        Some(&json!(["beta"]))
    );
    assert!(!scope_use_skill_spec(&CATALOG, &mut spec, &|_| false));
}

#[test]
fn catalog_lookups_honour_owners() {
    assert_eq!(CATALOG.pack_for_tool("b_one").map(|p| p.id), Some("beta"));
    assert!(CATALOG.pack_for_tool("ghost").is_none());
    assert_eq!(CATALOG.all_packed_tool_names(), ["a_one", "a_two", "b_one"]);
    // An owner keeps its belt, including a thread-renamed `<owner>_<thread>` id.
    assert_eq!(
        CATALOG.packed_tool_names_for_agent("alpha_agent"),
        ["b_one"]
    );
    assert_eq!(
        CATALOG.packed_tool_names_for_agent("alpha_agent_t1"),
        ["b_one"]
    );
    assert_eq!(
        CATALOG.packed_tool_names_for_agent("alpha_agentx"),
        ["a_one", "a_two", "b_one"]
    );
    assert_eq!(CATALOG.callable_pack_ids(&|n| n == "a_two"), ["alpha"]);
}
