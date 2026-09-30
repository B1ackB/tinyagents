//! The always-on tool that stands in for every packed tool.

use async_trait::async_trait;
use serde_json::{Value, json};
use tinytools::{PermissionLevel, Tool, ToolCallOptions, ToolResult, ToolRunContext};

use super::catalog::PackCatalog;
use super::handle::{PackRegistryHandle, ToolVec};
use super::render::{named_tool, render_pack};

/// Name of the disclosure-and-dispatch tool.
pub const USE_SKILL: &str = "use_skill";

/// The one always-on tool that stands in for every packed tool.
///
/// It is both halves of the pack seam. Called with a `skill` alone it renders
/// that pack's tool schemas into the conversation; called with a `skill` and a
/// `tool` it executes that tool. These were two tools — `load_skill` and
/// `use_skill` — until the pack index that each carried in its own description
/// made the pair spend 3.3 kB of every single turn saying one list twice, and
/// made the first call of any packed tool a mandatory two-call round trip.
/// A model that learned the retired name gets the harness's "unknown tool"
/// recovery result (#4249) and retries against the schema it can actually see,
/// so no alias is carried for it.
///
/// **Permission forwarding is load-bearing.** The harness gates a call on the
/// tool's `permission_level_with_args`, so a proxy reporting its own level would
/// launder every packed tool's risk down to this one's — a crypto write would
/// be admitted on a channel that refuses crypto writes. Both accessors resolve
/// the inner tool and defer to it; the arg-less one has nothing to resolve
/// from, so it reports the highest level any packed tool needs rather than
/// guessing low. A call that names no `tool` reads a schema and nothing else,
/// so that one branch is genuinely `ReadOnly`.
pub struct UseSkillTool {
    handle: PackRegistryHandle,
    catalog: PackCatalog,
    description: String,
}

impl UseSkillTool {
    pub fn new(handle: PackRegistryHandle, catalog: PackCatalog) -> Self {
        let description = format!(
            "Reach a skill's tools. Their names, descriptions and argument schemas are NOT in \
             your context until you ask for them: call this with `skill` alone to see them, then \
             again with `skill` + `tool` + `args` to run one.\n\nSkills:\n{}",
            catalog.pack_index_markdown()
        );
        Self {
            handle,
            catalog,
            description,
        }
    }

    fn skill_enum(&self) -> Vec<&'static str> {
        self.catalog.packs().iter().map(|p| p.id).collect()
    }

    fn resolve(&self, args: &Value) -> Option<(ToolVec, usize)> {
        let skill = args.get("skill").and_then(Value::as_str)?;
        let tool = named_tool(args)?;
        self.handle.resolve(&self.catalog, skill, tool)
    }
}

#[async_trait]
impl Tool for UseSkillTool {
    fn name(&self) -> &str {
        USE_SKILL
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "skill": { "type": "string", "enum": self.skill_enum(), "description": "Skill to read or run a tool from." },
                "tool": { "type": "string", "description": "Tool to run. Omit to list the skill's tools and their arguments instead." },
                "args": {
                    "type": "object",
                    "description": "The tool's own arguments, as documented in the listing.",
                    "additionalProperties": true
                }
            },
            "required": ["skill"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.execute_with_context(args, ToolCallOptions::default(), None)
            .await
    }

    async fn execute_with_options(
        &self,
        args: Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        self.execute_with_context(args, options, None).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        options: ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let Some(skill) = args.get("skill").and_then(Value::as_str) else {
            return Ok(ToolResult::error(format!(
                "`skill` is required.\n\nSkills:\n{}",
                self.catalog.pack_index_markdown()
            )));
        };

        // Disclosure half: no tool named, so render the pack's schemas.
        let Some(name) = named_tool(&args) else {
            return Ok(match render_pack(&self.catalog, skill, &self.handle) {
                Ok(text) => ToolResult::success(text),
                Err(message) => ToolResult::error(message),
            });
        };

        let Some((tools, idx)) = self.handle.resolve(&self.catalog, skill, name) else {
            return Ok(ToolResult::error(format!(
                "{} No tool `{name}` in skill `{skill}`. Call `use_skill {{ \"skill\": \"{skill}\" }}` \
                 to see what it contains.\n\nSkills:\n{}",
                self.catalog.not_found_marker(),
                self.catalog.pack_index_markdown()
            )));
        };
        let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
        tracing::debug!(tool = tools[idx].name(), "[toolpacks] use_skill dispatch");
        tools[idx]
            .execute_with_context(inner_args, options, context)
            .await
    }

    fn supports_markdown(&self) -> bool {
        // The inner result is forwarded verbatim, markdown rendering included,
        // so advertise the capability rather than suppressing a real saving.
        true
    }

    fn external_effect_with_args(&self, args: &Value) -> bool {
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].external_effect_with_args(&inner_args)
            }
            None => false,
        }
    }

    fn timeout_policy(&self, args: &Value) -> tinytools::ToolTimeout {
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].timeout_policy(&inner_args)
            }
            None => tinytools::ToolTimeout::Inherit,
        }
    }

    fn permission_level(&self) -> PermissionLevel {
        let packed = self.catalog.all_packed_tool_names();
        self.handle
            .registries()
            .iter()
            .flat_map(|tools| tools.iter())
            .filter(|t| packed.contains(&t.name()))
            .map(|t| t.permission_level())
            .max()
            // Unbound or empty: report the ceiling, never a permissive default.
            .unwrap_or(PermissionLevel::Dangerous)
    }

    fn permission_level_with_args(&self, args: &Value) -> PermissionLevel {
        // Naming no tool renders a schema and does nothing else. Reporting the
        // packed ceiling here would put an approval prompt in front of reading
        // a tool list, which is the round trip this tool exists to remove.
        if named_tool(args).is_none() {
            return PermissionLevel::ReadOnly;
        }
        match self.resolve(args) {
            Some((tools, idx)) => {
                let inner = args.get("args").cloned().unwrap_or_else(|| json!({}));
                tools[idx].permission_level_with_args(&inner)
            }
            // Unresolvable: the call will fail anyway, but report the ceiling so
            // a malformed call can never be admitted on a channel that would
            // have refused the real tool.
            None => self.permission_level(),
        }
    }

    /// The registry handle rides on the vocabulary's erased host extension:
    /// `PackRegistryHandle` is a harness concept, and `tinytools` has no
    /// business naming it. A host reads it back by downcasting to
    /// [`PackRegistryHandle`].
    fn host_extension(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        Some(&self.handle)
    }
}
