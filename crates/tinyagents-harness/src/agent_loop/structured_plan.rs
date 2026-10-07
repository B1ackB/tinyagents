//! Structured-output planning for one turn.
//!
//! Split out of `run_loop.rs`. Resolves the policy's response format against
//! the model the turn resolved to and rewrites the request accordingly; the
//! chosen strategy drives extraction of the final response.

use super::*;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Resolves the structured-output plan against the resolved model.
    ///
    /// `Auto` consults the model profile to choose provider-native schema mode
    /// versus a tool-call fallback; an explicit `JsonSchema` always uses
    /// provider-native mode. `registered_tool_count` is the run's registered
    /// tool surface (the schema tool is forced only when it is the sole tool).
    ///
    /// Returns the plan together with the tool schemas the plan minted (the
    /// `ToolCall` / `ToolCallUnion` fallback tools pushed onto `request.tools`).
    /// A host that renders its own static tool catalogue composed it before
    /// this planning ran, so it cannot have advertised them;
    /// `RunDialect::apply_to_request` appends their catalogue entries even in
    /// the host-rendered case.
    pub(super) fn plan_structured_output(
        &self,
        ctx: &RunContext<Ctx>,
        request: &mut ModelRequest,
        profile: Option<&tinyinference_llm::model::ModelProfile>,
        registered_tool_count: usize,
    ) -> (Option<(StructuredStrategy, String, Value)>, Vec<ToolSchema>) {
        // Marks where any structured-output fallback tool gets pushed below, so
        // it can be told apart afterward from what was already on
        // `request.tools`.
        let tools_before_structured_plan = request.tools.len();
        let structured_plan: Option<(StructuredStrategy, String, Value)> = match request
            .response_format
            .clone()
        {
            Some(ResponseFormat::Auto { name, schema })
                if matches!(
                    self.policy.structured_strategy_override,
                    Some(crate::runtime::StructuredStrategyOverride::Prompted { .. })
                ) =>
            {
                let template = match &self.policy.structured_strategy_override {
                    Some(crate::runtime::StructuredStrategyOverride::Prompted { template }) => {
                        template.clone()
                    }
                    _ => unreachable!("guarded by the match arm above"),
                };
                let schema = crate::tool::apply_profile_schema_transform(&schema, profile);
                request.response_format = Some(ResponseFormat::Text);
                let instructions = template
                    .clone()
                    .unwrap_or_else(|| crate::structured::default_prompted_template().to_string());
                let schema_text = serde_json::to_string_pretty(&schema).unwrap_or_default();
                crate::cache::prepend_system_message(
                    request,
                    format!("{instructions}\n\nJSON Schema for `{name}`:\n{schema_text}"),
                );
                Some((StructuredStrategy::Prompted { template }, name, schema))
            }
            Some(ResponseFormat::Auto { name, schema })
                if matches!(
                    self.policy.structured_strategy_override,
                    Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion { .. })
                ) =>
            {
                let variants = match &self.policy.structured_strategy_override {
                    Some(crate::runtime::StructuredStrategyOverride::ToolCallUnion {
                        variants,
                    }) => variants.clone(),
                    _ => unreachable!("guarded by the match arm above"),
                };
                request.response_format = Some(ResponseFormat::Text);
                for (variant_name, variant_schema) in &variants {
                    let variant_schema =
                        crate::tool::apply_profile_schema_transform(variant_schema, profile);
                    let schema_tool = ToolSchema {
                        name: variant_name.clone(),
                        description: format!("Return the result as `{variant_name}`."),
                        parameters: variant_schema,
                        format: tinyinference_llm::tool::ToolFormat::Json,
                    };
                    request.tools.push(match &self.policy.tool_schemas {
                        Some(preparation) => {
                            crate::tool::prepare_tool_schema(&schema_tool, preparation)
                        }
                        None => schema_tool,
                    });
                }
                let _ = schema;
                Some((StructuredStrategy::ToolCallUnion, name, Value::Null))
            }
            Some(ResponseFormat::Auto { name, schema }) => {
                let schema = crate::tool::apply_profile_schema_transform(&schema, profile);
                let strategy = StructuredStrategy::for_profile(profile);
                match strategy {
                    StructuredStrategy::ProviderSchema => {
                        request.response_format =
                            Some(ResponseFormat::json_schema(name.clone(), schema.clone()));
                    }
                    StructuredStrategy::ToolCall => {
                        request.response_format = Some(ResponseFormat::Text);
                        let fallback_schema = ToolSchema {
                            name: name.clone(),
                            description: format!("Return the result as `{name}`."),
                            parameters: schema.clone(),
                            format: tinyinference_llm::tool::ToolFormat::Json,
                        };
                        // This schema is generated here, after the
                        // direct and bridge schemas above were
                        // prepared for the target provider, so it
                        // needs the same projection or it reaches the
                        // wire raw (see the `tool_schemas` and bridge
                        // preparation above).
                        request.tools.push(match &self.policy.tool_schemas {
                            Some(preparation) => {
                                crate::tool::prepare_tool_schema(&fallback_schema, preparation)
                            }
                            None => fallback_schema,
                        });
                        // Force the schema tool **only** when it is the
                        // sole tool available. Forcing it inside a
                        // tool-using loop makes the model emit the
                        // structured call on turn 1, which terminates
                        // the loop before any registered tool can ever
                        // run — the agent silently loses its tools, and
                        // the symptom points nowhere near this code.
                        // LangChain likewise binds a schema tool with a
                        // forced `tool_choice` only in its terminal
                        // wrapper, never in the tool-calling loop.
                        // A final-call middleware can withdraw the
                        // ordinary request tools before this planner
                        // runs. In that case the synthetic schema tool
                        // is the only remaining callable tool, even
                        // though the run-level registry still contains
                        // the withdrawn tools.
                        if registered_tool_count == 0
                            || (tools_before_structured_plan == 0
                                && request.tool_choice == ToolChoice::None)
                        {
                            request.tool_choice = ToolChoice::Tool(name.clone());
                        } else {
                            tracing::debug!(
                                target: "tinyagents::agent_loop",
                                run_id = %ctx.run_id(),
                                schema_name = %name,
                                registered_tools = registered_tool_count,
                                "[agent_loop] structured tool offered but not forced; \
                                 registered tools stay callable"
                            );
                        }
                    }
                    // A profile whose `default_structured_mode` is
                    // `Prompted` reaches this arm too (not only
                    // through the dedicated
                    // `structured_strategy_override` arm above): the
                    // schema goes into the system segment instead of
                    // a provider API field, mirroring the override
                    // arm's construction.
                    StructuredStrategy::Prompted { ref template } => {
                        request.response_format = Some(ResponseFormat::Text);
                        let instructions = template.clone().unwrap_or_else(|| {
                            crate::structured::default_prompted_template().to_string()
                        });
                        let schema_text = serde_json::to_string_pretty(&schema).unwrap_or_default();
                        crate::cache::prepend_system_message(
                            request,
                            format!("{instructions}\n\nJSON Schema for `{name}`:\n{schema_text}"),
                        );
                    }
                    // `for_profile` never returns `ToolCallUnion`;
                    // that strategy is reached exclusively through
                    // the dedicated `structured_strategy_override`
                    // arm above.
                    StructuredStrategy::ToolCallUnion => {
                        unreachable!("StructuredStrategy::for_profile never returns ToolCallUnion")
                    }
                }
                Some((strategy, name, schema))
            }
            Some(ResponseFormat::JsonSchema { name, schema }) => {
                let schema = crate::tool::apply_profile_schema_transform(&schema, profile);
                request.response_format = Some(ResponseFormat::JsonSchema {
                    name: name.clone(),
                    schema: schema.clone(),
                });
                Some((StructuredStrategy::ProviderSchema, name, schema))
            }
            _ => None,
        };
        let synthesized_tools: Vec<ToolSchema> =
            request.tools[tools_before_structured_plan..].to_vec();
        (structured_plan, synthesized_tools)
    }
}
