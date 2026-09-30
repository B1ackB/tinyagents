//! Rendering for the `use_skill` disclosure half: pack listings, not-found
//! results, routing sentences and per-session spec scoping.

use serde_json::Value;
use tinytools::ToolSpec;

use super::catalog::PackCatalog;
use super::handle::PackRegistryHandle;

/// Render a pack's listing, showing only the tools `is_callable` admits.
///
/// **The filter is the whole point.** `load_skill` used to render every tool in
/// the pack this build compiled, and `use_skill` then refused any of them the
/// session's allowlist denies (`tinyagents::middleware::channel_permission_block`).
/// A non-owner was handed a menu it could not order from: the orchestrator
/// loaded `workflows`, read `propose_workflow` off the listing, called it, and
/// was told it "is not allowed in the current session". The denial named no
/// alternative, so the model retried — one live chat turn died on the
/// repeated-failure breaker after six identical denials.
///
/// `registry.rs` used to claim non-owners "reach them through `use_skill`".
/// That was never true: the gate (`d5a09ea81`, 2026-08-21) predates the comment
/// asserting it (`a8f0a002b`, 2026-08-23). The listing is the side that was
/// wrong, so the listing is the side that changed.
///
/// `route` is the sentence to append when the session can call nothing in the
/// pack — see [`route_sentence`]. Empty means "say nothing extra".
pub fn render_pack_filtered(
    catalog: &PackCatalog,
    skill: &str,
    handle: &PackRegistryHandle,
    is_callable: &dyn Fn(&str) -> bool,
    route: &str,
) -> Result<String, String> {
    let Some(pack) = catalog.pack(skill) else {
        // Scoped too: offering a hallucinating model a pack it cannot use is the
        // same wrong turn the advertised index used to take, one error later.
        return Err(format!(
            "{} Unknown skill `{skill}`. Available:\n{}",
            catalog.not_found_marker(),
            catalog.pack_index_markdown_filtered(is_callable)
        ));
    };
    if handle.registries().is_empty() {
        return Err(
            "The skill registry is not available in this session; the tools in this skill \
             cannot be loaded."
                .to_string(),
        );
    }

    let mut out = format!("# Skill `{}`\n\n{}\n\n", pack.id, pack.summary);
    if !pack.guide.trim().is_empty() {
        out.push_str(pack.guide.trim());
        out.push_str("\n\n");
    }
    out.push_str(&format!(
        "Call these with `use_skill {{ \"skill\": \"{}\", \"tool\": \"<name>\", \"args\": {{ … }} }}`. \
         `args` is the tool's own argument object, exactly as documented below.\n\n",
        pack.id
    ));

    let mut found = 0usize;
    for name in pack.tools {
        // A pack may name a tool this build compiled out (feature gate) or that
        // this agent never had. Rendering the ones that exist beats failing the
        // whole load.
        let Some((tools, idx)) = handle.find(name) else {
            continue;
        };
        // Listing a tool the gate will refuse is worse than omitting it: a
        // model cannot tell a policy denial from a transient failure, so it
        // retries the same call instead of routing around it.
        if !is_callable(name) {
            continue;
        }
        let tool = &tools[idx];
        found += 1;
        out.push_str(&format!(
            "## `{}`\n\n{}\n\n",
            tool.name(),
            tool.description()
        ));
        // Minified, matching what the provider receives for a natively
        // advertised tool. Pretty-printing costs roughly a third more tokens
        // for indentation and newlines the model gains nothing from, and this
        // text is charged to the context window exactly like a native schema.
        out.push_str("```json\n");
        out.push_str(
            &serde_json::to_string(&tool.parameters_schema()).unwrap_or_else(|_| "{}".to_string()),
        );
        out.push_str("\n```\n\n");
    }

    if found == 0 {
        let mut message = format!(
            "Skill `{}` has no tools available in this session.",
            pack.id
        );
        if !route.is_empty() {
            message.push(' ');
            message.push_str(route);
        }
        return Err(message);
    }
    Ok(out)
}

/// A `use_skill` call naming a tool its skill does not contain (#6302).
///
/// Typed so the gate and its tests read one source: an invented name
/// (`install_skill` in `skills`) must read as "no such tool" plus what the
/// session can call instead, never as a permission denial. The not-found
/// marker makes the failure classify as `NotFound`.
pub struct NoSuchPackTool<'a> {
    /// The not-found marker of the host's [`PackCatalog`].
    pub not_found_marker: &'a str,
    pub skill: &'a str,
    pub tool: &'a str,
    /// Tools in the skill this session can call, in pack order.
    pub callable: Vec<&'a str>,
    /// The hand-off sentence to use when `callable` is empty.
    pub route: String,
}

impl NoSuchPackTool<'_> {
    pub fn render(&self) -> String {
        let mut out = format!(
            "{} There is no tool `{}` in skill `{}`.",
            self.not_found_marker, self.tool, self.skill
        );
        if !self.callable.is_empty() {
            let names = self
                .callable
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(" The tools in it you can call: {names}."));
        } else if !self.route.is_empty() {
            out.push(' ');
            out.push_str(&self.route);
        } else {
            out.push_str(" Nothing in it is available in this session.");
        }
        out
    }
}

/// The "go here instead" sentence shared by the `use_skill` listing and the
/// `use_skill` denial, so a model never sees two different stories.
///
/// `callable_delegates` are delegation tool names the caller has already
/// confirmed this session can invoke — naming the *tool* rather than the agent
/// is the difference between guidance and an instruction, and a model left to
/// guess the call retries. When none can be reached the owning agents are named
/// instead: strictly worse, but still better than a bare denial.
pub fn route_sentence(callable_delegates: &[String], owners: &[&str]) -> String {
    if !callable_delegates.is_empty() {
        let names = callable_delegates
            .iter()
            .map(|t| format!("`{t}`"))
            .collect::<Vec<_>>()
            .join(" or ");
        return format!(
            "Call {names} instead — that agent owns these tools and runs them directly. \
             Do not retry this skill."
        );
    }
    if owners.is_empty() {
        return String::new();
    }
    format!(
        "These tools belong to {}; hand the task to one of them rather than calling directly.",
        owners
            .iter()
            .map(|o| format!("`{o}`"))
            .collect::<Vec<_>>()
            .join(" or ")
    )
}

pub(super) fn render_pack(
    catalog: &PackCatalog,
    skill: &str,
    handle: &PackRegistryHandle,
) -> Result<String, String> {
    render_pack_filtered(catalog, skill, handle, &|_| true, "")
}

/// Rewrite `use_skill`'s advertised spec to match what this session can do.
///
/// The description is built once in [`UseSkillTool::new`], before any session
/// exists, so every agent was told all ten packs were loadable — including ones
/// it can call nothing in. Post-#(routing fix) that costs one wasted round trip
/// instead of a dead turn; it should cost zero.
///
/// Both halves are rewritten, and the schema is the stronger one: narrowing the
/// `skill` enum makes an unusable pack *unrepresentable* rather than merely
/// discouraged in prose, and a shorter enum is fewer tokens, not more.
///
/// Returns `false` when this session can call nothing in any pack — the caller
/// should then drop `use_skill` from the wire entirely, because an empty index
/// and an empty enum are not a tool.
pub fn scope_use_skill_spec(
    catalog: &PackCatalog,
    spec: &mut ToolSpec,
    is_callable: &dyn Fn(&str) -> bool,
) -> bool {
    let ids = catalog.callable_pack_ids(is_callable);
    if ids.is_empty() {
        return false;
    }
    if let Some(index) = spec.description.find("\n\nSkills:\n") {
        spec.description.truncate(index);
        spec.description.push_str("\n\nSkills:\n");
        spec.description
            .push_str(&catalog.pack_index_markdown_filtered(is_callable));
    }
    if let Some(enum_slot) = spec
        .parameters
        .pointer_mut("/properties/skill/enum")
        .filter(|v| v.is_array())
    {
        *enum_slot = Value::Array(
            ids.iter()
                .map(|id| Value::String((*id).to_string()))
                .collect(),
        );
    }
    true
}

/// The tool named in `args`, if the caller named one at all.
///
/// An absent (or empty) `tool` is not a malformed call: it is the disclosure
/// half of this tool, and the distinction decides both which branch
/// [`UseSkillTool::execute_with_context`] takes and what permission level the
/// call is gated at. Public because the policy middleware has to draw the same
/// line — it intercepts the disclosure half to scope the listing to the session
/// and lets the execution half through to its gate — and two spellings of "did
/// the caller name a tool" would be two chances to disagree.
pub fn named_tool(args: &Value) -> Option<&str> {
    args.get("tool")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
}
