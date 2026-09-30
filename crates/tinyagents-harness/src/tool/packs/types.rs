//! Tool-pack types: the unit of on-demand tool disclosure.

/// A named bundle of tools that is **not** advertised to the model by default.
///
/// The pack's tools stay fully constructed and executable; what changes is that
/// their JSON schemas never reach the provider until the agent asks for them.
/// That trade is the whole point: an orchestrator carrying ~77 tool schemas
/// spends far more of its fixed per-turn budget on schemas than on its own
/// instructions, and most of those tools go untouched in most conversations.
#[derive(Debug)]
pub struct ToolPack {
    /// Stable id the agent names in `use_skill`.
    pub id: &'static str,
    /// One line, rendered in the always-on pack index. This is the only text
    /// about the pack the model sees before loading it, so it has to carry
    /// enough intent for the model to know when to reach for it.
    pub summary: &'static str,
    /// Tool names this pack owns. A name listed here is removed from the
    /// agent's advertised surface and reachable only through `use_skill`.
    pub tools: &'static [&'static str],
    /// Agent ids for which this pack is **not** applied.
    ///
    /// Withholding is a bet that the tools are idle in most turns. That bet is
    /// wrong for the specialist a family was delegated to: `workflow_builder`
    /// exists precisely to run the flow authoring tools, so packing them would
    /// put a `use_skill` round trip in front of the first call of every one of
    /// its turns and buy nothing — its whole belt is the pack.
    ///
    /// The earlier packs did not need this because they held only synthesised
    /// `delegate_*` tools, which exist on the orchestrator alone. Packs over
    /// raw tools do, and an owner list is the narrowest way to say so.
    pub owners: &'static [&'static str],
    /// The skill's playbook, printed when `use_skill` loads the pack, between
    /// the summary and the tool schemas. Empty for a pack that is only a
    /// schema bundle.
    ///
    /// This is what replaced most single-belt specialists: a sub-agent whose
    /// whole value was a prompt over a handful of tools is a ~500-token guide
    /// here plus `Deferred` tools the orchestrator reaches itself, instead of
    /// a separate context, model call and hand-off envelope. Lookup detail
    /// belongs in the guide; a rule that must bind before the model thinks to
    /// load the skill (confirm before moving money) stays in the orchestrator
    /// prompt. Kept under ~550 tokens by `toolpacks_tests.rs`.
    pub guide: &'static str,
}

impl ToolPack {
    pub fn owns(&self, tool: &str) -> bool {
        self.tools.contains(&tool)
    }

    /// Whether `agent_id` owns this pack, including a web-chat thread name
    /// formed as `<owner>_<thread>` after the session is built.
    pub fn is_owner(&self, agent_id: &str) -> bool {
        self.owners.iter().any(|owner| {
            agent_id == *owner
                || agent_id
                    .strip_prefix(*owner)
                    .is_some_and(|suffix| suffix.starts_with('_'))
        })
    }
}
