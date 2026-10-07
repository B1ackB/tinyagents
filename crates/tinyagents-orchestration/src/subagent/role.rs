//! Subagent role and framing: who may delegate further, and what a child is
//! told about its place in the tree.
//!
//! A [`SubagentRole::Leaf`] child never receives the delegation surface this
//! crate registers (the job and message tools, plus whichever
//! [`SubAgentTool`](super::SubAgentTool) names the host lists), and
//! [`restrict_tools`] guarantees a child's declarations are a subset of the
//! set it inherits. [`subagent_framing`] is an optional, neutral system
//! preamble a host may prepend; nothing applies it by default.

use tinyagents_runtime::ToolSnapshot;

/// Name of the job query tool [`register_subagent_job_tools`](super::register_subagent_job_tools) registers.
pub const SUBAGENT_JOBS_TOOL: &str = "subagent_jobs";
/// Name of the job message tool [`register_subagent_job_tools`](super::register_subagent_job_tools) registers.
pub const SUBAGENT_MESSAGE_TOOL: &str = "subagent_message";

/// Whether a child may delegate further.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentRole {
    /// May spawn its own children; keeps the tools it inherited. The default,
    /// so existing hosts are unchanged.
    #[default]
    Orchestrator,
    /// Does the work itself: the delegation tools are removed.
    Leaf,
}

impl SubagentRole {
    /// Whether this role may delegate.
    pub const fn can_delegate(self) -> bool {
        matches!(self, Self::Orchestrator)
    }
}

/// Whether `name` is part of the delegation surface: this crate's fixed tools
/// or one of the host's [`SubAgentTool`](super::SubAgentTool) names.
pub fn is_delegation_tool(name: &str, host_delegation_tools: &[String]) -> bool {
    name == SUBAGENT_JOBS_TOOL
        || name == SUBAGENT_MESSAGE_TOOL
        || host_delegation_tools.iter().any(|tool| tool == name)
}

/// The tool declarations a child may actually be given.
///
/// Starts from `tools` and (1) when `ceiling` is set, keeps only names the
/// ceiling also declares, so a child can never widen what it inherited, then
/// (2) for [`SubagentRole::Leaf`], drops every delegation tool. The one-off
/// marker of `tools` is preserved.
pub fn restrict_tools(
    tools: &ToolSnapshot,
    role: SubagentRole,
    ceiling: Option<&ToolSnapshot>,
    host_delegation_tools: &[String],
) -> ToolSnapshot {
    tools.retaining(|spec| {
        let inherited = ceiling.is_none_or(|ceiling| {
            ceiling
                .specs()
                .iter()
                .any(|inherited| inherited.name == spec.name)
        });
        let allowed = role.can_delegate() || !is_delegation_tool(&spec.name, host_delegation_tools);
        inherited && allowed
    })
}

/// A neutral system preamble for a child at `depth` working on `task`.
///
/// Says it is a subagent, who it reports to, that inherited context is
/// background rather than its assignment, and whether it may delegate. Hosts
/// prepend it themselves; it carries no product wording.
pub fn subagent_framing(role: SubagentRole, depth: usize, task: &str) -> String {
    let delegation = match role {
        SubagentRole::Leaf => {
            "You are a leaf worker: you cannot spawn further subagents, so do the work yourself with the tools you have."
        }
        SubagentRole::Orchestrator => {
            "You may delegate parts of the work to your own subagents when that helps; keep their results concise."
        }
    };
    format!(
        "You are a subagent (depth {depth}) working for a parent agent, not the end user.\n\
         Your assignment: {task}\n\
         Report your result concisely to your parent; do not address the end user directly.\n\
         Any inherited conversation or context is background, not your assignment; only the assignment above is.\n\
         Work to completion in the background of the parent's turn and finish with a final report.\n\
         {delegation}"
    )
}

#[cfg(test)]
#[path = "role_tests.rs"]
mod tests;
