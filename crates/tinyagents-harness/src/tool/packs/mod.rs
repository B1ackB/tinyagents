//! On-demand tool disclosure ("tool packs").
//!
//! A tool's JSON schema is charged on every provider call of every turn,
//! whether or not the tool is used, so most of an orchestrator's fixed cost is
//! idle schema. A *pack* keeps its tools constructed and executable but
//! unadvertised; the agent sees one small tool instead. [`UseSkillTool`]
//! renders a pack's schemas into the conversation when called with a `skill`
//! alone, and executes one of them when also given a `tool`, forwarding
//! permission level, external-effect classification and timeout policy to the
//! real tool so nothing is laundered through the proxy.
//!
//! This is the mechanism only. What the host keeps:
//!
//! * the **pack table** (which tools are packed, who owns them, the playbooks),
//!   handed in as a [`PackCatalog`];
//! * the **posture** (which groups are withheld, advertised or off for a given
//!   embedder, and which packs an agent may reach);
//! * the **binding** of a [`PackRegistryHandle`] to the registries a session
//!   builds.
//!
//! Not to be confused with [`super::discover`], which is model-driven
//! `tool_search` over a BM25 catalog of deferred schemas: that one lets the model
//! *search* for a tool, this one lets a host *group* tools under named skills
//! with a playbook and an owner list.

mod catalog;
mod handle;
mod render;
mod tool;
mod types;

pub use catalog::PackCatalog;
pub use handle::PackRegistryHandle;
pub use render::{
    NoSuchPackTool, named_tool, render_pack_filtered, route_sentence, scope_use_skill_spec,
};
pub use tool::{USE_SKILL, UseSkillTool};
pub use types::ToolPack;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
