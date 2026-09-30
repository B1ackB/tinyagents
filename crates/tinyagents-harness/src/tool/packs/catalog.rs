//! The pack table a host hands to the generic `use_skill` machinery.
//!
//! The table itself is host data — which tools are packed, who owns them, what
//! the playbook says — and stays with the host. [`PackCatalog`] is the lookup
//! surface over it, so the disclosure tool, the listing renderer and the spec
//! scoper never name a global.

use super::types::ToolPack;

/// A host's compiled-in pack table plus the one piece of host vocabulary the
/// rendered text needs: the marker that makes a "no such skill / tool" result
/// classify as not-found in the host's status taxonomy.
#[derive(Debug, Clone, Copy)]
pub struct PackCatalog {
    packs: &'static [ToolPack],
    not_found_marker: &'static str,
}

impl PackCatalog {
    /// A catalog over `packs`. `not_found_marker` prefixes every "unknown skill"
    /// or "no such tool" result so the host's error classifier can recognise it.
    pub const fn new(packs: &'static [ToolPack], not_found_marker: &'static str) -> Self {
        Self {
            packs,
            not_found_marker,
        }
    }

    /// Every pack, in table order.
    pub const fn packs(&self) -> &'static [ToolPack] {
        self.packs
    }

    /// The not-found marker this catalog's host supplied.
    pub const fn not_found_marker(&self) -> &'static str {
        self.not_found_marker
    }

    /// The pack with this id.
    pub fn pack(&self, id: &str) -> Option<&'static ToolPack> {
        self.packs.iter().find(|p| p.id == id)
    }

    /// The pack owning `tool`, if any.
    pub fn pack_for_tool(&self, tool: &str) -> Option<&'static ToolPack> {
        self.packs.iter().find(|p| p.owns(tool))
    }

    /// Every packed tool name across all packs.
    pub fn all_packed_tool_names(&self) -> Vec<&'static str> {
        self.packs
            .iter()
            .flat_map(|p| p.tools.iter().copied())
            .collect()
    }

    /// Every packed tool name that applies to `agent_id`.
    ///
    /// A pack is skipped entirely for agents listed as its owners — see
    /// [`ToolPack::owners`].
    pub fn packed_tool_names_for_agent(&self, agent_id: &str) -> Vec<&'static str> {
        self.packs
            .iter()
            .filter(|p| !p.is_owner(agent_id))
            .flat_map(|p| p.tools.iter().copied())
            .collect()
    }

    /// The always-on index: one line per pack, rendered into `use_skill`'s own
    /// description so the model can pick a pack without a round trip.
    pub fn pack_index_markdown(&self) -> String {
        self.pack_index_markdown_filtered(&|_| true)
    }

    /// The pack index, limited to packs this session can call at least one tool in.
    ///
    /// A pack with nothing callable is not an answer to "which skills can I load",
    /// and advertising it costs a round trip: the model loads it, learns it cannot
    /// use it, and comes back. The capability does not disappear — a pack's owners
    /// reach the model through their own delegation tools, whose descriptions are
    /// already on the wire. Keeping the pack listed here would duplicate that
    /// routing on every single turn.
    pub fn pack_index_markdown_filtered(&self, is_callable: &dyn Fn(&str) -> bool) -> String {
        let mut out = String::new();
        for p in self.packs {
            if !p.tools.iter().any(|t| is_callable(t)) {
                continue;
            }
            out.push_str(&format!("- `{}` — {}\n", p.id, p.summary));
        }
        out
    }

    /// Pack ids with at least one tool this session can call — the `skill` enum
    /// `use_skill` should actually offer.
    pub fn callable_pack_ids(&self, is_callable: &dyn Fn(&str) -> bool) -> Vec<&'static str> {
        self.packs
            .iter()
            .filter(|p| p.tools.iter().any(|t| is_callable(t)))
            .map(|p| p.id)
            .collect()
    }
}
