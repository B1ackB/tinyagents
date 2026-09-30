//! The late-bound view of the tool registries a pack tool dispatches into.

use std::sync::{Arc, RwLock, Weak};

use tinytools::Tool;

use super::catalog::PackCatalog;

/// An `Arc`-shared, owned view of the tool registry a pack tool lives in.
pub(super) type ToolVec = Arc<Vec<Box<dyn Tool>>>;

/// A non-owning view of the tool registry, kept to break the binding cycle.
type ToolRegistryRef = Weak<Vec<Box<dyn Tool>>>;

/// A late-bound, non-owning view of the tool registries a pack tool reads.
///
/// Late-bound because the pack tool is *inside* the registry it reads: the
/// vector cannot be built until it exists, and it cannot see it until it is
/// built. Non-owning because an `Arc` back into that same vector would be a
/// cycle that never drops.
///
/// **Two registries, not one, and that is load-bearing.** An agent's tools live
/// in two `Arc`s: the durable registry, and the `synthesized_tools` set that
/// `collect_orchestrator_tools` rebuilds whenever the Composio connection set
/// changes (they were split in #6145 so a reconcile cannot block on a reader).
/// Every `delegate_*` tool is in the second one. Binding only the first is why
/// `use_skill` could not reach a single packed delegate — `do_crypto`,
/// `run_skill`, `build_workflow` and four more were withheld from the wire and
/// then unreachable through the route that was supposed to replace them, which
/// is strictly worse than not packing them at all.
#[derive(Clone, Default)]
pub struct PackRegistryHandle {
    inner: Arc<RwLock<Slots>>,
}

/// The two registries, each independently rebindable.
///
/// They are separate slots rather than one vector because they are replaced on
/// different schedules: the durable registry is rebuilt when the agent is, the
/// synthesised one on every delegation refresh.
#[derive(Default)]
struct Slots {
    durable: Option<ToolRegistryRef>,
    synthesized: Option<ToolRegistryRef>,
}

impl PackRegistryHandle {
    /// Point this handle at the durable registry it lives in, replacing any
    /// previous binding.
    ///
    /// Rebinding has to actually take effect. This was a `OnceLock` whose
    /// second write was dropped, which silently contradicted
    /// the host's `bind_pack_registry`'s own instruction to "re-bind after any
    /// later rebuild of this `Arc`": once an agent replaced its tool vector the
    /// handle still pointed at the old allocation, the `Weak` failed to
    /// upgrade, and every `use_skill` call reported the registry as unavailable
    /// for the rest of the session. Last write wins.
    pub fn bind(&self, registry: ToolRegistryRef) {
        self.with_slots(|slots| slots.durable = Some(registry));
    }

    /// Point this handle at the synthesised delegate set.
    ///
    /// Call it again after **every** `refresh_delegation_tools`, which replaces
    /// that `Arc` wholesale — a stale `Weak` stops upgrading as soon as the last
    /// reader of the old allocation goes, and the packed delegates silently
    /// become unreachable.
    pub fn bind_synthesized(&self, registry: ToolRegistryRef) {
        self.with_slots(|slots| slots.synthesized = Some(registry));
    }

    fn with_slots(&self, edit: impl FnOnce(&mut Slots)) {
        match self.inner.write() {
            Ok(mut slots) => edit(&mut slots),
            // The lock is only ever held for a pointer read or write, so a
            // poisoned lock means a panic elsewhere. Recover rather than
            // propagate: a stale binding degrades to "skill unavailable",
            // which is the failure this rebinding exists to prevent.
            Err(poisoned) => edit(&mut poisoned.into_inner()),
        }
    }

    /// Every live registry, durable first.
    ///
    /// Order matters on a name collision: `drop_synthesized_name_collisions`
    /// gives the durable tool the name, so resolving durable-first is what
    /// makes this agree with what the harness would actually execute.
    pub(super) fn registries(&self) -> Vec<ToolVec> {
        let slots = match self.inner.read() {
            Ok(slots) => slots,
            Err(poisoned) => poisoned.into_inner(),
        };
        [slots.durable.as_ref(), slots.synthesized.as_ref()]
            .into_iter()
            .flatten()
            .filter_map(Weak::upgrade)
            .collect()
    }

    /// Resolve a packed tool by name, enforcing that it belongs to `skill`.
    ///
    /// The pack check is not decoration: without it `use_skill` would dispatch
    /// into any packed tool regardless of the skill named, and the model could
    /// reach a crypto write through a workflow skill.
    pub(super) fn resolve(
        &self,
        catalog: &PackCatalog,
        skill: &str,
        tool: &str,
    ) -> Option<(ToolVec, usize)> {
        catalog.pack(skill).filter(|p| p.owns(tool))?;
        self.find(tool)
    }

    /// Resolves `tool` in `skill`'s pack, returning the exact registry `Arc`
    /// it lives in — not a clone of the tool itself — so a caller can re-wrap
    /// it in the same `CanonicalSharedToolAdapter` seam the harness uses at
    /// registration for typed-dispatch selection.
    ///
    /// Public because a host's typed dispatch needs the same resolution
    /// [`UseSkillTool`](super::UseSkillTool) performs, so a packed delegation
    /// reached through `use_skill` can be re-dispatched through a live-parent
    /// seam instead of falling back to plain `Tool::execute_with_context`.
    pub fn resolve_registry_for(
        &self,
        catalog: &PackCatalog,
        skill: &str,
        tool: &str,
    ) -> Option<Arc<Vec<Box<dyn Tool>>>> {
        self.resolve(catalog, skill, tool).map(|(tools, _idx)| tools)
    }

    /// Locate `tool` in whichever registry holds it.
    pub(super) fn find(&self, tool: &str) -> Option<(ToolVec, usize)> {
        for tools in self.registries() {
            if let Some(idx) = tools.iter().position(|t| t.name() == tool) {
                return Some((tools, idx));
            }
        }
        None
    }
}

