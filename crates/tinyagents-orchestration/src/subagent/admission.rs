//! Spawn admission control: bounded fan-out for sub-agent spawning.
//!
//! A model that can call a sub-agent tool can call it in a loop, or several
//! times in one turn. [`SpawnPolicy`] bounds that fan-out and
//! [`SpawnAdmission`] enforces it with *reservation* semantics: a slot is
//! claimed atomically under one lock before anything is spawned, so concurrent
//! spawns cannot race past a cap by each observing "room left" before any of
//! them has registered. The claim is a [`SpawnReservation`] guard:
//!
//! - dropped without [`SpawnReservation::commit`] (the spawn failed before the
//!   child started) it refunds every counter it touched;
//! - committed, it keeps its slot until it is dropped, which a host does when
//!   the child reaches a terminal state (the background task owns the guard
//!   for exactly the child's lifetime, panics and aborts included).
//!
//! # Scope: what the caps bound
//!
//! Run ids are minted fresh per turn by most hosts, yet background children
//! outlive the turn that spawned them, so a cap keyed on a run id would reset
//! every turn. Limits are therefore keyed on a stable **scope key** resolved
//! from the *parent's* [`RunConfig`]: its `thread_id` (the conversation) when
//! it has one, else its `run_id`. Hosts override the rule with
//! [`SpawnAdmission::with_scope_key`] (for example to map every agent of one
//! tenant, or every descendant of one root, to a single key).
//!
//! # Limits
//!
//! - `max_children_per_parent`: children *live at once* that were spawned from
//!   one scope. The slot is released at the child's terminal state.
//! - `max_total_per_root`: children *ever spawned* from one scope — a spawn
//!   budget for the conversation, so it is not released when a child finishes
//!   (only when its spawn never happened). Nested agents resolve to their own
//!   scope under the default rule; a tree-wide budget needs a resolver that
//!   maps descendants to the root's key.
//! - `allowed_targets`: sub-agent names that may be spawned. `None` allows
//!   every target; `Some(vec![])` allows none.
//!
//! Every limit defaults to `None` (unlimited), so a host that never configures
//! a policy sees no behaviour change. Recommended starting values for a host
//! that wants a guard rail: `max_children_per_parent: Some(5)` (the value
//! OpenClaw ships) and a `max_total_per_root` of a few dozen.
//!
//! State lives in the [`SpawnAdmission`] value, never in a global: clones share
//! one ledger, and a host shares one instance across every tool/driver whose
//! spawns should count against the same limits.
//!
//! Only [`SubAgentTool`](super::SubAgentTool) and
//! [`SubagentDriver`](super::SubagentDriver) enforce admission. Calling
//! `SubAgent::invoke_in_parent` / `invoke_hosted_in_parent` directly bypasses
//! it; hosts that expose those paths must reserve themselves.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tinyagents_harness::context::RunConfig;
use serde::{Deserialize, Serialize};

const LOG_PREFIX: &str = "[subagent-admission]";

/// Declarative spawn limits. See the module docs for each field's semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnPolicy {
    /// Maximum children live at once per spawn scope (see the module docs).
    #[serde(default)]
    pub max_children_per_parent: Option<usize>,
    /// Maximum children ever spawned per spawn scope (see the module docs).
    #[serde(default)]
    pub max_total_per_root: Option<usize>,
    /// Sub-agent names that may be spawned; `None` allows every target.
    #[serde(default)]
    pub allowed_targets: Option<Vec<String>>,
}

/// Why a spawn was refused.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnRejection {
    /// The target is not in [`SpawnPolicy::allowed_targets`].
    TargetNotAllowed {
        /// The refused target name; empty when the spawn named no target.
        target: String,
    },
    /// The scope already has [`SpawnPolicy::max_children_per_parent`] live children.
    MaxChildrenPerParent {
        /// Live children at refusal time.
        active: usize,
        /// The configured cap.
        max: usize,
    },
    /// The scope already spawned [`SpawnPolicy::max_total_per_root`] children.
    MaxTotalPerRoot {
        /// Children spawned in the scope at refusal time.
        spawned: usize,
        /// The configured cap.
        max: usize,
    },
}

impl std::fmt::Display for SpawnRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TargetNotAllowed { target } if target.is_empty() => write!(
                f,
                "the spawn policy restricts targets but this spawn named none"
            ),
            Self::TargetNotAllowed { target } => {
                write!(f, "spawning `{target}` is not allowed by the spawn policy")
            }
            Self::MaxChildrenPerParent { active, max } => write!(
                f,
                "this agent already has the maximum number of active children ({active}/{max})"
            ),
            Self::MaxTotalPerRoot { spawned, max } => write!(
                f,
                "this conversation reached its total child budget ({spawned}/{max})"
            ),
        }
    }
}

impl std::error::Error for SpawnRejection {}

#[derive(Default, Debug)]
struct Ledger {
    /// Live (reserved or committed, not yet dropped) children per scope.
    active_per_scope: HashMap<String, usize>,
    /// Children spawned per scope.
    spawned_per_scope: HashMap<String, usize>,
}

fn decrement(map: &mut HashMap<String, usize>, key: &str) {
    if let Some(count) = map.get_mut(key) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            map.remove(key);
        }
    }
}

/// Shared, injectable admission ledger enforcing one [`SpawnPolicy`].
///
/// Cloning shares the ledger.
#[derive(Clone)]
pub struct SpawnAdmission {
    policy: Arc<SpawnPolicy>,
    ledger: Arc<Mutex<Ledger>>,
    scope_key: ScopeKeyFn,
}

type ScopeKeyFn = Arc<dyn Fn(&RunConfig) -> String + Send + Sync>;

/// Default scope rule: the parent's conversation (`thread_id`), else its run id.
fn default_scope_key(parent: &RunConfig) -> String {
    match &parent.thread_id {
        Some(thread) => thread.as_str().to_owned(),
        None => parent.run_id.as_str().to_owned(),
    }
}

impl Default for SpawnAdmission {
    fn default() -> Self {
        Self::new(SpawnPolicy::default())
    }
}

impl std::fmt::Debug for SpawnAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnAdmission")
            .field("policy", &self.policy)
            .field("ledger", &self.ledger)
            .finish_non_exhaustive()
    }
}

impl SpawnAdmission {
    /// Creates an admission ledger enforcing `policy`.
    pub fn new(policy: SpawnPolicy) -> Self {
        Self {
            policy: Arc::new(policy),
            ledger: Arc::default(),
            scope_key: Arc::new(default_scope_key),
        }
    }

    /// Replaces the scope rule (default: the parent's `thread_id`, else its
    /// `run_id`). `resolve` receives the *parent's* config and returns the key
    /// both counters are kept under. Set it before sharing or cloning the
    /// ledger; counts already taken under the old rule are not migrated.
    pub fn with_scope_key(
        mut self,
        resolve: impl Fn(&RunConfig) -> String + Send + Sync + 'static,
    ) -> Self {
        self.scope_key = Arc::new(resolve);
        self
    }

    /// The scope key `parent` resolves to under this ledger's rule.
    pub fn scope_of(&self, parent: &RunConfig) -> String {
        (self.scope_key)(parent)
    }

    /// The policy this ledger enforces.
    pub fn policy(&self) -> &SpawnPolicy {
        &self.policy
    }

    /// Live children currently holding a slot in `scope`.
    pub fn active_children(&self, scope: &str) -> usize {
        self.ledger
            .lock()
            .active_per_scope
            .get(scope)
            .copied()
            .unwrap_or(0)
    }

    /// Children spawned so far in `scope`.
    pub fn spawned_in_scope(&self, scope: &str) -> usize {
        self.ledger
            .lock()
            .spawned_per_scope
            .get(scope)
            .copied()
            .unwrap_or(0)
    }

    /// Atomically checks every limit and, if all pass, reserves a slot for a
    /// new child spawned by the parent described by `parent` (its config; the
    /// scope is resolved from it).
    pub fn try_reserve(
        &self,
        parent: &RunConfig,
        target: &str,
    ) -> Result<SpawnReservation, SpawnRejection> {
        self.reserve(self.scope_of(parent), target, true)
    }

    /// Like [`Self::try_reserve`] for resuming an *existing* child: it takes a
    /// live slot (so the resumed run counts against the scope's live cap) but
    /// not another unit of the total budget, which the original spawn paid. The
    /// scope is resolved exactly as for a fresh spawn.
    pub fn try_reserve_continuation(
        &self,
        parent: &RunConfig,
        target: &str,
    ) -> Result<SpawnReservation, SpawnRejection> {
        self.reserve(self.scope_of(parent), target, false)
    }

    fn reserve(
        &self,
        scope: String,
        target: &str,
        counts_toward_total: bool,
    ) -> Result<SpawnReservation, SpawnRejection> {
        if let Some(allowed) = &self.policy.allowed_targets
            && !allowed.iter().any(|name| name == target)
        {
            return Err(self.reject(
                SpawnRejection::TargetNotAllowed {
                    target: target.to_owned(),
                },
                &scope,
            ));
        }
        let mut ledger = self.ledger.lock();
        if counts_toward_total && let Some(max) = self.policy.max_total_per_root {
            let spawned = ledger.spawned_per_scope.get(&scope).copied().unwrap_or(0);
            if spawned >= max {
                drop(ledger);
                return Err(self.reject(SpawnRejection::MaxTotalPerRoot { spawned, max }, &scope));
            }
        }
        if let Some(max) = self.policy.max_children_per_parent {
            let active = ledger.active_per_scope.get(&scope).copied().unwrap_or(0);
            if active >= max {
                drop(ledger);
                return Err(self.reject(
                    SpawnRejection::MaxChildrenPerParent { active, max },
                    &scope,
                ));
            }
        }
        *ledger.active_per_scope.entry(scope.clone()).or_default() += 1;
        if counts_toward_total {
            *ledger.spawned_per_scope.entry(scope.clone()).or_default() += 1;
        }
        drop(ledger);
        tracing::debug!("{LOG_PREFIX} reserved scope={scope} target={target}");
        Ok(SpawnReservation {
            admission: self.clone(),
            scope,
            counts_toward_total,
            committed: false,
        })
    }

    fn reject(&self, rejection: SpawnRejection, scope: &str) -> SpawnRejection {
        tracing::debug!("{LOG_PREFIX} rejected scope={scope} reason={rejection}");
        rejection
    }
}

/// An admitted spawn slot. Dropping it releases the slot; see the module docs.
#[must_use = "dropping a reservation immediately releases its slot"]
#[derive(Debug)]
pub struct SpawnReservation {
    admission: SpawnAdmission,
    scope: String,
    counts_toward_total: bool,
    committed: bool,
}

impl SpawnReservation {
    /// Marks the spawn as having happened. The live slot is still released on
    /// drop (the child's terminal state), but the scope's total budget stays
    /// spent.
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for SpawnReservation {
    fn drop(&mut self) {
        let mut ledger = self.admission.ledger.lock();
        decrement(&mut ledger.active_per_scope, &self.scope);
        if self.counts_toward_total && !self.committed {
            decrement(&mut ledger.spawned_per_scope, &self.scope);
        }
        drop(ledger);
        tracing::debug!(
            "{LOG_PREFIX} released scope={} committed={}",
            self.scope,
            self.committed
        );
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
