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
//! # Limits
//!
//! - `max_children_per_parent`: children *live at once* under one parent run.
//!   The slot is released at the child's terminal state.
//! - `max_total_per_root`: children *ever spawned* under one root run — a
//!   fan-out budget for the whole tree, so it is not released when a child
//!   finishes (only when its spawn never happened).
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

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

const LOG_PREFIX: &str = "[subagent-admission]";

/// Declarative spawn limits. See the module docs for each field's semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnPolicy {
    /// Maximum children live at once under one parent run.
    #[serde(default)]
    pub max_children_per_parent: Option<usize>,
    /// Maximum children ever spawned under one root run.
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
        /// The refused target name.
        target: String,
    },
    /// The parent already has [`SpawnPolicy::max_children_per_parent`] live children.
    MaxChildrenPerParent {
        /// Live children at refusal time.
        active: usize,
        /// The configured cap.
        max: usize,
    },
    /// The root already spawned [`SpawnPolicy::max_total_per_root`] children.
    MaxTotalPerRoot {
        /// Children spawned under the root at refusal time.
        spawned: usize,
        /// The configured cap.
        max: usize,
    },
}

impl std::fmt::Display for SpawnRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TargetNotAllowed { target } => {
                write!(f, "spawning `{target}` is not allowed by the spawn policy")
            }
            Self::MaxChildrenPerParent { active, max } => write!(
                f,
                "this agent already has the maximum number of active children ({active}/{max})"
            ),
            Self::MaxTotalPerRoot { spawned, max } => write!(
                f,
                "this run tree reached its total child budget ({spawned}/{max})"
            ),
        }
    }
}

impl std::error::Error for SpawnRejection {}

#[derive(Default, Debug)]
struct Ledger {
    /// Live (reserved or committed, not yet dropped) children per parent run.
    active_per_parent: HashMap<String, usize>,
    /// Children spawned per root run.
    spawned_per_root: HashMap<String, usize>,
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
#[derive(Clone, Default, Debug)]
pub struct SpawnAdmission {
    policy: Arc<SpawnPolicy>,
    ledger: Arc<Mutex<Ledger>>,
}

impl SpawnAdmission {
    /// Creates an admission ledger enforcing `policy`.
    pub fn new(policy: SpawnPolicy) -> Self {
        Self {
            policy: Arc::new(policy),
            ledger: Arc::default(),
        }
    }

    /// The policy this ledger enforces.
    pub fn policy(&self) -> &SpawnPolicy {
        &self.policy
    }

    /// Live children currently holding a slot under `parent_run_id`.
    pub fn active_children(&self, parent_run_id: &str) -> usize {
        self.ledger
            .lock()
            .active_per_parent
            .get(parent_run_id)
            .copied()
            .unwrap_or(0)
    }

    /// Children spawned so far under `root_run_id`.
    pub fn spawned_in_root(&self, root_run_id: &str) -> usize {
        self.ledger
            .lock()
            .spawned_per_root
            .get(root_run_id)
            .copied()
            .unwrap_or(0)
    }

    /// Atomically checks every limit and, if all pass, reserves a slot for a
    /// new child of `parent_run_id` in the tree rooted at `root_run_id`.
    pub fn try_reserve(
        &self,
        root_run_id: &str,
        parent_run_id: &str,
        target: &str,
    ) -> Result<SpawnReservation, SpawnRejection> {
        self.reserve(root_run_id, parent_run_id, target, true)
    }

    /// Like [`Self::try_reserve`] for resuming an *existing* child: it takes a
    /// live slot (so the resumed run counts against the parent cap) but not
    /// another unit of the root's total budget, which the original spawn paid.
    pub fn try_reserve_continuation(
        &self,
        root_run_id: &str,
        parent_run_id: &str,
        target: &str,
    ) -> Result<SpawnReservation, SpawnRejection> {
        self.reserve(root_run_id, parent_run_id, target, false)
    }

    fn reserve(
        &self,
        root_run_id: &str,
        parent_run_id: &str,
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
                parent_run_id,
            ));
        }
        let mut ledger = self.ledger.lock();
        if counts_toward_total && let Some(max) = self.policy.max_total_per_root {
            let spawned = ledger
                .spawned_per_root
                .get(root_run_id)
                .copied()
                .unwrap_or(0);
            if spawned >= max {
                drop(ledger);
                return Err(self.reject(
                    SpawnRejection::MaxTotalPerRoot { spawned, max },
                    parent_run_id,
                ));
            }
        }
        if let Some(max) = self.policy.max_children_per_parent {
            let active = ledger
                .active_per_parent
                .get(parent_run_id)
                .copied()
                .unwrap_or(0);
            if active >= max {
                drop(ledger);
                return Err(self.reject(
                    SpawnRejection::MaxChildrenPerParent { active, max },
                    parent_run_id,
                ));
            }
        }
        *ledger
            .active_per_parent
            .entry(parent_run_id.to_owned())
            .or_default() += 1;
        if counts_toward_total {
            *ledger
                .spawned_per_root
                .entry(root_run_id.to_owned())
                .or_default() += 1;
        }
        tracing::debug!(
            "{LOG_PREFIX} reserved parent={parent_run_id} root={root_run_id} target={target}"
        );
        Ok(SpawnReservation {
            admission: self.clone(),
            root_run_id: root_run_id.to_owned(),
            parent_run_id: parent_run_id.to_owned(),
            counts_toward_total,
            committed: false,
        })
    }

    fn reject(&self, rejection: SpawnRejection, parent_run_id: &str) -> SpawnRejection {
        tracing::debug!("{LOG_PREFIX} rejected parent={parent_run_id} reason={rejection}");
        rejection
    }
}

/// An admitted spawn slot. Dropping it releases the slot; see the module docs.
#[must_use = "dropping a reservation immediately releases its slot"]
#[derive(Debug)]
pub struct SpawnReservation {
    admission: SpawnAdmission,
    root_run_id: String,
    parent_run_id: String,
    counts_toward_total: bool,
    committed: bool,
}

impl SpawnReservation {
    /// Marks the spawn as having happened. The live slot is still released on
    /// drop (the child's terminal state), but the root's total budget stays
    /// spent.
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for SpawnReservation {
    fn drop(&mut self) {
        let mut ledger = self.admission.ledger.lock();
        decrement(&mut ledger.active_per_parent, &self.parent_run_id);
        if self.counts_toward_total && !self.committed {
            decrement(&mut ledger.spawned_per_root, &self.root_run_id);
        }
        tracing::debug!(
            "{LOG_PREFIX} released parent={} committed={}",
            self.parent_run_id,
            self.committed
        );
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
