//! Background agent command center: a read-only projection of the durable run
//! ledger into five user-facing buckets (needs-input / working / completed /
//! failed / stopped), plus durable control verbs (stop / retry / continue /
//! follow-up) that move a run through the ledger.
//!
//! Live run state already persists to the ledger via the host's spawn tools and
//! progress bridge; this module only projects and transitions it. Hosts supply
//! an agent-id to display-name closure and own the RPC surface.

mod control;
mod types;
mod view;

pub use control::{ControlError, ControlVerb, apply_control};
pub use types::{AgentWorkBucket, AgentWorkRow, CommandCenterGroup, CommandCenterView};
pub use view::{bucket_for, build_view, list_agent_work};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
