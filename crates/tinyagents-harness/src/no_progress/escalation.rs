//! Configuration for the staged escalation of successful repeats.

/// Staged escalation of a successful repeat: **warn**, then **block**, then
/// **halt**.
///
/// The first threshold a repeat reaches (the tracker's `call_threshold` /
/// `output_threshold`) only *warns*: the host appends a note to the tool result
/// telling the model it is repeating itself. If the model repeats anyway the
/// call is **blocked** (not executed, answered with an error asking it to
/// reassess); a second block in the same run **halts** the run.
///
/// Without escalation (`Option<RepeatEscalation>` = `None` on the tracker, the
/// default for [`SuccessfulRepeatTracker::new`](super::SuccessfulRepeatTracker::new))
/// the first threshold halts immediately, which is the historical behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatEscalation {
    /// How many repeats past the warning the call is blocked at: with the
    /// default warning at 3 identical results, the 5th identical call is the
    /// first one blocked. Clamped to at least 1 so a warning always lands
    /// before a block.
    pub block_after_warn: u32,
    /// Blocks tolerated in one run before the run halts; the halting block is
    /// the `blocks_before_halt`-th. Clamped to at least 1 (halt on the first
    /// block).
    pub blocks_before_halt: u32,
}

/// Default for [`RepeatEscalation::block_after_warn`].
pub const DEFAULT_BLOCK_AFTER_WARN: u32 = 2;
/// Default for [`RepeatEscalation::blocks_before_halt`].
pub const DEFAULT_BLOCKS_BEFORE_HALT: u32 = 2;

impl Default for RepeatEscalation {
    fn default() -> Self {
        Self {
            block_after_warn: DEFAULT_BLOCK_AFTER_WARN,
            blocks_before_halt: DEFAULT_BLOCKS_BEFORE_HALT,
        }
    }
}

impl RepeatEscalation {
    pub(super) fn gap(&self) -> u32 {
        self.block_after_warn.max(1)
    }

    pub(super) fn halt_block(&self) -> u32 {
        self.blocks_before_halt.max(1)
    }
}

#[cfg(test)]
#[path = "escalation_tests.rs"]
mod tests;
