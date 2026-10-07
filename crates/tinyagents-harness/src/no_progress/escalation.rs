//! Configuration for the staged escalation of successful repeats.

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
    /// Escalation that blocks `block_after_warn` repeats after the warning and
    /// halts on the `blocks_before_halt`-th block of one call.
    pub fn new(block_after_warn: u32, blocks_before_halt: u32) -> Self {
        Self {
            block_after_warn,
            blocks_before_halt,
        }
    }

    /// Sets [`block_after_warn`](Self::block_after_warn).
    pub fn with_block_after_warn(mut self, repeats: u32) -> Self {
        self.block_after_warn = repeats;
        self
    }

    /// Sets [`blocks_before_halt`](Self::blocks_before_halt).
    pub fn with_blocks_before_halt(mut self, blocks: u32) -> Self {
        self.blocks_before_halt = blocks;
        self
    }

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
