//! Volatility-aware outcome fingerprinting.

/// Reduces a tool outcome to a stable identity.
pub trait OutcomeFingerprinter: Send + Sync {
    /// Returns the identity of `outcome`.
    fn fingerprint(&self, outcome: &str) -> String;
}

/// Default fingerprinter.
#[derive(Debug, Default, Clone, Copy)]
pub struct VolatileSpanNormalizer;

impl OutcomeFingerprinter for VolatileSpanNormalizer {
    fn fingerprint(&self, outcome: &str) -> String {
        outcome.to_string()
    }
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
