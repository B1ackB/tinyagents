//! Bounded in-memory ZIP, TAR and TAR.GZ inspection without filesystem extraction.

mod ops;
mod types;
pub(super) mod zip_admission;

pub use ops::inspect_archive;
pub use types::*;
