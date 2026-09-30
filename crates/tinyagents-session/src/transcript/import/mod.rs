//! One-time import of legacy sessions into TinyAgents stores, plus the live
//! dual-write and shadow read that keep new turns in the same layout.
//!
//! Legacy transcript JSONL (`session_raw/`, flat and `DDMMYYYY` date folders)
//! and legacy Markdown sessions are normalized into TinyAgents
//! `Store`/`AppendStore` records under `{workspace}/tinyagents_store/`. Sources
//! are never mutated; [`ops::run_import`] is idempotent (global marker +
//! per-item fingerprint ledger). See [`README.md`](README.md) for the
//! source/destination layout.
//!
//! The host supplies one seam: a [`convert::JournalProjector`] that turns a
//! durable [`TranscriptMessage`](crate::transcript::TranscriptMessage) into the
//! journal record, so a host that folds sidecar metadata into its message type
//! keeps that metadata in the journal. [`convert::plain_journal_message`] is the
//! neutral default. Whether the live mirror runs at all is the host's decision.

pub mod convert;
pub mod live;
pub mod ops;
pub mod scan;
pub mod types;

#[cfg(test)]
mod convert_test;
#[cfg(test)]
mod live_test;
#[cfg(test)]
mod ops_test;
