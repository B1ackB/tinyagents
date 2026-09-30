//! Configurable command hooks: a Cursor-compatible `hooks.json` engine.
//!
//! User-authored scripts, checked into a repository or installed system-wide,
//! observe and gate an agent: block `rm -rf`, run a formatter after an edit,
//! write an audit line per tool call. The contract follows Cursor's
//! `hooks.json` (<https://cursor.com/docs/hooks>): same event names, same stdin
//! envelope, same stdout decision object, same exit-code semantics.
//!
//! This module is host-neutral. The host supplies a [`HookEnvironment`]
//! (product name, home directory, shell, prompt evaluator) and mounts the
//! [`HookEngine`] onto its own tool/turn seams.
//!
//! | Module | Owns |
//! | ------ | ---- |
//! | [`types`] | the wire contract: events, the stdin envelope, the decision object |
//! | [`config`] | `hooks.json` parsing and the four-layer merge |
//! | [`matcher`] | which occurrences of an event reach a given hook |
//! | [`exec`] | running one hook: stdin, timeout, exit codes, fail-open/closed |
//! | [`engine`] | selection, ordering, aggregation, session state |
//! | [`context`] | assembling the envelope from ambient host facts |
//! | [`followup`] | queueing what a `stop` hook asks for next |
//! | [`environment`] | the seams a host provides |
//!
//! **The strictest verdict wins.** Layers concatenate rather than override, and
//! [`types::HookOutput::merge`] folds denial over ask over allow.
//!
//! **Gating costs a turn's latency; observing does not.**
//! [`types::HookEvent::is_gating`] decides between running hooks sequentially
//! in the turn's path and spawning them onto a background task.

pub mod config;
pub mod context;
pub mod engine;
pub mod environment;
pub mod exec;
pub mod followup;
pub mod matcher;
pub mod types;

#[cfg(test)]
mod tests;

pub use config::{HookConfig, HookDefinition, HookKind, HookLayer, HooksFile};
pub use engine::{HookEngine, HookOutcome};
pub use environment::{HookEnvironment, PromptEvaluator, ShellFactory};
pub use types::{HookEvent, HookInput, HookOutput, HookPayload, HookPermission};
