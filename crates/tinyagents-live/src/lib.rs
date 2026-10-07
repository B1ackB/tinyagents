//! Live (realtime voice) agent sessions on top of the TinyAgents harness.
//!
//! [`tinyliveagents`] standardizes live voice providers — Gemini Live,
//! ElevenLabs Agents, Sarvam — behind one session type, but deliberately does
//! not run tools. This crate does: [`LiveAgent::start`] declares a harness's
//! tools to the live model and executes every tool call the model makes
//! through the harness's own pipeline
//! ([`tinyagents_harness::agent_loop::phases::execute_tool_batch`]), so host
//! middleware — approvals, tool policy, budgets, credential scrubbing — applies
//! to a spoken request exactly as to a typed one. Results go back to the
//! provider automatically; the host only moves audio and renders
//! [`LiveAgentEvent`]s.
//!
//! What stays with the host: choosing and connecting the provider (and minting
//! any relay ticket), building the harness and its [`RunContext`], audio I/O,
//! and persisting transcripts.
//!
//! [`RunContext`]: tinyagents_harness::context::RunContext

mod declarations;
mod runner;
mod types;
mod worker;

pub use declarations::tool_declarations;
pub use runner::{LiveAgent, LiveAgentSession};
pub use tinyliveagents;
pub use types::{BoxedTask, LiveAgentEvent, LiveAgentOptions, TaskScope};
