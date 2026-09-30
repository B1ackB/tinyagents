//! Optional builtin harness tools.
//!
//! These tools implement the canonical [`tinytools::Tool`] interface. They are behind the `tools`
//! Cargo feature so applications that provide their own tool surface do not pull
//! in extra dependencies by default.

mod ask_clarification;
mod time;
mod time_parse;
mod wait;

pub use ask_clarification::AskClarificationTool;
pub use time::{CurrentTimeTool, ResolveTimeTool, register_time_tools, time_tools};
pub use wait::{WaitLoopTool, WaitTool};

#[cfg(test)]
mod ask_clarification_test;
#[cfg(test)]
mod time_parse_test;
#[cfg(test)]
mod time_test;
#[cfg(test)]
mod wait_test;
