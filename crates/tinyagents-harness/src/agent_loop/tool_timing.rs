//! Wall-clock duration shown to the model on each tool result (#6953 stub).

use tinyinference_llm::message::ToolMessage;

pub(super) fn duration_suffix(_duration_ms: u64) -> String {
    String::new()
}

pub(super) fn append_duration(_message: &mut ToolMessage, _duration_ms: u64) {}

#[cfg(test)]
#[path = "tool_timing_tests.rs"]
mod tests;
