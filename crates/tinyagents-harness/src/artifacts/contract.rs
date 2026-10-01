//! The prompt-side half of the artifact-offload convention (#3883).
//!
//! Single source of truth for the directory names the model is told about, so
//! the prompt and [`super::resolve_artifact_path`] can never drift. The write
//! tool the contract may name is a parameter: tool names are host vocabulary.

use std::collections::HashSet;

use super::{OUTPUTS_DIR, SCRATCH_DIR};

/// Approximate token budget quoted to the model as the offload trigger.
/// Mirrors [`super::DEFAULT_OFFLOAD_THRESHOLD_BYTES`] at the harness-wide
/// 4-chars-per-token estimate.
const CONTRACT_THRESHOLD_TOKENS: usize = 2_000;

/// Heading the contract is rendered under. Hosts use it for an idempotence check
/// so a re-rendered prompt never stacks two copies.
pub const ARTIFACT_OFFLOAD_HEADING: &str = "## Long-horizon Artifact Offload";

/// Whether the offload contract should be rendered for an agent whose visible
/// tools are `visible_tool_names`.
///
/// A prompt may only name tools the agent can really call: advertising one it
/// lacks produces hallucinated calls that fail. So only agents holding
/// `write_tool` are told to offload. Everyone else stays covered by the
/// harness-side automatic offload, which needs no cooperation from the model.
pub fn should_render_offload_contract(
    visible_tool_names: &HashSet<String>,
    write_tool: &str,
) -> bool {
    visible_tool_names.contains(write_tool)
}

/// Render the offload contract for an agent that holds the file-write tool
/// `write_tool`.
///
/// Byte-stable for a given `write_tool`: the system prompt is prefix-cached, so
/// this must render identically on every run. Absolute paths are never
/// interpolated for the same reason (and because a worktree-isolated worker
/// resolves its own action root). Names no tool other than `write_tool`.
pub fn render_artifact_offload_contract(write_tool: &str) -> String {
    format!(
        "{ARTIFACT_OFFLOAD_HEADING}\n\n\
Large results belong on disk, not in your reply. Two directories sit under your action directory:\n\
- `{OUTPUTS_DIR}/` for deliverables, anything the parent or a later step needs to read.\n\
- `{SCRATCH_DIR}/` for scratch, intermediate files you do not intend to hand back.\n\
\n\
When a result would run past roughly {CONTRACT_THRESHOLD_TOKENS} tokens:\n\
1. Write the full content to a file under `{OUTPUTS_DIR}/` with `{write_tool}`.\n\
2. Reply with that relative path plus a short abstract, not the full payload.\n\
3. Quote the path exactly, so the parent can open it verbatim.\n\
\n\
Rules:\n\
- Offload paths are always relative to your action directory. Never write outside it, and never target the core's internal workspace state.\n\
- Keep the abstract honest: say what the file holds and what is still open. Never present it as the complete result.\n\
- Small results stay inline. A pointer to a two-line file costs the parent more than the two lines.\n\
- If you inline an oversized result anyway, the harness persists it under `{OUTPUTS_DIR}/` and hands the parent the path instead.\n"
    )
}

#[cfg(test)]
#[path = "contract_test.rs"]
mod tests;
