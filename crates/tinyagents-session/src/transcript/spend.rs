//! Token/cost spend totals read back from persisted session transcripts.
//!
//! Every transcript records only what the agent that owns it spent, so walking a
//! thread's root transcripts and their `{root_stem}__{child}` descendants counts
//! each token exactly once. Pricing (re-auditing cost at current rates) is the
//! host's concern and stays out of here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{SessionTranscript, find_root_transcripts_for_thread, read_transcript};

/// One transcript's own spend, summed from the per-turn `turn_usage` records
/// the codec attaches to its assistant rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptSpend {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cost_usd: f64,
    /// Turns that recorded usage. The codec emits one record per durable
    /// append, and omits an all-zero one, so this is "turns that spent".
    pub turns: usize,
    /// The newest record's model and window, for the caller's last-turn view.
    pub last_input_tokens: u64,
    pub last_output_tokens: u64,
    pub model: Option<String>,
    pub context_window: u64,
}

/// Total one transcript file's own recorded spend.
///
/// The `_meta` header carries denormalised rollups for the same figures, but
/// nothing has written them on the root path since the TinyAgents runtime
/// cutover — every root transcript since reads `input_tokens: 0`
/// while its per-turn records hold the real numbers (#6460). The per-turn
/// records are the authoritative copy and the only one that is written on every
/// path, so this reads those and ignores the header.
///
/// `read_transcript` is the reader that applies compaction records and drops
/// interrupted partials, so a compacted or interrupted session is summed over
/// its logical message set rather than its raw append log.
pub fn transcript_spend(transcript: &SessionTranscript) -> TranscriptSpend {
    let mut spend = TranscriptSpend::default();
    for message in &transcript.messages {
        let Some(usage) = message.turn_usage.as_ref() else {
            continue;
        };
        // A text-dialect tool round's issuing row carries a provenance-only
        // record (its calls, zero spend); the turn's spend is on its final row.
        // It is not a turn that spent, and must not become the "last" one.
        if !usage.tool_calls.is_empty()
            && usage.usage.input == 0
            && usage.usage.output == 0
            && usage.usage.cached_input == 0
            && usage.usage.cost_usd == 0.0
        {
            continue;
        }
        spend.input_tokens = spend.input_tokens.saturating_add(usage.usage.input);
        spend.output_tokens = spend.output_tokens.saturating_add(usage.usage.output);
        spend.cached_input_tokens = spend
            .cached_input_tokens
            .saturating_add(usage.usage.cached_input);
        spend.cost_usd += usage.usage.cost_usd;
        spend.turns += 1;
        spend.last_input_tokens = usage.usage.input;
        spend.last_output_tokens = usage.usage.output;
        if usage.usage.context_window > 0 {
            spend.context_window = usage.usage.context_window;
        }
        if !usage.model.is_empty() {
            spend.model = Some(usage.model.clone());
        }
    }
    // A transcript whose model never landed on a usage record still has one in
    // its header; prefer the record, fall back to the header.
    if spend.model.is_none() {
        spend.model = transcript.meta.model.clone();
    }
    spend
}

/// Read one transcript, logging and skipping an unreadable file rather than
/// failing the whole thread's aggregate for it.
fn read(path: &Path) -> Option<SessionTranscript> {
    match read_transcript(path) {
        Ok(transcript) => Some(transcript),
        Err(err) => {
            tracing::warn!(
                "[transcript:spend] skipping unreadable transcript {}: {err}",
                path.display()
            );
            None
        }
    }
}

/// Every descendant transcript of `root`, at any delegation depth.
///
/// Sub-agent transcripts are named `{root_stem}__{child}` (and a grandchild
/// chains another `__`), which is the *only* durable record of the parent→child
/// relation. Selecting them by `_meta.thread_id` instead — as the session
/// crate's own summary helper does — drops the majority of them: a delegation
/// gets its own worker thread, and `inherited_thread_id` lets that worker id win
/// over the parent's, so the child's header names a thread the caller never
/// asked about (#6460).
fn descendant_transcripts(root: &Path) -> Vec<PathBuf> {
    let Some(dir) = root.parent() else {
        return Vec::new();
    };
    let Some(root_stem) = root.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let prefix = format!("{root_stem}__");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut matches: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem.starts_with(&prefix))
        })
        .collect();
    matches.sort();
    matches
}

/// One thread's spend, split into the orchestrator's own and each sub-agent
/// archetype's, each counted exactly once.
#[derive(Debug, Clone, Default)]
pub struct ThreadSpend {
    pub root: TranscriptSpend,
    /// Keyed by archetype (`_meta.agent`), with a run count.
    pub subagents: BTreeMap<String, (TranscriptSpend, usize)>,
    pub updated: Option<String>,
    pub found_transcript: bool,
}

/// Walk a thread's root transcripts and their descendants, totalling each file's
/// own recorded spend.
///
/// Correct-by-construction because every transcript records only what the agent
/// that owns it spent (see the host's per-turn usage codec): the walk visits
/// each file once, so no token is counted twice however deep the delegation went,
/// and a child whose usage never reached its parent's in-turn ledger (#6459) is
/// still counted from its own file.
pub fn thread_spend(workspace_dir: &Path, thread_id: &str) -> ThreadSpend {
    let mut out = ThreadSpend::default();
    let roots = find_root_transcripts_for_thread(workspace_dir, thread_id);
    for root in &roots {
        out.found_transcript = true;
        if let Some(transcript) = read(root) {
            let spend = transcript_spend(&transcript);
            out.root.input_tokens = out.root.input_tokens.saturating_add(spend.input_tokens);
            out.root.output_tokens = out.root.output_tokens.saturating_add(spend.output_tokens);
            out.root.cached_input_tokens = out
                .root
                .cached_input_tokens
                .saturating_add(spend.cached_input_tokens);
            out.root.cost_usd += spend.cost_usd;
            out.root.turns += spend.turns;
            // `find_root_transcripts_for_thread` returns oldest first, so the
            // last root to report a turn owns the last-turn view.
            if spend.turns > 0 {
                out.root.last_input_tokens = spend.last_input_tokens;
                out.root.last_output_tokens = spend.last_output_tokens;
            }
            if spend.context_window > 0 {
                out.root.context_window = spend.context_window;
            }
            if spend.model.is_some() {
                out.root.model = spend.model;
            }
            out.updated = Some(transcript.meta.updated);
        }
        for child in descendant_transcripts(root) {
            let Some(child_transcript) = read(&child) else {
                continue;
            };
            let spend = transcript_spend(&child_transcript);
            out.found_transcript = true;
            let entry = out
                .subagents
                .entry(child_transcript.meta.agent_name.clone())
                .or_default();
            entry.0.input_tokens = entry.0.input_tokens.saturating_add(spend.input_tokens);
            entry.0.output_tokens = entry.0.output_tokens.saturating_add(spend.output_tokens);
            entry.0.cached_input_tokens = entry
                .0
                .cached_input_tokens
                .saturating_add(spend.cached_input_tokens);
            entry.0.cost_usd += spend.cost_usd;
            entry.0.turns += spend.turns;
            if entry.0.model.is_none() {
                entry.0.model = spend.model;
            }
            entry.1 += 1;
        }
    }
    out
}

#[cfg(test)]
#[path = "spend_test.rs"]
mod tests;
