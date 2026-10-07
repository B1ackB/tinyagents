//! Result policy: how much of a child's final output reaches its parent, and
//! whether it is checked against a schema.
//!
//! The default policy changes nothing. A cap trims the visible text to its
//! head and tail around an explicit omission marker; with
//! [`ResultOverflow::Artifact`] the full output is first handed to a host
//! [`ArtifactStore`] and the parent gets the preview plus a path-free
//! [`ArtifactReference`]. An optional JSON schema is checked with the same
//! structural validator the tool-call boundary uses (`type`, `properties`,
//! `required`, `additionalProperties`, `items`, `enum`; no `$ref` or
//! combinators) and a mismatch is *reported* in
//! [`AppliedResult::schema_error`], never turned into a failure.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::ArtifactReference;

const LOG_PREFIX: &str = "[subagent-result-policy]";

/// What to do with output longer than [`ResultPolicy::max_chars`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResultOverflow {
    /// Keep the head and tail with an omission marker.
    #[default]
    Truncate,
    /// Store the full output via the [`ArtifactStore`], keep a truncated
    /// preview and the reference. Falls back to `Truncate` without a store.
    Artifact,
}

/// Host seam that persists an oversized output and names it neutrally.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Stores `content` for `task_id`; the reference must carry no path or URL.
    async fn store(&self, task_id: &str, content: &str) -> Result<ArtifactReference, String>;
}

/// Cap, overflow behaviour and optional schema for a child's final output.
#[derive(Clone, Default)]
pub struct ResultPolicy {
    /// Most characters of output kept visible; `None` is uncapped.
    pub max_chars: Option<usize>,
    /// What happens to output beyond the cap.
    pub overflow: ResultOverflow,
    /// JSON schema the final output must satisfy, when set.
    pub schema: Option<Value>,
    /// Where [`ResultOverflow::Artifact`] puts the full output.
    pub artifact_store: Option<Arc<dyn ArtifactStore>>,
}

impl std::fmt::Debug for ResultPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultPolicy")
            .field("max_chars", &self.max_chars)
            .field("overflow", &self.overflow)
            .field("schema", &self.schema)
            .field("artifact_store", &self.artifact_store.is_some())
            .finish()
    }
}

/// The policy applied to one output.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppliedResult {
    /// The text the parent sees.
    pub text: String,
    /// Characters dropped from the middle; `0` when nothing was cut.
    pub omitted_chars: usize,
    /// The stored full output, for [`ResultOverflow::Artifact`].
    pub artifact: Option<ArtifactReference>,
    /// Why the output does not satisfy [`ResultPolicy::schema`].
    pub schema_error: Option<String>,
    /// Why an artifact was wanted but not stored (no store, or store failed).
    pub artifact_error: Option<String>,
}

impl ResultPolicy {
    /// A policy that changes nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Caps the visible output at `max_chars` characters.
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = Some(max_chars);
        self
    }

    /// Chooses the overflow behaviour.
    pub fn with_overflow(mut self, overflow: ResultOverflow) -> Self {
        self.overflow = overflow;
        self
    }

    /// Requires the final output to satisfy `schema`.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Sets the store used by [`ResultOverflow::Artifact`].
    pub fn with_artifact_store(mut self, store: Arc<dyn ArtifactStore>) -> Self {
        self.artifact_store = Some(store);
        self
    }

    /// Whether applying this policy can alter or annotate an output.
    pub fn is_active(&self) -> bool {
        self.max_chars.is_some() || self.schema.is_some()
    }

    /// Applies the policy to `full`. `structured` is the run's already parsed
    /// value, preferred over re-parsing `full` for schema validation.
    pub async fn apply(
        &self,
        task_id: &str,
        full: &str,
        structured: Option<&Value>,
    ) -> AppliedResult {
        let mut applied = AppliedResult {
            text: full.to_owned(),
            ..AppliedResult::default()
        };
        if let Some(schema) = &self.schema {
            applied.schema_error = validate_output(schema, full, structured);
            if let Some(error) = &applied.schema_error {
                tracing::debug!("{LOG_PREFIX} schema_error task_id={task_id} error={error}");
            }
        }
        let Some(max) = self.max_chars else {
            return applied;
        };
        let (text, omitted) = truncate_head_tail(full, max);
        if omitted == 0 {
            return applied;
        }
        applied.text = text;
        applied.omitted_chars = omitted;
        if self.overflow == ResultOverflow::Artifact {
            match &self.artifact_store {
                Some(store) => match store.store(task_id, full).await {
                    Ok(reference) => applied.artifact = Some(reference),
                    Err(error) => {
                        tracing::warn!("{LOG_PREFIX} artifact_store_failed task_id={task_id}");
                        applied.artifact_error = Some(format!("artifact store failed: {error}"));
                    }
                },
                None => {
                    tracing::warn!("{LOG_PREFIX} artifact_without_store task_id={task_id}");
                    applied.artifact_error = Some("no artifact store configured".to_owned());
                }
            }
        }
        applied
    }
}

/// Shortens `text` to at most `max_chars` characters *including* a
/// `[… N chars omitted …]` marker between its head and tail; returns the text
/// and `N` (`0` when it already fits). Counts characters, never splitting a
/// code point. When `max_chars` is smaller than the marker itself, only the
/// marker is returned.
pub fn truncate_head_tail(text: &str, max_chars: usize) -> (String, usize) {
    let total = text.chars().count();
    if total <= max_chars {
        return (text.to_owned(), 0);
    }
    let marker = |omitted: usize| format!("\n[… {omitted} chars omitted …]\n");
    // The marker's width depends on the omitted count, which depends on how
    // much the marker leaves room for: iterate to a fixed point.
    let mut kept = max_chars;
    let mut omitted = total - kept;
    for _ in 0..4 {
        let room = max_chars.saturating_sub(marker(omitted).chars().count());
        if room == kept {
            break;
        }
        kept = room;
        omitted = total - kept;
    }
    let head = kept.div_ceil(2);
    let tail = kept - head;
    let head_end = text.char_indices().nth(head).map_or(text.len(), |(i, _)| i);
    let tail_start = text
        .char_indices()
        .nth(total - tail)
        .map_or(text.len(), |(i, _)| i);
    (
        format!(
            "{}{}{}",
            &text[..head_end],
            marker(omitted),
            &text[tail_start..]
        ),
        omitted,
    )
}

fn validate_output(schema: &Value, text: &str, structured: Option<&Value>) -> Option<String> {
    let parsed;
    let value = match structured {
        Some(value) => value,
        None => match serde_json::from_str::<Value>(text.trim()) {
            Ok(value) => {
                parsed = value;
                &parsed
            }
            Err(error) => return Some(format!("output is not valid JSON: {error}")),
        },
    };
    match tinyinference_llm::tool::validate_json_value(schema, value, "output") {
        Ok(()) => None,
        Err(tinyinference_llm::Error::Validation(message)) => Some(message),
        Err(other) => Some(other.to_string()),
    }
}

#[cfg(test)]
#[path = "result_policy_tests.rs"]
mod tests;
