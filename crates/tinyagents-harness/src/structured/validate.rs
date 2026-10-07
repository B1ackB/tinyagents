//! Local validation of an extracted structured value against its declared
//! JSON Schema.
//!
//! # Why this exists
//!
//! [`StructuredExtractor`][super::StructuredExtractor] stored its schema and
//! never read it: provider-schema mode ran a bare
//! [`serde_json::from_str`] and tool-call mode cloned the call's arguments
//! straight through. So `{"wrong_key": 1}` against a `score` schema *succeeded*,
//! and `run.structured` came back holding something no caller had asked for —
//! a failure that surfaces later, somewhere else, as a missing field.
//!
//! Validating here also gives the repair loop something to say: an error naming
//! the exact failing instance path is a message that can be handed back to the
//! model, where "deserialisation failed" is not.
//!
//! # The supported subset
//!
//! The same subset the tool-call boundary enforces: `type` (including union
//! types), object `properties`, `required`, `additionalProperties: false`,
//! array `items`, and `enum`. Unknown keywords are ignored, so a richer schema
//! can still be sent to a provider while the local boundary fails closed on
//! exactly the structural constraints it understands. An empty or null schema
//! imposes no constraints.
//!
//! It is intentionally **not** a general JSON Schema implementation: no
//! `$ref`, no `allOf`/`anyOf`/`oneOf`, no numeric or string facets. Those
//! belong in a dedicated validator crate if the need ever arises; guessing at
//! them here would produce confident wrong answers.

use serde_json::Value;
use tinyinference_llm::tool::validate_json_value;

use crate::error::{Result, TinyAgentsError};

/// Validates `value` against `schema`, reporting the failing instance path.
///
/// `root` names the value in error messages — the caller passes something like
/// `schema 'review'` so the message reads `schema 'review'.items[2].id must be
/// integer, got string`.
///
/// The checking itself is [`tinyinference_llm::tool::validate_json_value`],
/// the same validator the tool-call boundary uses, so a structured output and
/// a tool call are held to one definition of the supported subset.
///
/// # Errors
///
/// Returns [`TinyAgentsError::StructuredOutput`] naming the first failing
/// instance path.
pub fn validate_value(schema: &Value, value: &Value, root: &str) -> Result<()> {
    validate_json_value(schema, value, root).map_err(|error| {
        TinyAgentsError::StructuredOutput(match error {
            tinyinference_llm::Error::Validation(message) => message,
            other => other.to_string(),
        })
    })
}

#[cfg(test)]
#[path = "validate_test_tests.rs"]
mod test;
