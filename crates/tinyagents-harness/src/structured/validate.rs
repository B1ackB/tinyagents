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

use crate::error::{Result, TinyAgentsError};

/// Validates `value` against `schema`, reporting the failing instance path.
///
/// `root` names the value in error messages — the caller passes something like
/// `schema 'review'` so the message reads `schema 'review'.items[2].id must be
/// integer, got string`.
pub fn validate_value(schema: &Value, value: &Value, root: &str) -> Result<()> {
    validate_at(schema, value, root)
}

/// Recursive validation worker behind [`validate_value`]: checks `enum`,
/// `type`, `required`, `properties` (+ `additionalProperties: false`), and
/// `items` at one schema node, then recurses into matched properties/items
/// with `path` extended to name them.
fn validate_at(schema: &Value, value: &Value, path: &str) -> Result<()> {
    if schema.is_null() || schema.as_object().is_some_and(|map| map.is_empty()) {
        return Ok(());
    }

    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.iter().any(|candidate| candidate == value)
    {
        return Err(invalid(format!(
            "{path} must be one of the declared enum values"
        )));
    }

    if let Some(type_spec) = schema.get("type") {
        validate_type(type_spec, value, path)?;
    }

    // `required` is enforced independently of `properties`: a schema may name
    // required fields without describing them, and nesting the check under
    // `properties` would let such a schema fail open.
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        if let Some(object) = value.as_object() {
            for field in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(field) {
                    return Err(invalid(format!("{path}.{field} is required")));
                }
            }
        } else if schema.get("type").is_none() {
            return Err(invalid(format!(
                "{path} must be an object with the declared fields, got {}",
                kind_of(value)
            )));
        }
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        if let Some(object) = value.as_object() {
            if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(invalid(format!("{path}.{field} is not allowed")));
                    }
                }
            }
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_at(field_schema, field_value, &format!("{path}.{field}"))?;
                }
            }
        } else if schema.get("type").is_none() {
            return Err(invalid(format!(
                "{path} must be an object with the declared fields, got {}",
                kind_of(value)
            )));
        }
    }

    if let Some(items_schema) = schema.get("items")
        && let Some(items) = value.as_array()
    {
        for (index, item) in items.iter().enumerate() {
            validate_at(items_schema, item, &format!("{path}[{index}]"))?;
        }
    }

    Ok(())
}

/// Checks `value` against a `type` keyword, which may be a single type name
/// or an array of alternatives (a union), matching any one.
fn validate_type(type_spec: &Value, value: &Value, path: &str) -> Result<()> {
    if let Some(kind) = type_spec.as_str() {
        if matches_type(value, kind) {
            return Ok(());
        }
        return Err(invalid(format!(
            "{path} must be {kind}, got {}",
            kind_of(value)
        )));
    }

    if let Some(kinds) = type_spec.as_array() {
        let allowed: Vec<&str> = kinds.iter().filter_map(Value::as_str).collect();
        if allowed.iter().any(|kind| matches_type(value, kind)) {
            return Ok(());
        }
        return Err(invalid(format!(
            "{path} must be one of {}, got {}",
            allowed.join(", "),
            kind_of(value)
        )));
    }

    Ok(())
}

/// Whether `value`'s runtime JSON kind satisfies the named schema `kind`.
/// An unrecognised `kind` always matches (see the module doc's "supported
/// subset" note).
fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        // An unknown type keyword must not fail closed: providers accept richer
        // vocabularies than this subset understands.
        _ => true,
    }
}

/// Names a JSON value's kind for error messages, distinguishing `integer`
/// from `number` (a schema `type` vocabulary distinction JSON itself does not
/// make).
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Wraps a validation failure message as the error variant used throughout
/// this module.
fn invalid(message: String) -> TinyAgentsError {
    TinyAgentsError::StructuredOutput(message)
}

#[cfg(test)]
#[path = "validate_test_tests.rs"]
mod test;
