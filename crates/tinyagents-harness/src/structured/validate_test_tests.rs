use super::*;
use serde_json::json;

fn score_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "score": { "type": "integer" } },
        "required": ["score"],
        "additionalProperties": false
    })
}

#[test]
fn accepts_a_conforming_value() {
    validate_value(&score_schema(), &json!({ "score": 4 }), "schema 'score'").unwrap();
}

#[test]
fn rejects_a_missing_required_field_by_path() {
    let err = validate_value(
        &score_schema(),
        &json!({ "wrong_key": 1 }),
        "schema 'score'",
    )
    .expect_err("a missing required field is not valid");
    assert!(
        err.to_string().contains("schema 'score'.score is required"),
        "{err}"
    );
}

#[test]
fn rejects_a_wrong_type_by_path() {
    let err = validate_value(
        &score_schema(),
        &json!({ "score": "four" }),
        "schema 'score'",
    )
    .expect_err("a string is not an integer");
    assert!(
        err.to_string()
            .contains("schema 'score'.score must be integer, got string"),
        "{err}"
    );
}

#[test]
fn reports_a_nested_array_index() {
    let schema = json!({
        "type": "object",
        "properties": {
            "items": { "type": "array", "items": { "type": "object", "properties": { "id": { "type": "integer" } } } }
        }
    });
    let err = validate_value(
        &schema,
        &json!({ "items": [{ "id": 1 }, { "id": "two" }] }),
        "schema 'batch'",
    )
    .expect_err("the second item is invalid");
    assert!(err.to_string().contains("items[1].id"), "{err}");
}

#[test]
fn rejects_an_undeclared_field_when_additional_properties_is_false() {
    let err = validate_value(
        &score_schema(),
        &json!({ "score": 4, "extra": true }),
        "schema 'score'",
    )
    .expect_err("`extra` is not declared");
    assert!(err.to_string().contains("extra is not allowed"), "{err}");
}

#[test]
fn an_empty_schema_constrains_nothing() {
    validate_value(&json!({}), &json!("anything at all"), "schema 'free'").unwrap();
    validate_value(&Value::Null, &json!(7), "schema 'free'").unwrap();
}

#[test]
fn accepts_a_union_type() {
    let schema = json!({ "type": ["string", "null"] });
    validate_value(&schema, &json!(null), "schema 'maybe'").unwrap();
    validate_value(&schema, &json!("x"), "schema 'maybe'").unwrap();
    assert!(validate_value(&schema, &json!(3), "schema 'maybe'").is_err());
}

#[test]
fn ignores_unknown_type_keywords() {
    // A provider may accept a richer vocabulary than this subset knows.
    validate_value(&json!({ "type": "date-time" }), &json!("2026-01-01"), "s").unwrap();
}

#[test]
fn enforces_an_enum() {
    let schema = json!({ "enum": ["a", "b"] });
    validate_value(&schema, &json!("a"), "schema 'choice'").unwrap();
    assert!(validate_value(&schema, &json!("c"), "schema 'choice'").is_err());
}
