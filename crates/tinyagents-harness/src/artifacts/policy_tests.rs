use super::*;

#[test]
fn no_redaction_reports_the_body_unchanged() {
    let out = NoRedaction.redact("hunter2");
    assert_eq!(out.text, "hunter2");
    // If this ever reported `changed`, every pointer would carry a
    // redaction note for a body nothing touched.
    assert!(!out.changed);
}

#[test]
fn open_policy_forbids_nothing() {
    assert!(!OpenPathPolicy.is_internal_state(Path::new("/anywhere")));
    assert_eq!(OpenPathPolicy.internal_root(), None);
}

#[test]
fn redacted_constructors_set_the_changed_flag() {
    assert!(!Redacted::unchanged("a").changed);
    assert!(Redacted::rewritten("b").changed);
}
