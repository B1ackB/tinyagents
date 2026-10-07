use super::*;
use chrono::Utc;

fn call(call_id: &str) -> DanglingToolCall {
    DanglingToolCall {
        run_id: "run-1".into(),
        call_id: call_id.into(),
        tool: "shell".into(),
    }
}

fn effect(run_id: &str, call_id: &str, status: ToolEffectStatus) -> ToolEffectRow {
    ToolEffectRow {
        run_id: run_id.into(),
        call_id: call_id.into(),
        tool: "shell".into(),
        status,
        idempotency_key: Some("key".into()),
        effect_summary: None,
        started_at: Utc::now(),
        settled_at: None,
    }
}

fn class_of(status: Option<ToolEffectStatus>) -> RecoveryClass {
    let effects: Vec<_> = status
        .map(|status| effect("run-1", "c1", status))
        .into_iter()
        .collect();
    classify_recovery(&[call("c1")], &effects)[0].class
}

#[test]
fn a_call_the_ledger_never_saw_start_is_safe_to_resume() {
    assert_eq!(class_of(None), RecoveryClass::Resume);
}

#[test]
fn a_deferred_call_awaits_its_answer_and_is_not_plain_resume() {
    let class = class_of(Some(ToolEffectStatus::Deferred));
    assert_eq!(class, RecoveryClass::AwaitingAnswer);
    assert_ne!(class, RecoveryClass::Resume);
}

#[test]
fn the_most_cautious_row_wins_when_several_match_one_call() {
    let rows = [
        effect("run-1", "c1", ToolEffectStatus::Completed),
        effect("run-1", "c1", ToolEffectStatus::Started),
        effect("run-1", "c1", ToolEffectStatus::Deferred),
    ];
    let classified = classify_recovery(&[call("c1")], &rows);
    assert_eq!(classified[0].class, RecoveryClass::NeedsVerification);
    assert_eq!(classified[0].effect_status, Some(ToolEffectStatus::Started));

    // Order of the rows must not matter.
    let mut reversed = rows.to_vec();
    reversed.reverse();
    assert_eq!(
        classify_recovery(&[call("c1")], &reversed)[0].class,
        RecoveryClass::NeedsVerification
    );
}

#[test]
fn a_settled_call_resumes_report_only() {
    assert_eq!(
        class_of(Some(ToolEffectStatus::Completed)),
        RecoveryClass::ResumeReportOnly
    );
    assert_eq!(
        class_of(Some(ToolEffectStatus::Failed)),
        RecoveryClass::ResumeReportOnly
    );
}

#[test]
fn a_call_that_only_started_needs_verification() {
    assert_eq!(
        class_of(Some(ToolEffectStatus::Started)),
        RecoveryClass::NeedsVerification
    );
    assert_eq!(
        class_of(Some(ToolEffectStatus::Interrupted)),
        RecoveryClass::NeedsVerification
    );
}

#[test]
fn effects_match_on_run_and_call_id_together() {
    let effects = [
        effect("other-run", "c1", ToolEffectStatus::Started),
        effect("run-1", "c2", ToolEffectStatus::Started),
    ];
    let classified = classify_recovery(&[call("c1")], &effects);
    assert_eq!(
        classified[0].class,
        RecoveryClass::Resume,
        "no row for (run-1, c1)"
    );
    assert_eq!(classified[0].effect_status, None);
}

#[test]
fn each_call_is_classified_independently_in_tail_order() {
    let effects = [
        effect("run-1", "done", ToolEffectStatus::Completed),
        effect("run-1", "inflight", ToolEffectStatus::Started),
    ];
    let tail = [call("inflight"), call("done"), call("fresh")];
    let classified = classify_recovery(&tail, &effects);

    let summary: Vec<_> = classified
        .iter()
        .map(|c| (c.call.call_id.as_str(), c.class))
        .collect();
    assert_eq!(
        summary,
        [
            ("inflight", RecoveryClass::NeedsVerification),
            ("done", RecoveryClass::ResumeReportOnly),
            ("fresh", RecoveryClass::Resume),
        ]
    );
    assert_eq!(classified[0].effect_status, Some(ToolEffectStatus::Started));
    assert!(!classified[0].reason.is_empty());
}

#[test]
fn overall_recovery_is_the_most_cautious_class() {
    let effects = [
        effect("run-1", "a", ToolEffectStatus::Completed),
        effect("run-1", "b", ToolEffectStatus::Started),
    ];
    let all = classify_recovery(&[call("a"), call("b"), call("c")], &effects);
    assert_eq!(overall_recovery(&all), RecoveryClass::NeedsVerification);
    assert!(RecoveryClass::Resume < RecoveryClass::ResumeReportOnly);
    assert!(RecoveryClass::ResumeReportOnly < RecoveryClass::AwaitingAnswer);
    assert!(RecoveryClass::AwaitingAnswer < RecoveryClass::NeedsVerification);
    let deferred = [effect("run-1", "d", ToolEffectStatus::Deferred)];
    let awaiting = classify_recovery(&[call("d"), call("c")], &deferred);
    assert_eq!(overall_recovery(&awaiting), RecoveryClass::AwaitingAnswer);
    assert_eq!(overall_recovery(&all[..1]), RecoveryClass::ResumeReportOnly);
    assert_eq!(overall_recovery(&all[2..]), RecoveryClass::Resume);
    assert_eq!(
        overall_recovery(&[]),
        RecoveryClass::Resume,
        "nothing dangling"
    );
}

#[test]
fn no_dangling_calls_classify_to_nothing() {
    assert!(classify_recovery(&[], &[effect("run-1", "x", ToolEffectStatus::Started)]).is_empty());
}

#[test]
fn a_missing_row_needs_verification_when_started_write_failures_may_continue() {
    let classified =
        classify_recovery_with(&[call("c1")], &[], MissingEffectRow::Uncertain);
    assert_eq!(classified[0].class, RecoveryClass::NeedsVerification);
    assert_eq!(classified[0].effect_status, None);
    // The default reading is unchanged.
    assert_eq!(classify_recovery(&[call("c1")], &[])[0].class, RecoveryClass::Resume);
}

#[test]
fn equal_class_rows_pick_the_same_status_in_either_order() {
    let rows = [
        effect("run-1", "c1", ToolEffectStatus::Completed),
        effect("run-1", "c1", ToolEffectStatus::Failed),
    ];
    let forward = classify_recovery(&[call("c1")], &rows);
    let mut reversed = rows.to_vec();
    reversed.reverse();
    let backward = classify_recovery(&[call("c1")], &reversed);
    assert_eq!(forward[0].class, RecoveryClass::ResumeReportOnly);
    assert_eq!(forward[0].effect_status, backward[0].effect_status);
    assert_eq!(forward[0].reason, backward[0].reason);
}
