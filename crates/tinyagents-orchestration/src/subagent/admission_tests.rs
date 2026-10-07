use super::*;
use std::sync::{Arc, Barrier};

fn admission(
    parent: Option<usize>,
    total: Option<usize>,
    targets: Option<&[&str]>,
) -> SpawnAdmission {
    SpawnAdmission::new(SpawnPolicy {
        max_children_per_parent: parent,
        max_total_per_root: total,
        allowed_targets: targets.map(|t| t.iter().map(|s| (*s).to_owned()).collect()),
    })
}

/// A parent config with no thread: its scope is its run id.
fn cfg(run_id: &str) -> RunConfig {
    RunConfig::new(run_id)
}

/// A parent config on a conversation thread: its scope is the thread.
fn turn(run_id: &str, thread: &str) -> RunConfig {
    RunConfig::new(run_id).with_thread(thread)
}

#[test]
fn default_policy_is_unlimited() {
    let policy = SpawnPolicy::default();
    assert_eq!(policy.max_children_per_parent, None);
    assert_eq!(policy.max_total_per_root, None);
    assert_eq!(policy.allowed_targets, None);

    let admission = SpawnAdmission::default();
    let held: Vec<_> = (0..200)
        .map(|_| admission.try_reserve(&cfg("p"), "worker").unwrap())
        .collect();
    assert_eq!(held.len(), 200);
}

#[test]
fn per_scope_cap_rejects_then_admits_after_release() {
    let admission = admission(Some(2), None, None);
    let first = admission.try_reserve(&cfg("p"), "w").unwrap();
    let _second = admission.try_reserve(&cfg("p"), "w").unwrap();
    assert_eq!(
        admission.try_reserve(&cfg("p"), "w").unwrap_err(),
        SpawnRejection::MaxChildrenPerParent { active: 2, max: 2 }
    );
    assert_eq!(admission.active_children("run:p"), 2);

    drop(first);
    assert_eq!(admission.active_children("run:p"), 1);
    drop(admission.try_reserve(&cfg("p"), "w").unwrap());
}

#[test]
fn cap_is_scoped_to_each_parent_without_a_thread() {
    let admission = admission(Some(1), None, None);
    let _a = admission.try_reserve(&cfg("parent-a"), "w").unwrap();
    let _b = admission.try_reserve(&cfg("parent-b"), "w").unwrap();
    assert!(admission.try_reserve(&cfg("parent-a"), "w").is_err());
}

#[test]
fn turns_with_different_run_ids_on_one_thread_share_the_cap() {
    let admission = admission(Some(1), Some(2), None);
    assert_eq!(
        admission.scope_of(&turn("run-1", "thread-A")),
        "thread:thread-A",
        "the thread, not the run id, is the scope"
    );
    let mut first = admission
        .try_reserve(&turn("run-1", "thread-A"), "w")
        .unwrap();
    first.commit();

    // A later turn mints a new run id, but the first child is still alive.
    assert!(
        admission
            .try_reserve(&turn("run-2", "thread-A"), "w")
            .is_err()
    );
    // Another conversation is independent.
    drop(
        admission
            .try_reserve(&turn("run-3", "thread-B"), "w")
            .unwrap(),
    );

    // Reused run ids on different threads do not collide either.
    drop(
        admission
            .try_reserve(&turn("run-1", "thread-C"), "w")
            .unwrap(),
    );
    drop(first);
}

#[test]
fn total_budget_survives_run_id_churn_on_a_thread() {
    let admission = admission(None, Some(2), None);
    for n in 0..2 {
        let mut r = admission
            .try_reserve(&turn(&format!("run-{n}"), "thread-A"), "w")
            .unwrap();
        r.commit();
    }
    assert_eq!(admission.spawned_in_scope("thread:thread-A"), 2);
    assert_eq!(
        admission
            .try_reserve(&turn("run-9", "thread-A"), "w")
            .unwrap_err(),
        SpawnRejection::MaxTotalPerRoot { spawned: 2, max: 2 }
    );
}

#[test]
fn a_host_scope_resolver_overrides_the_default_rule() {
    let admission = admission(Some(1), None, None).with_scope_key(|_| "tenant".to_owned());
    assert_eq!(admission.scope_of(&cfg("anything")), "tenant");
    let _held = admission.try_reserve(&turn("r1", "thread-A"), "w").unwrap();
    assert!(admission.try_reserve(&turn("r2", "thread-B"), "w").is_err());
    assert_eq!(admission.active_children("tenant"), 1);
}

#[test]
fn total_per_scope_counts_spawned_children_even_after_they_finish() {
    let admission = admission(None, Some(2), None);
    for _ in 0..2 {
        let mut reservation = admission.try_reserve(&cfg("root"), "w").unwrap();
        reservation.commit();
        drop(reservation); // child reached a terminal state
    }
    assert_eq!(admission.spawned_in_scope("run:root"), 2);
    assert_eq!(
        admission.try_reserve(&cfg("root"), "w").unwrap_err(),
        SpawnRejection::MaxTotalPerRoot { spawned: 2, max: 2 }
    );
    // A different scope has its own budget.
    drop(admission.try_reserve(&cfg("other-root"), "w").unwrap());
}

#[test]
fn uncommitted_reservation_refunds_both_counters() {
    let admission = admission(Some(1), Some(1), None);
    let reservation = admission.try_reserve(&cfg("p"), "w").unwrap();
    drop(reservation); // spawn failed before the child started
    assert_eq!(admission.active_children("run:p"), 0);
    assert_eq!(admission.spawned_in_scope("run:p"), 0);
    drop(admission.try_reserve(&cfg("p"), "w").unwrap());
}

#[test]
fn continuation_takes_an_active_slot_but_no_total_budget() {
    let admission = admission(Some(1), Some(1), None);
    let mut first = admission.try_reserve(&cfg("p"), "w").unwrap();
    first.commit();
    drop(first);
    assert!(admission.try_reserve(&cfg("p"), "w").is_err());

    let mut resumed = admission.try_reserve_continuation(&cfg("p"), "w").unwrap();
    resumed.commit();
    assert_eq!(admission.active_children("run:p"), 1);
    assert!(admission.try_reserve_continuation(&cfg("p"), "w").is_err());
    drop(resumed);
    assert_eq!(admission.spawned_in_scope("run:p"), 1);
}

#[test]
fn continuation_resolves_the_same_scope_as_a_fresh_spawn() {
    let admission = admission(Some(1), None, None);
    let _fresh = admission
        .try_reserve(&turn("run-1", "thread-A"), "w")
        .unwrap();
    assert!(
        admission
            .try_reserve_continuation(&turn("run-2", "thread-A"), "w")
            .is_err()
    );
}

#[test]
fn allowed_targets_gate_the_target_name() {
    let admission = admission(None, None, Some(&["researcher"]));
    drop(admission.try_reserve(&cfg("p"), "researcher").unwrap());
    assert_eq!(
        admission.try_reserve(&cfg("p"), "coder").unwrap_err(),
        SpawnRejection::TargetNotAllowed {
            target: "coder".into()
        }
    );
    // An explicitly empty allowlist admits nothing.
    let closed = self::admission(None, None, Some(&[]));
    assert!(closed.try_reserve(&cfg("p"), "researcher").is_err());
}

#[test]
fn rejected_target_does_not_consume_a_slot() {
    let admission = admission(Some(1), Some(1), Some(&["w"]));
    assert!(admission.try_reserve(&cfg("p"), "nope").is_err());
    assert_eq!(admission.active_children("run:p"), 0);
    assert_eq!(admission.spawned_in_scope("run:p"), 0);
    drop(admission.try_reserve(&cfg("p"), "w").unwrap());
}

#[test]
fn clones_share_one_ledger() {
    let admission = admission(Some(1), None, None);
    let clone = admission.clone();
    let _held = admission.try_reserve(&cfg("p"), "w").unwrap();
    assert!(clone.try_reserve(&cfg("p"), "w").is_err());
}

#[test]
fn concurrent_reservations_cannot_race_past_the_cap() {
    const THREADS: usize = 64;
    const CAP: usize = 5;
    let admission = admission(Some(CAP), Some(CAP), None);
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let admission = admission.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                admission.try_reserve(&cfg("p"), "w").ok()
            })
        })
        .collect();
    let admitted: Vec<_> = handles
        .into_iter()
        .filter_map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(admitted.len(), CAP);
    assert_eq!(admission.active_children("run:p"), CAP);
}

#[test]
fn rejection_messages_name_the_limit() {
    let parent = SpawnRejection::MaxChildrenPerParent { active: 5, max: 5 }.to_string();
    assert!(parent.contains("5/5"), "{parent}");
    let root = SpawnRejection::MaxTotalPerRoot { spawned: 9, max: 9 }.to_string();
    assert!(root.contains("9/9"), "{root}");
    let target = SpawnRejection::TargetNotAllowed { target: "x".into() }.to_string();
    assert!(target.contains("`x`"), "{target}");
    let unnamed = SpawnRejection::TargetNotAllowed {
        target: String::new(),
    }
    .to_string();
    assert!(unnamed.contains("named none"), "{unnamed}");
}

#[test]
fn a_thread_id_and_a_run_id_with_the_same_text_do_not_share_a_scope() {
    let admission = admission(Some(1), Some(1), None);
    let mut threaded = admission.try_reserve(&turn("r", "x"), "w").unwrap();
    threaded.commit();
    // An unthreaded parent whose run id is "x" is a different scope.
    assert_ne!(
        admission.scope_of(&turn("r", "x")),
        admission.scope_of(&cfg("x"))
    );
    let _unthreaded = admission.try_reserve(&cfg("x"), "w").unwrap();
    assert_eq!(admission.active_children("thread:x"), 1);
    assert_eq!(admission.active_children("run:x"), 1);
}
