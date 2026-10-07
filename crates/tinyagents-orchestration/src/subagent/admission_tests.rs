use super::*;
use std::sync::{Arc, Barrier};

fn admission(parent: Option<usize>, root: Option<usize>, targets: Option<&[&str]>) -> SpawnAdmission {
    SpawnAdmission::new(SpawnPolicy {
        max_children_per_parent: parent,
        max_total_per_root: root,
        allowed_targets: targets.map(|t| t.iter().map(|s| (*s).to_owned()).collect()),
    })
}

#[test]
fn default_policy_is_unlimited() {
    let policy = SpawnPolicy::default();
    assert_eq!(policy.max_children_per_parent, None);
    assert_eq!(policy.max_total_per_root, None);
    assert_eq!(policy.allowed_targets, None);

    let admission = SpawnAdmission::default();
    let held: Vec<_> = (0..200)
        .map(|_| admission.try_reserve("root", "parent", "worker").unwrap())
        .collect();
    assert_eq!(held.len(), 200);
}

#[test]
fn per_parent_cap_rejects_then_admits_after_release() {
    let admission = admission(Some(2), None, None);
    let first = admission.try_reserve("r", "p", "w").unwrap();
    let _second = admission.try_reserve("r", "p", "w").unwrap();
    let rejected = admission.try_reserve("r", "p", "w").unwrap_err();
    assert_eq!(
        rejected,
        SpawnRejection::MaxChildrenPerParent { active: 2, max: 2 }
    );
    assert_eq!(admission.active_children("p"), 2);

    drop(first);
    assert_eq!(admission.active_children("p"), 1);
    drop(admission.try_reserve("r", "p", "w").unwrap());
}

#[test]
fn per_parent_cap_is_scoped_to_each_parent() {
    let admission = admission(Some(1), None, None);
    let _a = admission.try_reserve("r", "parent-a", "w").unwrap();
    let _b = admission.try_reserve("r", "parent-b", "w").unwrap();
    assert!(admission.try_reserve("r", "parent-a", "w").is_err());
}

#[test]
fn total_per_root_counts_spawned_children_even_after_they_finish() {
    let admission = admission(None, Some(2), None);
    for _ in 0..2 {
        let mut reservation = admission.try_reserve("root", "p", "w").unwrap();
        reservation.commit();
        drop(reservation); // child reached a terminal state
    }
    assert_eq!(admission.spawned_in_root("root"), 2);
    assert_eq!(
        admission.try_reserve("root", "p", "w").unwrap_err(),
        SpawnRejection::MaxTotalPerRoot { spawned: 2, max: 2 }
    );
    // A different root has its own budget.
    drop(admission.try_reserve("other-root", "p", "w").unwrap());
}

#[test]
fn uncommitted_reservation_refunds_both_counters() {
    let admission = admission(Some(1), Some(1), None);
    let reservation = admission.try_reserve("root", "p", "w").unwrap();
    drop(reservation); // spawn failed before the child started
    assert_eq!(admission.active_children("p"), 0);
    assert_eq!(admission.spawned_in_root("root"), 0);
    drop(admission.try_reserve("root", "p", "w").unwrap());
}

#[test]
fn continuation_reservation_takes_an_active_slot_but_no_total_budget() {
    let admission = admission(Some(1), Some(1), None);
    let mut first = admission.try_reserve("root", "p", "w").unwrap();
    first.commit();
    drop(first);
    assert!(admission.try_reserve("root", "p", "w").is_err());

    let mut resumed = admission.try_reserve_continuation("root", "p", "w").unwrap();
    resumed.commit();
    assert_eq!(admission.active_children("p"), 1);
    assert!(admission.try_reserve_continuation("root", "p", "w").is_err());
    drop(resumed);
    assert_eq!(admission.spawned_in_root("root"), 1);
}

#[test]
fn allowed_targets_gate_the_target_name() {
    let admission = admission(None, None, Some(&["researcher"]));
    drop(admission.try_reserve("r", "p", "researcher").unwrap());
    assert_eq!(
        admission.try_reserve("r", "p", "coder").unwrap_err(),
        SpawnRejection::TargetNotAllowed {
            target: "coder".into()
        }
    );
    // An explicitly empty allowlist admits nothing.
    let closed = admission_with_empty_targets();
    assert!(closed.try_reserve("r", "p", "researcher").is_err());
}

fn admission_with_empty_targets() -> SpawnAdmission {
    admission(None, None, Some(&[]))
}

#[test]
fn rejected_target_does_not_consume_a_slot() {
    let admission = admission(Some(1), Some(1), Some(&["w"]));
    assert!(admission.try_reserve("r", "p", "nope").is_err());
    assert_eq!(admission.active_children("p"), 0);
    assert_eq!(admission.spawned_in_root("r"), 0);
    drop(admission.try_reserve("r", "p", "w").unwrap());
}

#[test]
fn clones_share_one_ledger() {
    let admission = admission(Some(1), None, None);
    let clone = admission.clone();
    let _held = admission.try_reserve("r", "p", "w").unwrap();
    assert!(clone.try_reserve("r", "p", "w").is_err());
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
                admission.try_reserve("root", "p", "w").ok()
            })
        })
        .collect();
    let admitted: Vec<_> = handles
        .into_iter()
        .filter_map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(admitted.len(), CAP);
    assert_eq!(admission.active_children("p"), CAP);
}

#[test]
fn rejection_messages_name_the_limit() {
    let parent = SpawnRejection::MaxChildrenPerParent { active: 5, max: 5 }.to_string();
    assert!(parent.contains("5/5"), "{parent}");
    let root = SpawnRejection::MaxTotalPerRoot { spawned: 9, max: 9 }.to_string();
    assert!(root.contains("9/9"), "{root}");
    let target = SpawnRejection::TargetNotAllowed { target: "x".into() }.to_string();
    assert!(target.contains("`x`"), "{target}");
}
