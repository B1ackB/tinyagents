//! Tests for the per-call progress gate: ordering, the late-update guard,
//! coalescing under a flood, and the hand-off to `ToolExecutionContext`.

use std::time::Duration;

use serde_json::json;
use tinytools::{ToolProgress, ToolRunContext};

use super::{ToolProgressGate, ToolProgressLimits};
use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::testkit::EventRecorder;
use crate::tool::ToolExecutionContext;

fn gate(recorder: &EventRecorder, limits: ToolProgressLimits) -> std::sync::Arc<ToolProgressGate> {
    ToolProgressGate::new(CallId::new("call-1"), "build", recorder.sink(), limits)
}

fn unlimited() -> ToolProgressLimits {
    ToolProgressLimits {
        max_per_window: usize::MAX,
        window: Duration::from_secs(3600),
    }
}

fn messages(recorder: &EventRecorder) -> Vec<String> {
    recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::ToolProgress { message, .. } => Some(message),
            _ => None,
        })
        .collect()
}

#[test]
fn updates_become_events_in_order_and_are_queued_for_middleware() {
    let recorder = EventRecorder::new();
    let gate = gate(&recorder, unlimited());
    let sink = gate.sink();
    sink.report(ToolProgress::message("one"));
    sink.report(ToolProgress::message("two").with_fraction(0.5));
    sink.report(ToolProgress::default().with_partial(json!({"rows": 3})));
    gate.close();

    assert_eq!(messages(&recorder), vec!["one", "two", ""]);
    let events = recorder.events();
    assert!(matches!(
        &events[1],
        AgentEvent::ToolProgress { call_id, fraction: Some(f), partial: None, .. }
            if call_id == &CallId::new("call-1") && (*f - 0.5).abs() < f32::EPSILON
    ));
    assert!(matches!(
        &events[2],
        AgentEvent::ToolProgress { partial: Some(p), .. } if p == &json!({"rows": 3})
    ));

    let deltas = gate.take_pending();
    let contents: Vec<_> = deltas.iter().map(|d| d.content.as_str()).collect();
    assert_eq!(contents, vec!["one", "two", r#"{"rows":3}"#]);
    assert!(deltas.iter().all(|d| d.call_id == "call-1"));
    assert!(
        deltas
            .iter()
            .all(|d| d.tool_name.as_deref() == Some("build"))
    );
    assert!(gate.take_pending().is_empty(), "pending is drained once");
}

#[test]
fn an_update_after_close_is_dropped_not_emitted() {
    let recorder = EventRecorder::new();
    let gate = gate(&recorder, unlimited());
    let sink = gate.sink();
    sink.report(ToolProgress::message("before"));
    gate.close();
    sink.report(ToolProgress::message("late"));

    assert_eq!(messages(&recorder), vec!["before"]);
    assert_eq!(gate.take_pending().len(), 1);
}

#[test]
fn an_empty_update_is_ignored() {
    let recorder = EventRecorder::new();
    let gate = gate(&recorder, unlimited());
    gate.sink().report(ToolProgress::default());
    gate.close();
    assert!(messages(&recorder).is_empty());
}

#[test]
fn a_flood_is_coalesced_and_the_latest_state_survives_close() {
    let recorder = EventRecorder::new();
    let gate = gate(
        &recorder,
        ToolProgressLimits {
            max_per_window: 2,
            window: Duration::from_secs(3600),
        },
    );
    let sink = gate.sink();
    for n in 1..=6 {
        sink.report(ToolProgress::message(format!("step {n}")));
    }
    // Two pass straight through; the rest collapse into the newest one.
    assert_eq!(messages(&recorder), vec!["step 1", "step 2"]);
    gate.close();
    assert_eq!(messages(&recorder), vec!["step 1", "step 2", "step 6"]);
    assert_eq!(gate.take_pending().len(), 3);
}

#[test]
fn coalescing_merges_so_a_later_fraction_does_not_erase_an_earlier_message() {
    let recorder = EventRecorder::new();
    let gate = gate(
        &recorder,
        ToolProgressLimits {
            max_per_window: 0,
            window: Duration::from_secs(3600),
        },
    );
    let sink = gate.sink();
    sink.report(ToolProgress::message("compiling"));
    sink.report(ToolProgress::default().with_fraction(0.9));
    gate.close();
    let events = recorder.events();
    assert!(matches!(
        &events[0],
        AgentEvent::ToolProgress { message, fraction: Some(f), .. }
            if message == "compiling" && (*f - 0.9).abs() < f32::EPSILON
    ));
}

#[test]
fn a_new_window_flushes_what_the_last_one_held_back() {
    let recorder = EventRecorder::new();
    let gate = gate(
        &recorder,
        ToolProgressLimits {
            max_per_window: 1,
            window: Duration::from_millis(1),
        },
    );
    let sink = gate.sink();
    sink.report(ToolProgress::message("a"));
    sink.report(ToolProgress::message("b")); // held back
    std::thread::sleep(Duration::from_millis(20));
    sink.report(ToolProgress::message("c")); // new window: flush "b", pass "c"
    gate.close();
    assert_eq!(messages(&recorder), vec!["a", "b", "c"]);
}

#[tokio::test]
async fn the_execution_context_reports_through_the_scoped_gate() {
    let recorder = EventRecorder::new();
    let gate = gate(&recorder, unlimited());
    let config = RunConfig::new("run-1");
    let run: RunContext = RunContext::new(config, ());

    let inside = gate
        .scope(async {
            let ctx = ToolExecutionContext::from_run_context(&run, CallId::new("call-1"));
            ctx.report_progress(ToolProgress::message("scoped"));
            // A different call id must not borrow this call's sink.
            let other = ToolExecutionContext::from_run_context(&run, CallId::new("call-2"));
            other.report_progress(ToolProgress::message("wrong call"));
            ctx
        })
        .await;
    // The context outlives the scope (a tool may stash it) but stays bound to
    // the gate, so closing the gate still silences it.
    gate.close();
    inside.report_progress(ToolProgress::message("after close"));

    assert_eq!(messages(&recorder), vec!["scoped"]);

    // Outside any scope there is no sink, and reporting is a no-op.
    let bare = ToolExecutionContext::from_run_context(&run, CallId::new("call-1"));
    bare.report_progress(ToolProgress::message("nobody"));
    assert_eq!(messages(&recorder), vec!["scoped"]);
}
