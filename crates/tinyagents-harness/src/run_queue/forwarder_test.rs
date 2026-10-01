use std::sync::Mutex;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::runtime::AgentHarness;
use crate::testkit::ScriptedModel;

#[derive(Debug, Clone, PartialEq)]
struct Item {
    id: String,
    text: String,
    thread: String,
}

impl QueuedMessage for Item {
    fn id(&self) -> &str {
        &self.id
    }
    fn text(&self) -> &str {
        &self.text
    }
    fn requeued(id: String, text: String, thread_label: &str, _queued_at_ms: u64) -> Self {
        Self {
            id,
            text,
            thread: thread_label.to_string(),
        }
    }
}

fn item(id: &str, text: &str) -> Item {
    Item {
        id: id.into(),
        text: text.into(),
        thread: "thread-test".into(),
    }
}

fn recorder() -> (ForwardEventSink, Arc<Mutex<Vec<ForwardEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink_events = events.clone();
    let sink: ForwardEventSink = Arc::new(move |event| sink_events.lock().unwrap().push(event));
    (sink, events)
}

#[tokio::test]
async fn residual_collect_requeues_to_its_original_lane() {
    let queue = Arc::new(RunQueue::<Item>::new());
    let handle = SteeringHandle::allow_all();
    let (sink, events) = recorder();
    let guard = SteeringForwarderGuard::new(
        handle.clone(),
        Some(queue.clone()),
        None,
        "thread-test".to_string(),
        sink,
    );
    handle.send(SteeringCommand::InjectMessage(TaMessage::user(format!(
        "{COLLECT_PREFIX}recovered context"
    ))));
    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if queue.status().await.collects == 1 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("residual collect should be requeued before timeout");
    assert!(queue.drain(QueueLane::Steer).await.is_empty());
    let recovered = queue.drain(QueueLane::Collect).await;
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].text, "recovered context");
    assert_eq!(recovered[0].thread, "thread-test");
    let events = events.lock().unwrap();
    assert!(matches!(
        events.as_slice(),
        [ForwardEvent::Requeued { requeued: 1, text: Some(text), .. }] if text == "recovered context"
    ));
}

#[tokio::test]
async fn residual_steer_and_unframed_text_requeue_as_steers_without_double_prefix() {
    let queue = Arc::new(RunQueue::<Item>::new());
    let handle = SteeringHandle::allow_all();
    let (sink, _events) = recorder();
    let guard = SteeringForwarderGuard::new(
        handle.clone(),
        Some(queue.clone()),
        None,
        "t".to_string(),
        sink,
    );
    handle.send(SteeringCommand::InjectMessage(TaMessage::user(format!(
        "{STEER_PREFIX}framed"
    ))));
    handle.send(SteeringCommand::InjectMessage(TaMessage::user("raw")));
    handle.send(SteeringCommand::Pause);
    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if queue.status().await.steers == 2 {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("residual steers should be requeued");
    let texts: Vec<String> = queue
        .drain(QueueLane::Steer)
        .await
        .into_iter()
        .map(|i| i.text)
        .collect();
    assert_eq!(texts, vec!["framed".to_string(), "raw".to_string()]);
}

#[tokio::test]
async fn drop_runs_cleanup_once_and_stops_the_poll_loop() {
    let queue = Arc::new(RunQueue::<Item>::new());
    let handle = SteeringHandle::allow_all();
    let (sink, _events) = recorder();
    let cleaned = Arc::new(Mutex::new(0usize));
    let cleaned_flag = cleaned.clone();
    let guard = SteeringForwarderGuard::new(
        handle.clone(),
        Some(queue.clone()),
        Some(Box::new(move || *cleaned_flag.lock().unwrap() += 1)),
        "t".to_string(),
        sink,
    );
    drop(guard);
    assert_eq!(*cleaned.lock().unwrap(), 1);
    // A push after the drop is never forwarded: the poll task is gone.
    queue.push(QueueLane::Steer, item("late", "late")).await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(queue.status().await.steers, 1);
    assert!(handle.drain().is_empty());
}

#[tokio::test]
async fn steering_only_guard_without_a_queue_still_cleans_up() {
    let handle = SteeringHandle::allow_all();
    let (sink, _events) = recorder();
    let cleaned = Arc::new(Mutex::new(false));
    let flag = cleaned.clone();
    let guard = SteeringForwarderGuard::<Item>::new(
        handle,
        None,
        Some(Box::new(move || *flag.lock().unwrap() = true)),
        "t".to_string(),
        sink,
    );
    drop(guard);
    assert!(*cleaned.lock().unwrap());
}

#[tokio::test]
async fn forward_steers_injects_framed_messages_and_reports_delivery() {
    let queue = RunQueue::<Item>::new();
    queue.push(QueueLane::Steer, item("s1", "do x")).await;
    queue.push(QueueLane::Steer, item("s2", "then y")).await;
    let handle = SteeringHandle::allow_all();
    let (sink, events) = recorder();
    forward_steers(&queue, &handle, "thread-test", &sink).await;
    let injected: Vec<String> = handle
        .drain()
        .into_iter()
        .filter_map(|c| match c {
            SteeringCommand::InjectMessage(m) => Some(m.text().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        injected,
        vec![
            "[User steering message]: do x".to_string(),
            "[User steering message]: then y".to_string()
        ]
    );
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [ForwardEvent::Delivered {
            thread_label: "thread-test".into(),
            mode: "steer",
            delivered: 2,
            item_id: Some("s1".into()),
            text: Some("do x".into()),
        }]
    );
    // An empty lane is silent.
    forward_steers(&queue, &handle, "thread-test", &sink).await;
    assert_eq!(events.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn collect_reaches_the_next_model_boundary_as_additional_context() {
    let queue = Arc::new(RunQueue::<Item>::new());
    queue
        .push(
            QueueLane::Collect,
            item("queued-test", "the deployment finished successfully"),
        )
        .await;
    let handle = SteeringHandle::allow_all();
    let (sink, _events) = recorder();
    forward_collects(&queue, &handle, "thread-test", &sink).await;
    let model = Arc::new(ScriptedModel::replies(vec!["done"]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("scripted", model.clone())
        .set_default_model("scripted");
    harness
        .invoke_in_context(
            &(),
            RunContext::new(RunConfig::new("collect-boundary"), ()).with_steering(handle),
            vec![TaMessage::user("start")],
        )
        .await
        .expect("collect-context run should complete");
    let requests = model.requests();
    assert_eq!(requests.len(), 1, "one model boundary should be crossed");
    let collect = requests[0]
        .messages
        .iter()
        .find(|message| message.text().contains("deployment finished successfully"))
        .expect("the next model request should contain the collected context");
    assert!(matches!(collect, TaMessage::User(_)));
    assert_eq!(
        collect.text(),
        "[Additional context from user]: the deployment finished successfully"
    );
    assert!(!collect.text().starts_with(STEER_PREFIX));
}

#[tokio::test]
async fn requeue_front_keeps_returned_items_ahead_of_newer_ones() {
    let queue = RunQueue::<Item>::new();
    queue.push(QueueLane::Steer, item("newer", "newer")).await;
    queue
        .requeue_front(
            QueueLane::Steer,
            vec![item("old-1", "old-1"), item("old-2", "old-2")],
        )
        .await;
    let order: Vec<String> = queue
        .drain(QueueLane::Steer)
        .await
        .iter()
        .map(|msg| msg.id().to_string())
        .collect();
    assert_eq!(order, ["old-1", "old-2", "newer"]);
}
