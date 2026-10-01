use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::start_progress_dispatcher;
use crate::host::{ProgressEvent, ProgressSink};
use crate::ids::RunId;

struct BlockingProgressSink {
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Semaphore>,
    events: Mutex<Vec<ProgressEvent>>,
}

#[async_trait]
impl ProgressSink for BlockingProgressSink {
    async fn emit(&self, event: ProgressEvent) {
        self.entered.notify_one();
        self.release
            .acquire()
            .await
            .expect("test progress sink remains open")
            .forget();
        self.events.lock().expect("progress lock").push(event);
    }
}

#[tokio::test]
async fn terminal_progress_is_delivered_after_nonterminal_slots_are_saturated() {
    let sink = Arc::new(BlockingProgressSink {
        entered: Arc::new(tokio::sync::Notify::new()),
        release: Arc::new(tokio::sync::Semaphore::new(0)),
        events: Mutex::new(Vec::new()),
    });
    let sender = start_progress_dispatcher(Some(sink.clone()))
        .expect("Tokio test runtime provides a dispatcher");
    let run = RunId::new("saturated-progress");
    sender.send_nonterminal(ProgressEvent::Token {
        run: run.clone(),
        text: "first".to_string(),
    });
    sink.entered.notified().await;

    for slot in 0..128 {
        sender.send_nonterminal(ProgressEvent::Token {
            run: run.clone(),
            text: format!("queued-{slot}"),
        });
    }
    sender.send_terminal(ProgressEvent::Finished { run, usage: None });

    // The sink holds the receiver on the first item, so all 128 ordinary
    // slots are occupied. Finished therefore proves the reserved terminal
    // slot was still available after nonterminal backpressure saturated.
    sink.release.add_permits(130);
    for _ in 0..256 {
        if sink
            .events
            .lock()
            .expect("progress lock")
            .iter()
            .any(ProgressEvent::is_terminal)
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        sink.events
            .lock()
            .expect("progress lock")
            .iter()
            .filter(|event| event.is_terminal())
            .count(),
        1,
        "the terminal event cannot be dropped behind 128 progress updates"
    );
}
