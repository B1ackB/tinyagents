//! A provider error sent *inside* an HTTP 200 stream, through the harness.
//!
//! OpenRouter-style providers report a request error as an SSE event carrying
//! the HTTP status as a numeric `code` (`{"error":{"code":400,…}}`) while the
//! response itself is a 200. A 400 is deterministic: retrying it re-sends the
//! same bad request. These tests stand up a loopback server that answers every
//! request with that stream and count how many requests the harness makes
//! (openhuman#6724).
//!
//! The server is hand-rolled on `std::net` for the same reason as
//! `provider_local_wire.rs`: the dev-dependency `tokio` has no `net` feature.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::openai::OpenAiModel;

/// Answers every request with `sse_body` as a 200 event stream and counts
/// the requests.
struct StreamErrorServer {
    base_url: String,
    requests: Arc<AtomicUsize>,
}

impl StreamErrorServer {
    fn start(sse_body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
        let port = listener.local_addr().expect("local addr").port();
        let requests = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                serve(stream, sse_body, &counter);
            }
        });
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

/// Reads one request (headers and body), counts it, then writes the stream.
/// Counting happens before the reply so the client cannot observe the reply
/// before the count.
fn serve(mut stream: TcpStream, sse_body: &str, counter: &AtomicUsize) {
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(value) = line
            .to_ascii_lowercase()
            .strip_prefix("content-length:")
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            content_length = value;
        }
    }
    let mut body = vec![0u8; content_length];
    let _ = reader.read_exact(&mut body);
    counter.fetch_add(1, Ordering::SeqCst);

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        sse_body.len(),
        sse_body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

async fn streamed_run_against(server: &StreamErrorServer) -> tinyagents_harness::Result<()> {
    let model = OpenAiModel::new("test-key")
        .with_base_url(&server.base_url)
        .with_model("test-model");
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("remote", Arc::new(model))
        .set_default_model("remote");
    harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .map(|_| ())
}

#[tokio::test]
async fn a_streamed_numeric_400_error_fails_on_the_first_attempt() {
    let server = StreamErrorServer::start(
        "data: {\"error\":{\"code\":400,\"message\":\"Provider returned error\"}}\n\ndata: [DONE]\n\n",
    );

    let result = streamed_run_against(&server).await;

    assert!(result.is_err(), "a 400 must fail the run");
    assert_eq!(
        server.requests(),
        1,
        "a deterministic 400 is not retried: exactly one provider request"
    );
}

#[tokio::test]
async fn a_streamed_numeric_503_error_is_still_retried() {
    // Control: the status is read as a status, so a transient 5xx keeps its
    // retries. Without this, "one request" above could also mean retries were
    // switched off altogether.
    let server = StreamErrorServer::start(
        "data: {\"error\":{\"code\":503,\"message\":\"Provider returned error\"}}\n\ndata: [DONE]\n\n",
    );

    let result = streamed_run_against(&server).await;

    assert!(result.is_err(), "the 503 persists, so the run still fails");
    assert!(
        server.requests() > 1,
        "a transient 503 is retried; saw {} request(s)",
        server.requests()
    );
}
