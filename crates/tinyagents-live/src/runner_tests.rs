use super::*;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::context::RunConfig;
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::testkit::FakeTool;
use tinyagents_harness::tinyinference_llm::tool::ToolCall as HarnessToolCall;
use tinyagents_harness::TinyAgentsError;
use tinyliveagents::{
    AudioFormat, Capabilities, ClientCommand, CloseReason, LiveSession, SessionInfo, ToolCall,
    ToolResult,
};

/// A provider whose session ends are handed to the test.
#[derive(Debug)]
struct ScriptedProvider {
    ends: Mutex<Option<(mpsc::Receiver<ClientCommand>, mpsc::Sender<LiveEvent>)>>,
    seen: Mutex<Option<LiveConfig>>,
    session: Mutex<Option<LiveSession>>,
}

impl ScriptedProvider {
    fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let (ev_tx, ev_rx) = mpsc::channel(64);
        Self {
            ends: Mutex::new(Some((cmd_rx, ev_tx))),
            seen: Mutex::new(None),
            session: Mutex::new(Some(LiveSession::from_channels(cmd_tx, ev_rx))),
        }
    }

    fn take_ends(&self) -> (mpsc::Receiver<ClientCommand>, mpsc::Sender<LiveEvent>) {
        self.ends.lock().unwrap().take().unwrap()
    }
}

#[async_trait]
impl LiveProvider for ScriptedProvider {
    fn id(&self) -> &'static str {
        "scripted"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_audio: true,
            tools: true,
            server_vad: true,
            manual_activity: false,
            text_input: true,
            interruptions: true,
            resumption: false,
            input_sample_rates: &[16_000],
        }
    }

    async fn connect(&self, config: LiveConfig) -> tinyliveagents::Result<LiveSession> {
        *self.seen.lock().unwrap() = Some(config);
        self.session
            .lock()
            .unwrap()
            .take()
            .ok_or(tinyliveagents::Error::Closed)
    }
}

/// A tool that sleeps before answering.
struct SlowTool {
    delay: Duration,
}

#[async_trait]
impl tinyagents_harness::tinytools::Tool for SlowTool {
    fn name(&self) -> &str {
        "slow"
    }
    fn description(&self) -> &str {
        "sleeps"
    }
    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _args: Value) -> anyhow::Result<tinyagents_harness::tinytools::ToolResult> {
        tokio::time::sleep(self.delay).await;
        Ok(tinyagents_harness::tinytools::ToolResult::success("slept"))
    }
}

/// Refuses one tool in `before_tool`, the way a policy middleware would.
struct Refuse {
    tool: &'static str,
    approval: bool,
}

#[async_trait]
impl Middleware<()> for Refuse {
    fn name(&self) -> &str {
        "refuse"
    }
    async fn before_tool(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        call: &mut HarnessToolCall,
    ) -> tinyagents_harness::Result<()> {
        if call.name != self.tool {
            return Ok(());
        }
        if self.approval {
            Err(TinyAgentsError::ApprovalRequired {
                metadata: json!({ "reason": "test" }),
            })
        } else {
            Err(TinyAgentsError::ToolFailed("blocked by policy".into()))
        }
    }
}

fn harness() -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(FakeTool::returning("get_time", "noon")));
    harness.register_tool(Arc::new(FakeTool::failing("broken", "kaput")));
    harness.register_tool(Arc::new(FakeTool::returning("blocked", "never")));
    harness.register_tool(Arc::new(FakeTool::returning("gated", "never")));
    harness.register_tool(Arc::new(SlowTool {
        delay: Duration::from_millis(300),
    }));
    harness.push_middleware(Arc::new(Refuse {
        tool: "blocked",
        approval: false,
    }));
    harness.push_middleware(Arc::new(Refuse {
        tool: "gated",
        approval: true,
    }));
    harness
}

fn call(id: &str, name: &str) -> LiveEvent {
    LiveEvent::ToolCall(ToolCall {
        call_id: id.into(),
        name: name.into(),
        args: Value::Null,
    })
}

fn ready() -> LiveEvent {
    LiveEvent::Ready(SessionInfo {
        provider: "scripted".into(),
        session_id: None,
        model: None,
        input_format: AudioFormat::default(),
        output_format: AudioFormat::pcm16(24_000),
    })
}

async fn start(
    provider: &ScriptedProvider,
    options: LiveAgentOptions,
) -> LiveAgentSession {
    let agent = LiveAgent::new(Arc::new(harness()), Arc::new(()));
    assert!(format!("{agent:?}").contains("LiveAgent"));
    assert!(agent.harness().tools().schemas().len() >= 5);
    let ctx = RunContext::new(RunConfig::new("live-test"), ());
    let config = LiveConfig::new().with_tool(tinyliveagents::ToolDeclaration::new(
        "get_time",
        "host override",
        json!({}),
    ));
    agent.start(provider, config, ctx, options).await.unwrap()
}

async fn next_result(commands: &mut mpsc::Receiver<ClientCommand>) -> ToolResult {
    loop {
        let command = tokio::time::timeout(Duration::from_secs(10), commands.recv())
            .await
            .expect("timed out waiting for a tool result")
            .expect("session ended");
        if let ClientCommand::ToolResult(result) = command {
            return result;
        }
    }
}

async fn next_finished(session: &mut LiveAgentSession) -> (String, bool, bool) {
    loop {
        let event = tokio::time::timeout(Duration::from_secs(10), session.recv())
            .await
            .expect("timed out")
            .expect("session ended");
        if let LiveAgentEvent::ToolFinished {
            call_id,
            is_error,
            cancelled,
            ..
        } = event
        {
            return (call_id, is_error, cancelled);
        }
    }
}

#[tokio::test]
async fn declares_tools_and_answers_calls_through_the_harness() {
    let provider = ScriptedProvider::new();
    let (mut commands, events) = provider.take_ends();
    let mut session = start(&provider, LiveAgentOptions::default()).await;
    assert!(format!("{session:?}").contains("LiveAgentSession"));

    let declared = provider.seen.lock().unwrap().clone().unwrap().tools;
    let get_time: Vec<_> = declared.iter().filter(|t| t.name == "get_time").collect();
    assert_eq!(get_time.len(), 1, "host declarations win over duplicates");
    assert_eq!(get_time[0].description, "host override");
    assert!(declared.iter().any(|t| t.name == "slow"));

    events.send(ready()).await.unwrap();
    assert!(matches!(
        session.recv().await,
        Some(LiveAgentEvent::Live(LiveEvent::Ready(_)))
    ));
    events.send(call("c1", "get_time")).await.unwrap();
    assert!(matches!(
        session.recv().await,
        Some(LiveAgentEvent::Live(LiveEvent::ToolCall(_)))
    ));
    assert_eq!(
        session.recv().await,
        Some(LiveAgentEvent::ToolStarted {
            call_id: "c1".into(),
            name: "get_time".into()
        })
    );
    let result = next_result(&mut commands).await;
    assert_eq!(result.call_id, "c1");
    assert_eq!(result.output, json!("noon"));
    assert!(!result.is_error);
    assert_eq!(next_finished(&mut session).await, ("c1".into(), false, false));

    // Commands the host sends pass straight through.
    session.sender().send_text("hi").await.unwrap();
    assert_eq!(
        commands.recv().await,
        Some(ClientCommand::Text("hi".into()))
    );

    events
        .send(LiveEvent::Closed(CloseReason::Client))
        .await
        .unwrap();
    assert!(matches!(
        session.recv().await,
        Some(LiveAgentEvent::Live(LiveEvent::Closed(_)))
    ));
}

#[tokio::test]
async fn failures_denials_approvals_and_unknown_tools_become_error_results() {
    let provider = ScriptedProvider::new();
    let (mut commands, events) = provider.take_ends();
    let mut session = start(&provider, LiveAgentOptions::default()).await;
    for (id, name) in [
        ("f", "broken"),
        ("d", "blocked"),
        ("a", "gated"),
        ("u", "no_such_tool"),
    ] {
        events.send(call(id, name)).await.unwrap();
        let result = next_result(&mut commands).await;
        assert_eq!(result.call_id, id);
        assert!(result.is_error || name == "no_such_tool", "{name}: {result:?}");
        let (finished, _, cancelled) = next_finished(&mut session).await;
        assert_eq!(finished, id);
        assert!(!cancelled);
        if name == "gated" {
            assert!(result.output_text().contains("approval"), "{result:?}");
        }
        if name == "blocked" {
            assert!(result.output_text().contains("blocked by policy"), "{result:?}");
        }
    }
}

#[tokio::test]
async fn cancelled_calls_are_skipped_or_not_answered() {
    let provider = ScriptedProvider::new();
    let (mut commands, events) = provider.take_ends();
    let mut session = start(&provider, LiveAgentOptions::default()).await;
    // `s1` starts running; `s2` waits behind it and is cancelled first.
    events.send(call("s1", "slow")).await.unwrap();
    events.send(call("s2", "get_time")).await.unwrap();
    loop {
        if let Some(LiveAgentEvent::ToolStarted { call_id, .. }) = session.recv().await {
            assert_eq!(call_id, "s1");
            break;
        }
    }
    events
        .send(LiveEvent::ToolCallCancelled {
            call_ids: vec!["s1".into(), "s2".into()],
        })
        .await
        .unwrap();
    let (finished, _, cancelled) = next_finished(&mut session).await;
    assert_eq!(finished, "s1");
    assert!(cancelled);
    // Neither result reaches the provider.
    events.send(call("s3", "get_time")).await.unwrap();
    let result = next_result(&mut commands).await;
    assert_eq!(result.call_id, "s3");
}

#[tokio::test]
async fn slow_tools_time_out() {
    let provider = ScriptedProvider::new();
    let (mut commands, events) = provider.take_ends();
    let _session = start(
        &provider,
        LiveAgentOptions {
            tool_timeout: Duration::from_millis(20),
            include_deferred_tools: true,
        },
    )
    .await;
    events.send(call("t", "slow")).await.unwrap();
    let result = next_result(&mut commands).await;
    assert!(result.is_error);
    assert!(result.output_text().contains("did not finish"));
}

#[tokio::test]
async fn connection_errors_are_returned() {
    let provider = ScriptedProvider::new();
    let _ = provider.session.lock().unwrap().take();
    let agent = LiveAgent::new(Arc::new(harness()), Arc::new(()));
    let ctx = RunContext::new(RunConfig::new("live-test"), ());
    let result = agent
        .start(&provider, LiveConfig::new(), ctx, LiveAgentOptions::default())
        .await;
    assert!(matches!(result, Err(tinyliveagents::Error::Closed)));
}

#[tokio::test]
async fn dropping_the_session_stops_it() {
    let provider = ScriptedProvider::new();
    let (_commands, events) = provider.take_ends();
    let session = start(&provider, LiveAgentOptions::default()).await;
    drop(session);
    tokio::time::sleep(Duration::from_millis(20)).await;
    // The driver is gone, so the provider's event channel has no reader.
    assert!(events.send(ready()).await.is_err());
}
