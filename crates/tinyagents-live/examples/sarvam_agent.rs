//! A live Sarvam session whose tool calls run through a TinyAgents harness.
//!
//! ```sh
//! SARVAM_API_KEY=... cargo run -p tinyagents-live --example sarvam_agent
//! ```
//!
//! Sends a typed question (no microphone needed), lets the model call the
//! harness's `get_time` tool, and prints the events, including the spoken
//! reply's transcript and how much audio came back.

use std::sync::Arc;
use std::time::Duration;

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::testkit::FakeTool;
use tinyagents_live::tinyliveagents::sarvam::SarvamCascade;
use tinyagents_live::tinyliveagents::{LiveConfig, LiveEvent};
use tinyagents_live::{LiveAgent, LiveAgentEvent, LiveAgentOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("SARVAM_API_KEY").map_err(|_| "set SARVAM_API_KEY")?;
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(FakeTool::returning(
        "get_time",
        r#"{"time":"14:05","timezone":"UTC"}"#,
    )));
    let agent = LiveAgent::new(Arc::new(harness), Arc::new(()));
    let config = LiveConfig::new()
        .with_language("en-IN")
        .with_system_instruction(
            "You are a concise voice assistant. Always call get_time to answer questions about the time. Reply in one sentence.",
        );
    let ctx = RunContext::new(RunConfig::new("sarvam-live-example"), ());
    let provider = SarvamCascade::new(key);
    let mut session = agent
        .start(&provider, config, ctx, LiveAgentOptions::default())
        .await?;
    let sender = session.sender();
    let mut audio_bytes = 0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    while let Ok(Some(event)) = tokio::time::timeout_at(deadline, session.recv()).await {
        match event {
            LiveAgentEvent::Live(LiveEvent::Ready(info)) => {
                println!("ready: {} -> {} Hz", info.provider, info.output_format.sample_rate);
                sender.send_text("What time is it in UTC right now?").await?;
            }
            LiveAgentEvent::Live(LiveEvent::Audio(pcm)) => audio_bytes += pcm.len(),
            LiveAgentEvent::Live(LiveEvent::OutputTranscript { text, is_final: true }) => {
                println!("agent said: {text}");
            }
            LiveAgentEvent::Live(LiveEvent::TurnComplete { .. }) => {
                println!("turn complete; {audio_bytes} bytes of audio");
                break;
            }
            LiveAgentEvent::Live(LiveEvent::Closed(reason)) => {
                println!("closed: {reason:?}");
                break;
            }
            other @ (LiveAgentEvent::ToolStarted { .. } | LiveAgentEvent::ToolFinished { .. }) => {
                println!("{other:?}");
            }
            _ => {}
        }
    }
    sender.close().await?;
    Ok(())
}
