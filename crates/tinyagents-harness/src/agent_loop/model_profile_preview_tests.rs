//! The agent loop previews the target model's profile onto
//! [`RunContext::model_profile`] before `before_model` middleware runs, so
//! middleware can shape what it adds for that model (#6962).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result;
use crate::middleware::Middleware;
use crate::runtime::AgentHarness;
use crate::testkit::ScriptedModel;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelProfile, ModelRequest};

/// Records the profile each `before_model` call saw.
struct SeenProfiles(Arc<Mutex<Vec<Option<ModelProfile>>>>);

#[async_trait]
impl Middleware<()> for SeenProfiles {
    fn name(&self) -> &str {
        "seen-profiles"
    }

    async fn before_model(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
        _request: &mut ModelRequest,
    ) -> Result<()> {
        self.0.lock().unwrap().push(ctx.model_profile.clone());
        Ok(())
    }
}

#[tokio::test]
async fn before_model_sees_the_target_model_profile() {
    let profile = ModelProfile {
        model: Some("deepseek-chat".into()),
        hoists_system_messages: true,
        ..ModelProfile::default()
    };
    let model = Arc::new(ScriptedModel::replies(vec!["done"]).with_profile(profile.clone()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model);
    harness.push_middleware(Arc::new(SeenProfiles(seen.clone())));

    harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.as_slice(), [Some(profile)]);
}
