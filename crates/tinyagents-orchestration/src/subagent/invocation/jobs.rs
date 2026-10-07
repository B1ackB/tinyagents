//! Asynchronous subagent job registry and host-facing control tools.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::ids::next_seq;
use tinyagents_harness::steering::{
    RecentRequestIds, SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringPolicy,
};
use tinyagents_harness::tool::{ToolDispatch, ToolRegistry};
use tinyinference_llm::message::Message;
use tinytools::{Tool, ToolResult};

use crate::subagent::{AppliedResult, IncompleteKind};

use super::{
    JobLink, SubAgentJob, SubAgentJobEntry, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry,
    SubAgentJobStatus,
};

const LOG_PREFIX: &str = "[subagent-jobs]";

/// Settles an inline job if its tool future is dropped (tool timeout, parent
/// stream drop) or unwinds from a panic before the result is recorded.
/// Call [`Self::disarm`] once the result has been written.
pub(crate) struct InlineJobGuard {
    jobs: SubAgentJobRegistry,
    id: SubAgentJobId,
    armed: bool,
}

impl InlineJobGuard {
    pub(crate) fn new(jobs: SubAgentJobRegistry, id: SubAgentJobId) -> Self {
        Self {
            jobs,
            id,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for InlineJobGuard {
    fn drop(&mut self) {
        if self.armed {
            self.jobs.mark_aborted(&self.id, std::thread::panicking());
        }
    }
}

impl SubAgentJobRegistry {
    /// Creates an empty asynchronous job registry.
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(dead_code)]
    pub(crate) fn create(&self, agent: &str, owner: u64) -> (SubAgentJobId, SteeringHandle) {
        self.create_with_cancellation(agent, owner, CancellationToken::new(), JobLink::default())
    }

    /// Registers a job whose child run observes `cancellation`, so
    /// [`Self::cancel_owned`] can stop exactly this job.
    pub(crate) fn create_with_cancellation(
        &self,
        agent: &str,
        owner: u64,
        cancellation: CancellationToken,
        link: JobLink,
    ) -> (SubAgentJobId, SteeringHandle) {
        let id = SubAgentJobId(format!("subagent-job-{}", next_seq()));
        let steering =
            SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));
        let entry = SubAgentJobEntry {
            job: SubAgentJob {
                id: id.clone(),
                agent: agent.to_owned(),
                status: SubAgentJobStatus::Queued,
                output: None,
                error: None,
                subagent_run_id: link.subagent_run_id,
                parent_tool_call_id: link.parent_tool_call_id,
                incomplete_kind: None,
                artifacts: Vec::new(),
                schema_error: None,
                artifact_error: None,
            },
            owner,
            steering: steering.clone(),
            cancellation: Some(cancellation),
            message_requests: RecentRequestIds::default(),
            cancellation_requested: false,
        };
        self.write().insert(id.clone(), entry);
        (id, steering)
    }

    pub(crate) fn mark_running(&self, id: &SubAgentJobId) {
        if let Some(entry) = self.write().get_mut(id)
            && entry.job.status == SubAgentJobStatus::Queued
        {
            entry.job.status = SubAgentJobStatus::Running;
        }
    }

    pub(crate) fn mark_result(
        &self,
        id: &SubAgentJobId,
        result: Result<tinyagents_harness::middleware::AgentRun, TinyAgentsError>,
    ) {
        let mut entries = self.write();
        let Some(entry) = entries.get_mut(id) else {
            return;
        };
        if entry.job.status.is_terminal() {
            // Already settled (e.g. cancelled by the owner): the first
            // terminal state wins.
            tracing::debug!(
                "{LOG_PREFIX} mark_result.ignored job_id={id} status={:?}",
                entry.job.status
            );
            return;
        }
        entry.cancellation = None;
        let cancellation_requested = entry.cancellation_requested;
        match result {
            Ok(run) => {
                if cancellation_requested {
                    entry.job.status = SubAgentJobStatus::Cancelled;
                    entry.job.error = Some(TinyAgentsError::Cancelled.to_string());
                } else {
                    entry.job.status = SubAgentJobStatus::Completed;
                    entry.job.output = run.text();
                }
            }
            Err(TinyAgentsError::Cancelled) => {
                entry.job.status = SubAgentJobStatus::Cancelled;
                entry.job.error = Some(TinyAgentsError::Cancelled.to_string());
            }
            Err(error @ TinyAgentsError::LimitExceeded(_)) => {
                entry.job.status = SubAgentJobStatus::Incomplete;
                entry.job.incomplete_kind = Some(IncompleteKind::BudgetExceeded);
                entry.job.error = Some(error.to_string());
            }
            Err(error @ TinyAgentsError::Timeout(_)) => {
                entry.job.status = SubAgentJobStatus::Incomplete;
                entry.job.incomplete_kind = Some(IncompleteKind::Timeout);
                entry.job.error = Some(error.to_string());
            }
            Err(error) => {
                entry.job.status = SubAgentJobStatus::Failed;
                entry.job.error = Some(error.to_string());
            }
        }
    }

    /// Folds a result-policy application into a job that completed: replaces
    /// the visible output and records any artifact or schema error.
    pub(crate) fn apply_result(&self, id: &SubAgentJobId, applied: AppliedResult) {
        let mut entries = self.write();
        let Some(entry) = entries.get_mut(id) else {
            return;
        };
        if entry.job.status != SubAgentJobStatus::Completed {
            return;
        }
        entry.job.output = Some(applied.text);
        entry.job.artifacts.extend(applied.artifact);
        entry.job.schema_error = applied.schema_error;
        entry.job.artifact_error = applied.artifact_error;
    }

    /// Points the job link at the attempt that is now running.
    pub(crate) fn set_attempt_run_id(&self, id: &SubAgentJobId, run_id: &str) {
        if let Some(entry) = self.write().get_mut(id)
            && !entry.job.status.is_terminal()
        {
            entry.job.subagent_run_id = Some(run_id.to_owned());
        }
    }

    /// Settles a job whose run finished but overshot a budget: it keeps the
    /// (policy-applied) output and ends `Incomplete(BudgetExceeded)`.
    pub(crate) fn mark_budget_overrun(
        &self,
        id: &SubAgentJobId,
        applied: AppliedResult,
        reason: String,
    ) {
        let mut entries = self.write();
        let Some(entry) = entries.get_mut(id) else {
            return;
        };
        if entry.job.status.is_terminal() {
            return;
        }
        entry.cancellation = None;
        if entry.cancellation_requested {
            // An owner cancellation that raced the finish wins, as in
            // `mark_result`.
            entry.job.status = SubAgentJobStatus::Cancelled;
            entry.job.error = Some(TinyAgentsError::Cancelled.to_string());
            return;
        }
        entry.job.status = SubAgentJobStatus::Incomplete;
        entry.job.incomplete_kind = Some(IncompleteKind::BudgetExceeded);
        entry.job.error = Some(reason);
        entry.job.output = Some(applied.text);
        entry.job.artifacts.extend(applied.artifact);
        entry.job.schema_error = applied.schema_error;
        entry.job.artifact_error = applied.artifact_error;
    }

    /// Marks a job `Failed` because its child task panicked or was aborted
    /// before it could report a result.
    pub(crate) fn mark_aborted(&self, id: &SubAgentJobId, panicked: bool) {
        let mut entries = self.write();
        let Some(entry) = entries.get_mut(id) else {
            return;
        };
        if entry.job.status.is_terminal() {
            return;
        }
        tracing::warn!("{LOG_PREFIX} child_task.aborted job_id={id} panicked={panicked}");
        entry.cancellation = None;
        if panicked {
            entry.job.status = SubAgentJobStatus::Failed;
            entry.job.error = Some("subagent job panicked before completing".to_owned());
        } else {
            entry.job.status = SubAgentJobStatus::Cancelled;
            entry.job.error = Some(TinyAgentsError::Cancelled.to_string());
        }
    }

    /// Cancels one queued or running job owned by `owner` and marks it
    /// `Cancelled`. The job's own cancellation token is tripped, so the parent
    /// run and sibling jobs are unaffected.
    pub(crate) fn cancel_owned(
        &self,
        job_id: &str,
        owner: u64,
    ) -> Result<SubAgentJob, SubAgentJobError> {
        let id = SubAgentJobId(job_id.to_owned());
        let mut entries = self.write();
        let entry = entries
            .get_mut(&id)
            .filter(|entry| entry.owner == owner)
            .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
        if entry.job.status.is_terminal() {
            return Err(SubAgentJobError::Terminal {
                job_id: job_id.to_owned(),
                status: entry.job.status,
            });
        }
        tracing::debug!("{LOG_PREFIX} cancel_owned job_id={job_id}");
        if let Some(token) = entry.cancellation.take() {
            token.cancel();
        }
        entry.cancellation_requested = true;
        let mut snapshot = entry.job.clone();
        snapshot.error =
            Some("cancellation requested; job will be cancelled when the child unwinds".to_owned());
        Ok(snapshot)
    }

    /// Returns a snapshot for `job_id` when it belongs to `owner`.
    pub(crate) fn get_owned(&self, job_id: &str, owner: u64) -> Option<SubAgentJob> {
        self.read()
            .get(&SubAgentJobId(job_id.to_owned()))
            .filter(|entry| entry.owner == owner)
            .map(|entry| entry.job.clone())
    }

    /// Returns a job snapshot for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn get(&self, job_id: &str) -> Option<SubAgentJob> {
        self.read()
            .get(&SubAgentJobId(job_id.to_owned()))
            .map(|entry| entry.job.clone())
    }

    /// Returns this run's jobs in stable id order.
    fn list_owned(&self, owner: u64) -> Vec<SubAgentJob> {
        let mut jobs = self
            .read()
            .values()
            .filter(|entry| entry.owner == owner)
            .map(|entry| entry.job.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        jobs
    }

    /// Returns every job for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn list(&self) -> Vec<SubAgentJob> {
        let mut jobs = self
            .read()
            .values()
            .map(|entry| entry.job.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        jobs
    }

    /// Queues a user message for delivery at the running child's next safe
    /// steering checkpoint.
    #[allow(dead_code)]
    pub(crate) fn send_message_owned(
        &self,
        job_id: &str,
        owner: u64,
        message: impl Into<String>,
    ) -> Result<(), SubAgentJobError> {
        self.send_message_with_request_id(job_id, owner, message, None)
            .map(|_| ())
    }

    /// Idempotent [`Self::send_message_owned`]: a `request_id` already applied
    /// to this job is acknowledged (`Ok(true)`, "duplicate") without queueing
    /// the message again. Only the last
    /// [`RecentRequestIds::DEFAULT_CAPACITY`] ids per job are remembered, and a
    /// rejected send (unknown, foreign or terminal job) never consumes its id.
    pub(crate) fn send_message_with_request_id(
        &self,
        job_id: &str,
        owner: u64,
        message: impl Into<String>,
        request_id: Option<&str>,
    ) -> Result<bool, SubAgentJobError> {
        let id = SubAgentJobId(job_id.to_owned());
        let mut entries = self.write();
        let entry = entries
            .get_mut(&id)
            .filter(|entry| entry.owner == owner)
            .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
        if entry.job.status.is_terminal() {
            return Err(SubAgentJobError::Terminal {
                job_id: job_id.to_owned(),
                status: entry.job.status,
            });
        }
        if entry.cancellation_requested {
            return Err(SubAgentJobError::Cancelling(job_id.to_owned()));
        }
        if let Some(request_id) = request_id {
            match entry.message_requests.claim(request_id) {
                Ok(false) => {
                    tracing::debug!("{LOG_PREFIX} send_message.duplicate job_id={job_id}");
                    return Ok(true);
                }
                Ok(true) => {}
                Err(_) => return Err(SubAgentJobError::RequestIdTooLong),
            }
        }
        entry
            .steering
            .send(SteeringCommand::InjectMessage(Message::user(
                message.into(),
            )));
        Ok(false)
    }

    fn read(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, std::collections::HashMap<SubAgentJobId, SubAgentJobEntry>>
    {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, std::collections::HashMap<SubAgentJobId, SubAgentJobEntry>>
    {
        self.inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Typed host tool that queries a job or lists jobs owned by the requesting run.
#[derive(Clone)]
pub struct SubAgentJobsTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentJobsTool {
    /// Creates the query tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentJobsTool {
    fn name(&self) -> &str {
        "subagent_jobs"
    }

    fn description(&self) -> &str {
        "Query an asynchronous subagent job by id, list all subagent jobs, or cancel one job with action \"cancel\"."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" },
                "action": {
                    "type": "string",
                    "enum": ["query", "cancel"],
                    "description": "`query` (default) reads a job or lists jobs; `cancel` stops the job named by job_id."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_jobs requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentJobsTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        _call_id: tinyagents_harness::ids::CallId,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let object = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("arguments must be an object"))?;
        match object
            .get("action")
            .filter(|value| !value.is_null())
            .map(Value::as_str)
        {
            None | Some(Some("query")) => {}
            Some(Some("cancel")) => {
                let job_id = object
                    .get("job_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        anyhow::anyhow!("job_id must be a string for action `cancel`")
                    })?;
                let job = self.jobs.cancel_owned(job_id, parent.instance_id())?;
                return Ok(ToolResult::json(serde_json::to_value(job)?));
            }
            Some(_) => anyhow::bail!("action must be `query` or `cancel`"),
        }
        if let Some(value) = object.get("job_id") {
            let job_id = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("job_id must be a string when provided"))?;
            let job = self
                .jobs
                .get_owned(job_id, parent.instance_id())
                .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
            Ok(ToolResult::json(serde_json::to_value(job)?))
        } else {
            Ok(ToolResult::json(serde_json::to_value(
                self.jobs.list_owned(parent.instance_id()),
            )?))
        }
    }
}

/// Host tool that sends a message to a queued or running subagent job.
#[derive(Clone)]
pub struct SubAgentMessageTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentMessageTool {
    /// Creates the message tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentMessageTool {
    fn name(&self) -> &str {
        "subagent_message"
    }

    fn description(&self) -> &str {
        "Send a message to a queued or running asynchronous subagent job."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" },
                "message": { "type": "string" },
                "request_id": {
                    "type": "string",
                    "description": "Optional idempotency key: resending the same request_id to the same job does not queue the message again while it is among the most recent 64 request ids remembered for that job (older ids are evicted); ids over 128 bytes are rejected."
                }
            },
            "required": ["job_id", "message"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_message requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentMessageTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        _call_id: tinyagents_harness::ids::CallId,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let job_id = args
            .get("job_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("job_id must be a string"))?;
        let message = args
            .get("message")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("message must be a string"))?;
        let request_id = match args.get("request_id") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let id = value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("request_id must be a string when provided"))?;
                Some(id)
            }
        };
        let duplicate = self.jobs.send_message_with_request_id(
            job_id,
            parent.instance_id(),
            message,
            request_id,
        )?;
        let mut payload = json!({
            "job_id": job_id,
            "status": "message_queued"
        });
        if duplicate {
            payload["duplicate"] = Value::Bool(true);
        }
        Ok(ToolResult::json(payload))
    }
}

/// Registers the standard run-scoped query and message tools in a harness registry.
pub fn register_subagent_job_tools<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    jobs: SubAgentJobRegistry,
) -> &mut ToolRegistry<State, Ctx> {
    registry
        .register_dispatch(Arc::new(SubAgentJobsTool::new(jobs.clone())))
        .register_dispatch(Arc::new(SubAgentMessageTool::new(jobs)))
}
