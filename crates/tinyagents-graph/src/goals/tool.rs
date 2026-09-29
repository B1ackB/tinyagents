//! Model-facing controls for the thread goal, exposed as ordinary harness
//! [`Tool`]s.
//!
//! Ownership is **asymmetric**: a model may read the goal (`goal_get`), create
//! or replace it (`goal_set`), and mark it done (`goal_complete`). Pause /
//! resume / clear are host-driven controls — constructible as tools for a host
//! that wants to expose them, but not part of the default model-facing set
//! returned by [`goal_tools`].
//!
//! Every control answers with the JSON `{ "goal": <ThreadGoal|null>, "text":
//! <rendered block> }`: `goal` is the structured camelCase goal a UI can draw
//! a banner from, `text` the model-readable rendering (also attached as the
//! markdown form). A host that must react to a change (publish an event, refresh
//! a chip) registers a hook with [`GoalTool::with_update_hook`].
//!
//! The target thread is resolved from
//! [`ToolExecutionContext::thread_id`](tinyagents_harness::tool::ToolExecutionContext),
//! the harness analogue of an ambient thread id: a tool never takes a
//! `thread_id` argument, so a model can't address another thread's goal. The
//! bare [`Tool::execute`] entry point (no context) errors, matching the "tools
//! require an active thread" contract.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::store;
use super::types::ThreadGoal;
use tinyagents_harness::error::Result;
use tinyagents_harness::store::Store;
use tinyagents_harness::tool::ToolRegistry;
use tinytools::{PermissionLevel, Tool, ToolPolicy, ToolResult, ToolRunContext, ToolSideEffects};

/// Callback a host registers to observe a goal a tool just wrote.
///
/// Called after a successful `goal_set` / `goal_complete` / `goal_pause` /
/// `goal_resume` with the persisted goal (never for `goal_get` or `goal_clear`).
pub type GoalUpdateHook = Arc<dyn Fn(&ThreadGoal) + Send + Sync>;

/// Which thread-goal control a [`GoalTool`] implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalToolKind {
    /// Read the current thread goal.
    Get,
    /// Create or replace the current thread goal.
    Set,
    /// Mark the current thread goal complete.
    Complete,
    /// Pause the current thread goal (host control).
    Pause,
    /// Resume a paused thread goal (host control).
    Resume,
    /// Delete the current thread goal (host control).
    Clear,
}

impl GoalToolKind {
    /// The default model-facing controls (asymmetric ownership).
    pub const MODEL_FACING: [Self; 3] = [Self::Get, Self::Set, Self::Complete];

    /// Every control, including the host-driven pause/resume/clear.
    pub const ALL: [Self; 6] = [
        Self::Get,
        Self::Set,
        Self::Complete,
        Self::Pause,
        Self::Resume,
        Self::Clear,
    ];

    /// Stable model-visible tool name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Get => "goal_get",
            Self::Set => "goal_set",
            Self::Complete => "goal_complete",
            Self::Pause => "goal_pause",
            Self::Resume => "goal_resume",
            Self::Clear => "goal_clear",
        }
    }

    /// Short model-visible description.
    pub fn description(self) -> &'static str {
        match self {
            Self::Get => {
                "Read this thread's goal — the durable objective you're pursuing across \
                 turns — with its status (active/paused/budget_limited/complete) and token \
                 usage. Returns 'no goal set' when the thread has none."
            }
            Self::Set => {
                "Set (or replace) this thread's goal — the durable objective you should \
                 keep pursuing across turns until it's complete. Use at the start of a \
                 non-trivial request, or to refine the objective as it sharpens. Changing \
                 the objective resets usage counters. Optionally set a token_budget; when \
                 reached, the goal pauses with a progress summary."
            }
            Self::Complete => {
                "Mark this thread's goal complete. Only call this when concrete evidence \
                 confirms the objective is satisfied — completing stops autonomous \
                 continuation."
            }
            Self::Pause => "Pause this thread's active goal (host control).",
            Self::Resume => "Resume this thread's paused goal (host control).",
            Self::Clear => "Delete this thread's goal (host control).",
        }
    }

    /// Whether the control only reads state.
    fn read_only(self) -> bool {
        matches!(self, Self::Get)
    }

    /// The model-visible JSON-schema parameters for the control.
    fn parameters(self) -> Value {
        match self {
            Self::Set => json!({
                "type": "object",
                "required": ["objective"],
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The durable objective — what 'done' looks like for this thread."
                    },
                    "token_budget": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional token ceiling for the goal. Omit for no limit."
                    }
                }
            }),
            _ => json!({ "type": "object", "properties": {} }),
        }
    }
}

/// A harness [`Tool`] for one thread-goal control, backed by a
/// [`Store`](tinyagents_harness::store::Store).
pub struct GoalTool {
    kind: GoalToolKind,
    store: Arc<dyn Store>,
    on_update: Option<GoalUpdateHook>,
}

impl GoalTool {
    /// Creates one goal tool of `kind` backed by `store`.
    pub fn new(kind: GoalToolKind, store: Arc<dyn Store>) -> Self {
        Self {
            kind,
            store,
            on_update: None,
        }
    }

    /// Registers a [`GoalUpdateHook`] invoked after each successful write.
    #[must_use]
    pub fn with_update_hook(mut self, hook: GoalUpdateHook) -> Self {
        self.on_update = Some(hook);
        self
    }

    /// The control kind this tool implements.
    pub fn kind(&self) -> GoalToolKind {
        self.kind
    }

    /// Dispatches the control against `thread_id`, returning the goal (when
    /// one exists after the call) and a model-facing note.
    async fn dispatch(
        &self,
        thread_id: &str,
        args: &Value,
    ) -> std::result::Result<(Option<ThreadGoal>, String), String> {
        let stringify = |error: tinyagents_harness::error::TinyAgentsError| error.to_string();
        match self.kind {
            GoalToolKind::Get => match store::get(&self.store, thread_id)
                .await
                .map_err(stringify)?
            {
                Some(goal) => Ok((Some(goal), String::new())),
                None => Ok((None, "no goal set for this thread".to_string())),
            },
            GoalToolKind::Set => {
                let Some(objective) = args.get("objective").and_then(Value::as_str) else {
                    return Err("Missing 'objective' parameter".to_string());
                };
                let token_budget = args.get("token_budget").and_then(Value::as_u64);
                let goal = store::set(&self.store, thread_id, objective, token_budget)
                    .await
                    .map_err(stringify)?;
                Ok((Some(goal), "Goal set.".to_string()))
            }
            GoalToolKind::Complete => {
                let goal = store::complete(&self.store, thread_id)
                    .await
                    .map_err(stringify)?;
                Ok((Some(goal), "Goal marked complete.".to_string()))
            }
            GoalToolKind::Pause => {
                let goal = store::pause(&self.store, thread_id)
                    .await
                    .map_err(stringify)?;
                Ok((Some(goal), String::new()))
            }
            GoalToolKind::Resume => {
                let goal = store::resume(&self.store, thread_id)
                    .await
                    .map_err(stringify)?;
                Ok((Some(goal), String::new()))
            }
            GoalToolKind::Clear => {
                let removed = store::clear(&self.store, thread_id)
                    .await
                    .map_err(stringify)?;
                Ok((None, format!("Goal cleared (removed={removed}).")))
            }
        }
    }
}

/// Builds the `{ goal, text }` payload every goal control answers with.
/// `text` is `note` followed by the rendered goal block (or just `note` when
/// there is no goal).
fn goal_payload(goal: Option<&ThreadGoal>, note: &str) -> Value {
    let text = match goal {
        Some(goal) if note.is_empty() => render_goal(goal),
        Some(goal) => format!("{note}\n{}", render_goal(goal)),
        None => note.to_string(),
    };
    json!({
        "goal": goal.map(|goal| serde_json::to_value(goal).unwrap_or(Value::Null)),
        "text": text,
    })
}

/// Renders a goal as a compact, model-readable block.
fn render_goal(goal: &ThreadGoal) -> String {
    let budget = match goal.token_budget {
        Some(b) => format!(
            "{} used / {b} budget ({} left)",
            goal.tokens_used,
            goal.budget_remaining().unwrap_or(0)
        ),
        None => format!("{} used / no budget", goal.tokens_used),
    };
    format!(
        "[thread_goal]\nstatus: {}\nobjective: {}\ntokens: {budget}\n[/thread_goal]",
        goal.status.as_str(),
        goal.objective
    )
}

fn error_result(message: impl Into<String>) -> ToolResult {
    ToolResult::error(message)
}

/// Builds the default **model-facing** goal controls (`goal_get`, `goal_set`,
/// `goal_complete`).
pub fn goal_tools(store: Arc<dyn Store>) -> Vec<Arc<GoalTool>> {
    GoalToolKind::MODEL_FACING
        .into_iter()
        .map(|kind| Arc::new(GoalTool::new(kind, store.clone())))
        .collect()
}

/// Registers the default model-facing goal controls into a tool registry.
pub fn register_goal_tools<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    store: Arc<dyn Store>,
) -> &mut ToolRegistry<State, Ctx> {
    for tool in goal_tools(store) {
        registry.register(tool);
    }
    registry
}

#[async_trait]
impl Tool for GoalTool {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn description(&self) -> &str {
        self.kind.description()
    }

    fn parameters_schema(&self) -> Value {
        self.kind.parameters()
    }

    fn supports_markdown(&self) -> bool {
        true
    }

    fn permission_level(&self) -> PermissionLevel {
        if self.kind.read_only() {
            PermissionLevel::ReadOnly
        } else {
            PermissionLevel::Write
        }
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy {
            classified: true,
            side_effects: ToolSideEffects {
                read_only: self.kind.read_only(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Ok(error_result(
            "thread goal tools require an active chat thread",
        ))
    }

    async fn execute_with_context(
        &self,
        args: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let Some(thread_id) = context.and_then(ToolRunContext::thread_id) else {
            return Ok(error_result(
                "thread goal tools require an active chat thread",
            ));
        };
        tracing::debug!(
            tool = self.kind.name(),
            thread_id,
            "[thread_goals] goal tool execute"
        );
        let (goal, note) = match self.dispatch(thread_id, &args).await {
            Ok(outcome) => outcome,
            Err(message) => return Ok(error_result(message)),
        };
        if let (Some(goal), Some(hook)) = (goal.as_ref(), self.on_update.as_ref())
            && !self.kind.read_only()
        {
            hook(goal);
        }
        let payload = goal_payload(goal.as_ref(), &note);
        let text = payload["text"].as_str().unwrap_or_default().to_string();
        Ok(ToolResult::success(payload.to_string()).with_markdown(text))
    }
}
