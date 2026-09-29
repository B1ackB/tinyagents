//! Builtin time/date tools: [`CurrentTimeTool`] and [`ResolveTimeTool`].
//!
//! Both implement `tinytools::Tool` directly (no recursive dispatch) and are
//! registered via [`register_time_tools`] or collected with [`time_tools`].
//! They exist so a model can ground relative expressions ("in 10 minutes",
//! "tomorrow") in an exact timestamp instead of hand-computing one, which
//! models are unreliable at. Parsing (`resolve_expr`, `parse_relative_duration`)
//! and timezone handling (`ResolveZone`) live in `time_parse.rs`, free of any
//! `Tool` plumbing so they are unit-testable on their own.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Local, SecondsFormat, Utc};
use chrono_tz::Tz;
use serde_json::json;

use super::time_parse::{ResolveZone, resolve_expr};
use crate::tool::ToolRegistry;
use tinytools::{Tool, ToolCallOptions, ToolPolicy, ToolResult};

/// Declared name of [`CurrentTimeTool`].
const CURRENT_TIME_NAME: &str = "current_time";
/// Declared name of [`ResolveTimeTool`].
const RESOLVE_TIME_NAME: &str = "resolve_time";

/// Tool that returns the current time in UTC and local time, optionally
/// converted to an IANA timezone.
pub struct CurrentTimeTool;

impl CurrentTimeTool {
    /// Creates a current-time tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CurrentTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CurrentTimeTool {
    fn name(&self) -> &str {
        CURRENT_TIME_NAME
    }

    fn description(&self) -> &str {
        "Get the current date and time in UTC and the machine's local timezone. \
         Optionally convert to a specific IANA timezone such as \
         'America/Los_Angeles' or 'Asia/Kolkata'. Use before scheduling tasks or \
         when a user refers to relative times like 'in 10 minutes', 'tomorrow', \
         or 'tonight'."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone name, for example 'Europe/London'."
                }
            }
        })
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    fn supports_markdown(&self) -> bool {
        true
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.execute_with_options(arguments, ToolCallOptions::default())
            .await
    }

    async fn execute_with_options(
        &self,
        arguments: serde_json::Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        tracing::debug!(args = %arguments, "[current_time] execute start");
        let payload = current_time_payload(&arguments);
        let mut result = ToolResult::success(serde_json::to_string_pretty(&payload)?);
        if options.prefer_markdown {
            result = result.with_markdown(current_time_markdown(&payload));
        }
        Ok(result)
    }
}

/// Renders the [`CurrentTimeTool`] payload as a compact markdown list.
fn current_time_markdown(payload: &serde_json::Value) -> String {
    let text = |value: &serde_json::Value| value.as_str().unwrap_or("").to_string();
    let mut md = String::new();
    md.push_str(&format!("- **utc**: {}\n", text(&payload["utc"])));
    md.push_str(&format!(
        "- **local**: {} ({})\n",
        text(&payload["local"]),
        text(&payload["local_timezone"])
    ));
    md.push_str(&format!("- **weekday**: {}\n", text(&payload["weekday"])));
    md.push_str(&format!(
        "- **unix_seconds**: {}\n",
        payload["unix_seconds"].as_i64().unwrap_or(0)
    ));
    if let Some(requested) = payload.get("requested_timezone") {
        md.push_str(&format!(
            "- **{}**: {} ({})\n",
            text(&requested["name"]),
            text(&requested["time"]),
            text(&requested["weekday"])
        ));
    }
    if let Some(error) = payload
        .get("requested_timezone_error")
        .and_then(|value| value.as_str())
    {
        md.push_str(&format!("- **timezone error**: {error}\n"));
    }
    md
}

/// Builds the JSON payload for [`CurrentTimeTool`]: always UTC + local time,
/// plus a `requested_timezone` (or `requested_timezone_error`) entry when
/// `args.timezone` names a valid (or invalid) IANA zone.
fn current_time_payload(args: &serde_json::Value) -> serde_json::Value {
    let now_utc = Utc::now();
    let now_local = Local::now();

    let mut payload = json!({
        "utc": now_utc.to_rfc3339_opts(SecondsFormat::Secs, true),
        "local": now_local.to_rfc3339_opts(SecondsFormat::Secs, true),
        "local_timezone": now_local.format("%Z").to_string(),
        "unix_seconds": now_utc.timestamp(),
        "weekday": now_local.format("%A").to_string(),
    });

    if let Some(tz_name) = args.get("timezone").and_then(|value| value.as_str()) {
        let trimmed = tz_name.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<Tz>() {
                Ok(tz) => {
                    let converted = now_utc.with_timezone(&tz);
                    payload["requested_timezone"] = json!({
                        "name": trimmed,
                        "time": converted.to_rfc3339_opts(SecondsFormat::Secs, true),
                        "weekday": converted.format("%A").to_string(),
                    });
                }
                Err(_) => {
                    payload["requested_timezone_error"] = json!(format!(
                        "Unknown IANA timezone '{trimmed}' - use names like 'America/Los_Angeles'."
                    ));
                }
            }
        }
    }

    payload
}

/// Tool that resolves relative or absolute time expressions to exact
/// timestamps.
pub struct ResolveTimeTool;

impl ResolveTimeTool {
    /// Creates a resolve-time tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ResolveTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ResolveTimeTool {
    fn name(&self) -> &str {
        RESOLVE_TIME_NAME
    }

    fn description(&self) -> &str {
        "Resolve a relative or absolute time expression into an exact timestamp. \
         Use this to produce date/time arguments for other tools instead of \
         hand-computing Unix seconds. Accepted expressions include 'now', \
         '24h ago', '7d', '2 weeks ago', 'in 10 minutes', '30m from now', \
         'today', 'yesterday', 'tomorrow', 'tomorrow at 9am', 'since Monday', \
         'next Friday at 3:30 pm', '11pm tonight', RFC-3339 timestamps, bare dates, \
         and 'YYYY-MM-DD HH:MM:SS'."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "Time expression to resolve."
                },
                "format": {
                    "type": "string",
                    "enum": ["unix_s", "unix_ms", "slack_ts", "rfc3339"],
                    "description": "Representation to place in the top-level value field. Defaults to unix_s."
                },
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone used to interpret offset-less inputs."
                }
            },
            "required": ["expr"]
        })
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    fn supports_markdown(&self) -> bool {
        true
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.execute_with_options(arguments, ToolCallOptions::default())
            .await
    }

    async fn execute_with_options(
        &self,
        arguments: serde_json::Value,
        options: ToolCallOptions,
    ) -> anyhow::Result<ToolResult> {
        tracing::debug!(args = %arguments, "[resolve_time] execute start");
        let expr = match arguments.get("expr").and_then(|value| value.as_str()) {
            Some(expr) => expr,
            None => {
                return Ok(ToolResult::error(
                    "resolve_time: `expr` is required (e.g. \"24h ago\", \
                     \"2026-06-09T19:12:00Z\", \"now\").",
                ));
            }
        };

        let zone = match arguments.get("timezone").and_then(|value| value.as_str()) {
            Some(tz_name) if !tz_name.trim().is_empty() => match tz_name.trim().parse::<Tz>() {
                Ok(tz) => ResolveZone::Iana(tz),
                Err(_) => {
                    return Ok(ToolResult::error(format!(
                        "resolve_time: unknown IANA timezone '{}' - use names like 'America/Los_Angeles'.",
                        tz_name.trim()
                    )));
                }
            },
            _ => ResolveZone::Local,
        };

        let dt = match resolve_expr(expr, zone) {
            Ok(dt) => dt,
            Err(error) => {
                tracing::debug!(expr = expr, error = %error, "[resolve_time] parse failed");
                return Ok(ToolResult::error(format!("resolve_time: {error}")));
            }
        };

        let payload = resolve_time_payload(expr, &arguments, dt);
        tracing::debug!(
            "[resolve_time] resolved {expr:?} -> {} (unix_s={})",
            payload["rfc3339"],
            payload["unix_s"]
        );
        let mut result = ToolResult::success(serde_json::to_string_pretty(&payload)?);
        if options.prefer_markdown {
            result = result.with_markdown(resolve_time_markdown(&payload));
        }
        Ok(result)
    }
}

/// Renders the [`ResolveTimeTool`] payload as a compact markdown list.
fn resolve_time_markdown(payload: &serde_json::Value) -> String {
    let text = |key: &str| match &payload[key] {
        serde_json::Value::String(value) => value.clone(),
        other => other.to_string(),
    };
    format!(
        "- **interpreted**: {}\n- **value**: {}\n- **unix_s**: {}\n- **unix_ms**: {}\n\
         - **slack_ts**: {}\n- **rfc3339**: {}\n",
        text("interpreted"),
        text("value"),
        text("unix_s"),
        text("unix_ms"),
        text("slack_ts"),
        text("rfc3339"),
    )
}

/// Builds the JSON payload for [`ResolveTimeTool`]: the resolved instant
/// rendered in every supported representation (`unix_s`/`unix_ms`/
/// `slack_ts`/`rfc3339`), with `args.format` (default `unix_s`) selecting
/// which one is duplicated into the top-level `value` field.
fn resolve_time_payload(
    expr: &str,
    args: &serde_json::Value,
    dt: DateTime<Utc>,
) -> serde_json::Value {
    let unix_s = dt.timestamp();
    let unix_ms = dt.timestamp_millis();
    let slack_ts = format!("{unix_s}.000000");
    let rfc3339 = dt.to_rfc3339_opts(SecondsFormat::Secs, true);

    let format = args
        .get("format")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|format| !format.is_empty())
        .unwrap_or("unix_s");
    let value = match format {
        "unix_ms" => unix_ms.to_string(),
        "slack_ts" => slack_ts.clone(),
        "rfc3339" => rfc3339.clone(),
        _ => unix_s.to_string(),
    };

    json!({
        "interpreted": expr,
        "value": value,
        "unix_s": unix_s,
        "unix_ms": unix_ms,
        "slack_ts": slack_ts,
        "rfc3339": rfc3339,
    })
}

/// Returns the builtin time tool set.
pub fn time_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(CurrentTimeTool::new()),
        Arc::new(ResolveTimeTool::new()),
    ]
}

/// Registers the builtin time tool set into an existing registry.
pub fn register_time_tools<State: Send + Sync + 'static, Ctx: Send + Sync + 'static>(
    registry: &mut ToolRegistry<State, Ctx>,
) {
    for tool in time_tools() {
        registry.register(tool);
    }
}
