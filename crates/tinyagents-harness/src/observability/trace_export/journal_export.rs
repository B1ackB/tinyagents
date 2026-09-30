//! Pure journal-export helpers: stamp run lineage and trace attribution onto a
//! Langfuse trace config, gate content on `capture_content`, and fold a run's
//! usage aggregate into an ingestion batch. Pushing stays with the host.

use std::borrow::Cow;

use crate::events::AgentEvent;
use crate::observability::{AgentObservation, LangfuseTraceConfig};
use serde_json::{Map, Value, json};

use super::ExportBrand;
use super::ingestion_batch::new_event_id;
use super::types::TraceContext;

/// Run-level usage aggregate folded into a Langfuse batch as one extra
/// `run.total` generation (when no per-call cost is already present).
#[derive(Debug, Clone, Default)]
pub struct RunTotals {
    /// Run id the totals belong to.
    pub run_id: String,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Input tokens served from cache (subset of `input_tokens`).
    pub cached_input_tokens: u64,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Number of tool calls.
    pub tool_count: u64,
    /// Model label, when known.
    pub model: Option<String>,
    /// Provider label, when known.
    pub provider: Option<String>,
    /// Most recent error, when any step failed.
    pub error: Option<String>,
}

/// Enrich `trace_ctx` with the run lineage (`run_id` / `parent_run_id` /
/// `root_run_id`) carried by the run's journalled `observations` (#4657).
///
/// A single export corresponds to one run's observation stream (the journal is
/// read per run id), so every observation shares the same lineage and the first
/// is representative. For a spawned sub-agent that lineage points back at the
/// spawning turn, which is exactly what links the sub-agent's trace to its
/// parent. Returns the context unchanged when there are no observations.
pub fn trace_ctx_with_run_lineage(
    trace_ctx: &TraceContext,
    observations: &[AgentObservation],
) -> TraceContext {
    let Some(first) = observations.first() else {
        return trace_ctx.clone();
    };
    trace_ctx.clone().with_run_lineage(
        trace_ctx
            .run_id
            .clone()
            .or_else(|| Some(first.run_id.as_str().to_string())),
        trace_ctx.parent_run_id.clone().or_else(|| {
            first
                .parent_run_id
                .as_ref()
                .map(|id| id.as_str().to_string())
        }),
        trace_ctx
            .root_run_id
            .clone()
            .or_else(|| Some(first.root_run_id.as_str().to_string())),
    )
}

/// A child is a root observation within its own trace. Preserve its original
/// run lineage on the trace metadata before calling this projection.
pub fn root_subagent_observations(observations: &[AgentObservation]) -> Vec<AgentObservation> {
    observations
        .iter()
        .cloned()
        .map(|mut observation| {
            observation.parent_run_id = None;
            observation.root_run_id = observation.run_id.clone();
            observation
        })
        .collect()
}

pub fn trace_config_from_context(
    trace_ctx: &TraceContext,
    environment: &str,
    brand: &ExportBrand<'_>,
) -> LangfuseTraceConfig {
    let mut metadata = Map::new();
    if let Some(client_id) = &trace_ctx.client_id {
        metadata.insert("client.id".into(), json!(client_id));
    }
    if let Some(agent_id) = &trace_ctx.agent_id {
        metadata.insert("agent.id".into(), json!(agent_id));
    }
    if let Some(source) = &trace_ctx.channel_source {
        metadata.insert("channel.source".into(), json!(source));
    }
    metadata.insert("run_type".into(), json!(trace_ctx.run_type.as_str()));
    metadata.insert("app.version".into(), json!(brand.version));
    // Run lineage (#4657): stamp the run/parent/root ids so a spawned sub-agent's
    // trace is navigable from — and threadable under — its parent turn. Omitted
    // keys (e.g. `parent_run_id` for a top-level turn) simply stay absent.
    if let Some(run_id) = &trace_ctx.run_id {
        metadata.insert("run_id".into(), json!(run_id));
    }
    if let Some(parent_run_id) = &trace_ctx.parent_run_id {
        metadata.insert("parent_run_id".into(), json!(parent_run_id));
    }
    if let Some(root_run_id) = &trace_ctx.root_run_id {
        metadata.insert("root_run_id".into(), json!(root_run_id));
    }

    let mut tags = vec![format!("run:{}", trace_ctx.run_type.as_str())];
    if let Some(source) = &trace_ctx.channel_source {
        tags.push(format!("source:{source}"));
    }

    LangfuseTraceConfig {
        trace_id: Some(trace_ctx.session_id.clone()),
        name: Some(match &trace_ctx.agent_id {
            Some(agent_id) => format!("agent.turn:{agent_id}"),
            None => "agent.turn".to_string(),
        }),
        user_id: trace_ctx.user_id.clone(),
        session_id: trace_ctx
            .session_group
            .clone()
            .or_else(|| Some(trace_ctx.session_id.clone())),
        environment: Some(environment.to_string()),
        release: Some(brand.version.to_string()),
        version: Some(brand.version.to_string()),
        tags,
        metadata: Value::Object(metadata),
    }
}

pub fn observations_for_export<'a>(
    trace_ctx: &TraceContext,
    observations: &'a [AgentObservation],
) -> Cow<'a, [AgentObservation]> {
    if trace_ctx.capture_content {
        return Cow::Borrowed(observations);
    }

    Cow::Owned(
        observations
            .iter()
            .cloned()
            .map(strip_observation_content)
            .collect(),
    )
}

fn strip_observation_content(mut observation: AgentObservation) -> AgentObservation {
    match &mut observation.event {
        AgentEvent::ModelCompleted { input, output, .. }
        | AgentEvent::ToolCompleted { input, output, .. } => {
            *input = None;
            *output = None;
        }
        _ => {}
    }
    observation
}

pub fn insert_run_telemetry_generation(
    payload: &mut Value,
    telemetry: Option<&RunTotals>,
    brand: &ExportBrand<'_>,
) -> bool {
    let Some(telemetry) = telemetry else {
        return false;
    };
    if telemetry.input_tokens == 0 && telemetry.output_tokens == 0 && telemetry.cost_usd == 0.0 {
        return false;
    }

    let Some(batch) = payload.get_mut("batch").and_then(Value::as_array_mut) else {
        return false;
    };
    // Native per-call charges are already counted by Langfuse. Adding the
    // run aggregate as another generation would double-count the same spend.
    if batch.iter().any(|event| {
        event["type"] == "generation-create"
            && event["body"]["costDetails"]["total"].as_f64().is_some()
    }) {
        return false;
    }
    let Some(trace_id) = batch
        .first()
        .and_then(|event| event.get("body"))
        .and_then(|body| body.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return false;
    };
    let start_time = batch
        .first()
        .and_then(|event| event.get("timestamp"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let end_time = batch
        .last()
        .and_then(|event| event.get("timestamp"))
        .and_then(Value::as_str)
        .unwrap_or(start_time.as_str())
        .to_string();

    let non_cached_input = telemetry
        .input_tokens
        .saturating_sub(telemetry.cached_input_tokens);
    let mut body = json!({
        "id": format!("{trace_id}:{}-run-telemetry", brand.product),
        "traceId": trace_id,
        "name": "run.total",
        "startTime": start_time,
        "endTime": end_time,
        "usageDetails": {
            "input": non_cached_input,
            "output": telemetry.output_tokens,
            "total": telemetry.input_tokens.saturating_add(telemetry.output_tokens),
            "cache_read_input_tokens": telemetry.cached_input_tokens,
        },
        "costDetails": {
            "total": telemetry.cost_usd,
        },
        "metadata": {
            "source": format!("{}.run_telemetry", brand.product),
            "run_id": telemetry.run_id,
            "tool_count": telemetry.tool_count,
        },
    });
    if let Some(model) = &telemetry.model {
        body["model"] = json!(model);
    }
    if let Some(provider) = &telemetry.provider {
        body["metadata"]["provider"] = json!(provider);
    }
    if let Some(error) = &telemetry.error {
        body["level"] = json!("ERROR");
        body["statusMessage"] = json!(error);
    }

    batch.insert(
        1,
        json!({
            "id": new_event_id(),
            "type": "generation-create",
            "timestamp": body["startTime"].clone(),
            "body": body,
        }),
    );
    true
}

#[cfg(test)]
#[path = "journal_export_test.rs"]
mod tests;
