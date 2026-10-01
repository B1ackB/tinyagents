# harness::observability::trace_export

Pure helpers that turn a finished run's spans into telemetry payloads: the
span data model, NDJSON serialization, OTLP/HTTP JSON in Langfuse conventions,
Langfuse ingestion-batch chunking, and the pure parts of exporting a durable
journal.

## Design

Nothing here performs I/O. The host owns credentials, URL resolution, the HTTP
transport, and the progress-event collector that builds `TraceSpan`s during a
run. This module only shapes data, so every function is deterministic apart
from id minting and is unit-tested without a network or a clock.

Product identity stays out of the crate: the host passes an `ExportBrand`
(product slug and version), which becomes the OTLP `service.name`, the
`<product>.agent` scope, trace metadata keys and the release version.

## Public surface

- `types` (re-exported at the module root): `TraceSpan`, `SpanKind`,
  `SpanStatus`, `RunType`, `TraceContext`, `trace_session_id`.
- `ExportBrand`: product identity stamped onto exported telemetry.
- `serialize`: content caps (`MAX_TOOL_CONTENT_CHARS`,
  `MAX_ERROR_MESSAGE_CHARS`, `MAX_MODEL_CONTENT_CHARS`),
  `capture_model_content` (keeps a large request structured, newest messages
  first, with an omitted-messages marker), `truncate_chars`,
  `truncate_capture_text`, the small `json_*` value builders, `status_of`, and
  `spans_to_ndjson` with its `SpanEnvelope`.
- `otlp`: `span_to_otlp`, `otlp_requests`, `prepare_subagent_root`,
  `compact_internal_searches`.
- `ingestion_batch`: `split_ingestion_batch` (bounded by
  `LANGFUSE_MAX_BATCH_EVENTS`), `iso_millis`, `new_event_id`.
- `journal_export` (feature `langfuse`): `RunTotals`,
  `trace_ctx_with_run_lineage`, `root_subagent_observations`,
  `trace_config_from_context`, `observations_for_export`,
  `insert_run_telemetry_generation`.

## Operational constraints

- Captured content is always bounded per field. Tool arguments and output,
  error text and model content are truncated to the caps above, which bounds
  each event. It does not bound a whole batch: `split_ingestion_batch` limits
  the event count, not serialized bytes, so a host sending to a backend with a
  request-size limit should keep content capture conservative or chunk by
  size itself.
- `capture_model_content` never drops the newest message: when it alone
  exceeds the budget, a bounded preview of it is kept.
- Langfuse caps the events per ingestion request, so payloads go through
  `split_ingestion_batch` before sending.
- Callers decide whether content capture is on at all; these helpers do not
  redact secrets. Run them on events that already passed a `RedactingSink`.
