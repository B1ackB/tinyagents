use super::*;
use serde_json::json;

#[test]
fn split_ingestion_batch_chunks_over_the_limit() {
    // 1201 events with max 500 -> chunks of 500, 500, 201.
    let events: Vec<Value> = (0..1201).map(|i| json!({ "id": i })).collect();
    let payload = json!({ "batch": events, "metadata": { "k": "v" } });

    let parts = split_ingestion_batch(payload, 500);
    assert_eq!(parts.len(), 3);
    let sizes: Vec<usize> = parts
        .iter()
        .map(|p| p["batch"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, vec![500, 500, 201]);
    // Every chunk stays within the limit and preserves other top-level keys.
    for p in &parts {
        assert!(p["batch"].as_array().unwrap().len() <= 500);
        assert_eq!(p["metadata"]["k"], json!("v"));
    }
    // The first event of the run (e.g. the trace-create) lands in chunk 0.
    assert_eq!(parts[0]["batch"][0]["id"], json!(0));
    // Order is preserved across the split.
    assert_eq!(parts[2]["batch"][0]["id"], json!(1000));
}

#[test]
fn split_ingestion_batch_passes_small_payloads_through() {
    let payload = json!({ "batch": [json!({ "id": 1 }), json!({ "id": 2 })] });
    let parts = split_ingestion_batch(payload.clone(), 500);
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], payload);
    // A payload without a `batch` array is returned untouched.
    let no_batch = json!({ "hello": "world" });
    assert_eq!(split_ingestion_batch(no_batch.clone(), 500), vec![no_batch]);
}

#[test]
fn iso_millis_formats_epoch_as_rfc3339() {
    // 2021-01-01T00:00:00Z = 1_609_459_200_000 ms.
    assert!(iso_millis(1_609_459_200_000).starts_with("2021-01-01T00:00:00"));
}
