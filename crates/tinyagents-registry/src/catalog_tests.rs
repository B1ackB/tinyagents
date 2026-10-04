use super::*;

#[test]
fn loads_seed_model_catalog_snapshot() {
    let catalog = ModelCatalog::seed().unwrap();

    assert_eq!(catalog.snapshot().schema_version, 1);
    assert!(catalog.get("openai", "gpt-4.1").is_some());
    assert!(catalog.get("anthropic", "claude-opus-4-5").is_some());
    assert!(catalog.get("gemini", "gemini-2.5-flash").is_some());
}

#[test]
fn looks_up_model_by_alias_or_id() {
    let catalog = ModelCatalog::seed().unwrap();

    // `openai/gpt-4.1` is a curated alias for `gpt-4.1` in the seed
    // snapshot, added by hand since models.dev does not publish
    // provider-prefixed aliases itself.
    let by_id = catalog.get_by_model_id("gpt-4.1").unwrap();
    let by_alias = catalog.get_by_model_id("openai/gpt-4.1").unwrap();

    assert_eq!(by_id.model_id, by_alias.model_id);
}

#[test]
fn bridges_catalog_entry_into_runtime_profile() {
    let catalog = ModelCatalog::seed().unwrap();
    let entry = catalog.get("openai", "gpt-4.1").unwrap();

    let profile = profile_from_entry(entry);
    assert_eq!(profile.provider.as_deref(), Some("openai"));
    assert_eq!(profile.model.as_deref(), Some("gpt-4.1"));
    // The catalog's advertised capability flags carry across the bridge.
    assert_eq!(profile.tool_calling, entry.capabilities.tool_calling);
    assert_eq!(profile.max_input_tokens, entry.max_input_tokens);

    // The convenience accessor returns the same bridged profile.
    let via_catalog = catalog.profile("openai", "gpt-4.1").unwrap();
    assert_eq!(via_catalog, profile);
}

// -----------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------

fn base_entry() -> ModelCatalogEntry {
    ModelCatalogEntry {
        provider: "openai".to_string(),
        model_id: "gpt-test".to_string(),
        aliases: Vec::new(),
        mode: "chat".to_string(),
        max_input_tokens: Some(100_000),
        max_output_tokens: Some(4_096),
        deprecation_date: None,
        release_date: Some("2026-01-01".to_string()),
        pricing: ModelPricing::default(),
        capabilities: ModelCapabilities::default(),
        source: "manual".to_string(),
        source_url: None,
        raw: Value::Null,
    }
}

fn base_snapshot(models: Vec<ModelCatalogEntry>) -> ModelCatalogSnapshot {
    ModelCatalogSnapshot {
        schema_version: 1,
        snapshot_id: "test".to_string(),
        created_at: "2026-01-01T00:00:00Z".to_string(),
        currency: "USD".to_string(),
        unit: "token".to_string(),
        description: None,
        sources: Vec::new(),
        models,
    }
}

#[test]
fn valid_snapshot_passes() {
    base_snapshot(vec![base_entry()]).validate().unwrap();
}

#[test]
fn rejects_duplicate_provider_model_id_pairs() {
    let snapshot = base_snapshot(vec![base_entry(), base_entry()]);
    let error = snapshot.validate().unwrap_err().to_string();
    assert!(error.contains("duplicate"), "got: {error}");
}

#[test]
fn rejects_negative_flat_price() {
    let mut entry = base_entry();
    entry.pricing.input_per_token = Some(-0.01);
    let error = base_snapshot(vec![entry])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("negative"), "got: {error}");
}

#[test]
fn rejects_negative_tiered_price() {
    let mut entry = base_entry();
    entry
        .pricing
        .tiers
        .push(tinyagents_harness::cost::PriceTier {
            up_to_tokens: None,
            input: Some(-1.0),
            output: None,
            cache_read: None,
            cache_write: None,
        });
    let error = base_snapshot(vec![entry])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("negative"), "got: {error}");
}

#[test]
fn rejects_missing_source() {
    let mut entry = base_entry();
    entry.source = String::new();
    let error = base_snapshot(vec![entry])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("missing a source"), "got: {error}");
}

#[test]
fn rejects_output_limit_exceeding_input_context() {
    let mut entry = base_entry();
    entry.max_input_tokens = Some(1_000);
    entry.max_output_tokens = Some(2_000);
    let error = base_snapshot(vec![entry])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("max_output_tokens"), "got: {error}");
}

#[test]
fn rejects_alias_collision() {
    let mut aliased = base_entry();
    aliased.model_id = "gpt-other".to_string();
    aliased.aliases = vec!["gpt-test".to_string()]; // collides with base_entry's id
    let error = base_snapshot(vec![base_entry(), aliased])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("collides"), "got: {error}");
}

#[test]
fn rejects_invalid_date() {
    let mut entry = base_entry();
    entry.deprecation_date = Some("not-a-date".to_string());
    let error = base_snapshot(vec![entry])
        .validate()
        .unwrap_err()
        .to_string();
    assert!(error.contains("not a valid date"), "got: {error}");
}

#[test]
fn accepts_plain_and_rfc3339_dates() {
    let mut entry = base_entry();
    entry.deprecation_date = Some("2026-01-01".to_string());
    base_snapshot(vec![entry.clone()]).validate().unwrap();
    entry.deprecation_date = Some("2026-01-01T00:00:00Z".to_string());
    base_snapshot(vec![entry]).validate().unwrap();
}

#[test]
fn default_validate_does_not_restrict_provider_ids() {
    // `validate()` (used by `from_json`/`try_from_snapshot`) must accept
    // a synthetic or fictional provider id, so a hand-written test or
    // example snapshot never has to name a real vendor.
    let mut entry = base_entry();
    entry.provider = "totally-fictional-vendor".to_string();
    base_snapshot(vec![entry]).validate().unwrap();
}

#[test]
fn validate_with_providers_rejects_ids_outside_the_allowlist() {
    let mut entry = base_entry();
    entry.provider = "totally-unknown-vendor".to_string();
    let error = base_snapshot(vec![entry])
        .validate_with_providers(Some(KNOWN_PROVIDERS))
        .unwrap_err()
        .to_string();
    assert!(error.contains("unrecognized provider"), "got: {error}");
}

#[test]
fn validate_with_providers_accepts_a_listed_id() {
    base_snapshot(vec![base_entry()])
        .validate_with_providers(Some(&["openai"]))
        .unwrap();
}

#[test]
fn from_json_rejects_an_invalid_snapshot() {
    let mut snapshot = base_snapshot(vec![base_entry()]);
    snapshot.models[0].pricing.input_per_token = Some(-1.0);
    let json = serde_json::to_string(&snapshot).unwrap();
    assert!(ModelCatalog::from_json(&json).is_err());
}

#[test]
fn media_profile_facts_are_explicit_and_unknown_models_stay_unknown() {
    let mut entry = base_entry();
    let profile = profile_from_entry(&entry);
    assert!(!profile.modalities.image_in);
    assert!(!profile.modalities.audio_in);
    assert!(!profile.modalities.video_in);
    assert!(!profile.modalities.document_in);
    entry.capabilities.pdf_input = true;
    entry.capabilities.audio_input = true;
    entry.capabilities.video_input = true;
    let profile = profile_from_entry(&entry);
    assert!(profile.modalities.audio_in);
    assert!(profile.modalities.video_in);
    assert!(profile.modalities.document_in);
    assert!(
        ModelCatalog::seed()
            .unwrap()
            .profile("unknown", "unknown")
            .is_none()
    );
}

#[test]
fn bundled_video_capabilities_match_raw_input_modalities() {
    let catalog = ModelCatalog::seed().unwrap();
    let mut video_models = 0;
    for entry in &catalog.snapshot().models {
        let expected = entry.raw["modalities"]["input"]
            .as_array()
            .is_some_and(|inputs| inputs.iter().any(|input| input == "video"));
        assert_eq!(
            entry.capabilities.video_input, expected,
            "{}",
            entry.model_id
        );
        assert_eq!(
            profile_from_entry(entry).modalities.video_in,
            expected,
            "{}",
            entry.model_id
        );
        video_models += usize::from(expected);
    }
    assert_eq!(video_models, 6);
    assert!(
        catalog
            .profile("gemini", "gemini-2.5-pro")
            .unwrap()
            .modalities
            .video_in
    );
}
