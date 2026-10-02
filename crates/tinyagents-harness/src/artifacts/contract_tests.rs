use std::collections::HashSet;

use super::*;

/// The rendered prompt is prefix-cached by hosts, so its bytes are a contract.
const EXPECTED: &str = include_str!("contract_expected.txt");

#[test]
fn contract_renders_byte_identically() {
    assert_eq!(render_artifact_offload_contract("file_write"), EXPECTED);
}

#[test]
fn contract_names_both_directories_heading_and_only_the_write_tool() {
    let rendered = render_artifact_offload_contract("file_write");
    assert!(rendered.starts_with(ARTIFACT_OFFLOAD_HEADING));
    assert!(rendered.contains(OUTPUTS_DIR));
    assert!(rendered.contains(SCRATCH_DIR));
    assert!(rendered.contains("`file_write`"));
    assert!(!rendered.contains("file_read"));
}

#[test]
fn contract_follows_the_write_tool_parameter() {
    let rendered = render_artifact_offload_contract("save_file");
    assert!(rendered.contains("with `save_file`"));
    assert!(!rendered.contains("file_write"));
}

#[test]
fn contract_is_rendered_only_for_agents_holding_the_write_tool() {
    let mut tools = HashSet::new();
    assert!(!should_render_offload_contract(&tools, "file_write"));
    tools.insert("web_search".to_string());
    assert!(!should_render_offload_contract(&tools, "file_write"));
    tools.insert("file_write".to_string());
    assert!(should_render_offload_contract(&tools, "file_write"));
    assert!(!should_render_offload_contract(&tools, "other_tool"));
}
