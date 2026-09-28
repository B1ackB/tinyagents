use super::unknown_tool::{closest_tool_names, unknown_tool_message};

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|name| (*name).to_string()).collect()
}

#[test]
fn keeps_the_unknown_tool_prefix_hosts_classify_on() {
    let message = unknown_tool_message(
        "web_search_tool",
        &serde_json::json!({}),
        &names(&["web_fetch"]),
        true,
    );
    assert!(
        message.starts_with("unknown tool `web_search_tool`"),
        "{message}"
    );
}

#[test]
fn points_to_tool_search_instead_of_listing_every_tool() {
    let mut available = names(&["web_fetch", "memory_recall", "shell"]);
    available.extend((0..200).map(|i| format!("CONNECTOR_ACTION_{i}")));
    let message = unknown_tool_message("web_search_tool", &serde_json::json!({}), &available, true);
    assert!(
        message.contains("call `tool_search` with what you want to do"),
        "{message}"
    );
    assert!(!message.contains("valid tools"), "{message}");
    assert!(!message.contains("CONNECTOR_ACTION"), "{message}");
    assert!(!message.contains("memory_recall"), "{message}");
    assert!(message.len() < 400, "corrective must stay small: {message}");
}

#[test]
fn without_tool_search_it_does_not_mention_it() {
    let message = unknown_tool_message(
        "web_search_tool",
        &serde_json::json!({}),
        &names(&["web_fetch"]),
        false,
    );
    assert!(!message.contains("tool_search"), "{message}");
    assert!(
        message.contains("Use only the tools in your tool list."),
        "{message}"
    );
}

#[test]
fn names_close_matches_for_a_near_miss() {
    let available = names(&["web_fetch", "web_search", "file_read", "shell"]);
    let message =
        unknown_tool_message("web_search_tool", &serde_json::json!({}), &available, false);
    assert!(
        message.contains("Closest available: `web_search`, `web_fetch`."),
        "{message}"
    );
}

#[test]
fn no_close_match_means_no_suggestion_line() {
    let message = unknown_tool_message(
        "frobnicate",
        &serde_json::json!({}),
        &names(&["file_read", "shell"]),
        true,
    );
    assert!(!message.contains("Closest available"), "{message}");
}

#[test]
fn closest_ranks_by_shared_tokens_then_prefix_and_caps_the_count() {
    let available = names(&[
        "mcp_search_566bdc22e5d1",
        "web_fetch",
        "exa_search",
        "web_search_news",
        "search",
        "file_read",
    ]);
    let got = closest_tool_names("web_search_tool", &available, 3);
    assert_eq!(got, vec!["web_search_news", "web_fetch", "search"]);
}

#[test]
fn generic_tokens_alone_are_not_a_match() {
    let available = names(&["file_read_tool", "mcp_list_tools"]);
    assert!(closest_tool_names("web_search_tool", &available, 3).is_empty());
}

#[test]
fn the_requested_name_itself_is_never_suggested() {
    let available = names(&["web_search_tool", "web_fetch"]);
    assert_eq!(
        closest_tool_names("web_search_tool", &available, 3),
        vec!["web_fetch"]
    );
}

#[test]
fn message_echoes_the_attempted_arguments() {
    let args = serde_json::json!({"path": "/tmp/a"});
    let message = unknown_tool_message("missing", &args, &names(&["file_read"]), false);
    assert!(message.starts_with("unknown tool `missing`"), "{message}");
    assert!(message.contains("\"path\":\"/tmp/a\""), "{message}");
}
