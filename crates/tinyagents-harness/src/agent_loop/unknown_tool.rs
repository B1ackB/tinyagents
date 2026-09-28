//! The corrective a model gets back when it calls a tool that does not exist
//! (under [`UnknownToolPolicy::ReturnToolError`](crate::runtime::types::UnknownToolPolicy)).
//!
//! It used to end with `valid tools: [..]`, every model-callable name. With a
//! connector catalog behind `tool_search` that is hundreds of names (a 212-tool
//! dump was observed) re-sent on every wrong guess: it floods the context, and
//! the model still has to guess from bare names with no descriptions. When the
//! discovery bridge is advertised the corrective sends the model to
//! `tool_search` instead, which returns matching tools with their schemas.
//! Either way it names at most a few close matches, so a near-miss typo still
//! resolves in one step.
//!
//! The message keeps its `unknown tool \`<name>\`` prefix: hosts classify
//! the result by it.

use crate::tool::discover::TOOL_SEARCH_NAME;

/// Most close matches named in the corrective.
pub(super) const MAX_SUGGESTIONS: usize = 3;

/// Name tokens too generic to count as a match (`web_search_tool` and
/// `file_read_tool` share nothing meaningful).
const GENERIC_TOKENS: &[&str] = &["tool", "tools", "mcp", "get", "set", "run"];

/// Builds the tool-error text for an unknown call to `requested`.
///
/// `available` is what the model may call (already filtered by the host
/// allowlist); `tool_search_available` is whether the discovery bridge is
/// advertised on this run.
pub(super) fn unknown_tool_message(
    requested: &str,
    available: &[String],
    tool_search_available: bool,
) -> String {
    let mut message = format!(
        "unknown tool `{requested}`: no tool with that name is available to you, and calling it \
         again will fail the same way."
    );
    let suggestions = closest_tool_names(requested, available, MAX_SUGGESTIONS);
    if !suggestions.is_empty() {
        let names = suggestions
            .iter()
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(", ");
        message.push_str(&format!(" Closest available: {names}."));
    }
    if tool_search_available {
        message.push_str(&format!(
            " To find the right tool, call `{TOOL_SEARCH_NAME}` with what you want to do in plain \
             words, then call a tool it returns."
        ));
    } else {
        message.push_str(" Use only the tools in your tool list.");
    }
    message
}

/// Lower-case name tokens, split on anything that is not alphanumeric.
fn tokens(name: &str) -> Vec<String> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() > 1)
        .map(str::to_ascii_lowercase)
        .filter(|token| !GENERIC_TOKENS.contains(&token.as_str()))
        .collect()
}

/// Up to `limit` available names sharing at least one meaningful token with
/// `requested`, best first: more shared tokens, then a longer common prefix,
/// then the shorter name, then alphabetical (so the output is stable).
pub(super) fn closest_tool_names<'a>(
    requested: &str,
    available: &'a [String],
    limit: usize,
) -> Vec<&'a str> {
    let wanted = tokens(requested);
    if wanted.is_empty() {
        return Vec::new();
    }
    let requested_lower = requested.to_ascii_lowercase();
    let mut scored: Vec<(usize, usize, &str)> = available
        .iter()
        .filter(|name| name.as_str() != requested)
        .filter_map(|name| {
            let own = tokens(name);
            let shared = wanted.iter().filter(|token| own.contains(token)).count();
            (shared > 0).then(|| {
                let prefix = requested_lower
                    .chars()
                    .zip(name.to_ascii_lowercase().chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                (shared, prefix, name.as_str())
            })
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(b.1.cmp(&a.1))
            .then(a.2.len().cmp(&b.2.len()))
            .then(a.2.cmp(b.2))
    });
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, name)| name)
        .collect()
}
