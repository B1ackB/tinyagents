//! File-operation lists carried by compaction summaries.
//!
//! A summary that forgets which files the agent already read or changed sends
//! it back to re-read them. Compaction therefore appends `<read-files>` and
//! `<modified-files>` sections, derived from the tool calls being folded, and
//! carries the previous summary's lists forward so they accumulate across
//! compactions. Port of pi's `compaction/utils.ts` file-op tracking.
//!
//! Extraction is pluggable ([`FileOpExtractor`]). [`DefaultFileOpExtractor`]
//! reads the common path arguments (`path`, `file`, `file_path`, `paths`) and
//! classifies the call by tool name: names containing a mutating verb
//! (`write`, `edit`, `patch`, `create`, `delete`, `remove`, `append`,
//! `replace`, `move`, `rename`, `save`, `touch`, `mkdir`) are modifications,
//! every other path-carrying call is a read. A host with differently named
//! tools supplies its own extractor.

use std::collections::BTreeSet;

use tinyinference_llm::message::Message;
use tinyinference_llm::tool::ToolCall;

const READ_OPEN: &str = "<read-files>\n";
const READ_CLOSE: &str = "\n</read-files>";
const MODIFIED_OPEN: &str = "<modified-files>\n";
const MODIFIED_CLOSE: &str = "\n</modified-files>";

/// Argument names [`DefaultFileOpExtractor`] treats as file paths.
const PATH_ARGS: [&str; 4] = ["path", "file", "file_path", "paths"];

/// Tool-name fragments [`DefaultFileOpExtractor`] treats as modifying.
const MUTATING_VERBS: [&str; 13] = [
    "write", "edit", "patch", "create", "delete", "remove", "append", "replace", "move", "rename",
    "save", "touch", "mkdir",
];

/// Files touched by the tool calls of a stretch of conversation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileOperations {
    read: BTreeSet<String>,
    modified: BTreeSet<String>,
}

impl FileOperations {
    /// Records a file that was read.
    pub fn add_read(&mut self, path: &str) {
        if !path.is_empty() {
            self.read.insert(path.to_string());
        }
    }

    /// Records a file that was created, written, edited or otherwise changed.
    pub fn add_modified(&mut self, path: &str) {
        if !path.is_empty() {
            self.modified.insert(path.to_string());
        }
    }

    /// Unions `other` into `self`.
    pub fn merge(&mut self, other: &FileOperations) {
        self.read.extend(other.read.iter().cloned());
        self.modified.extend(other.modified.iter().cloned());
    }

    /// Files that were modified, sorted.
    pub fn modified(&self) -> Vec<&str> {
        self.modified.iter().map(String::as_str).collect()
    }

    /// Files that were read and never modified, sorted.
    pub fn read_only(&self) -> Vec<&str> {
        self.read
            .iter()
            .filter(|path| !self.modified.contains(*path))
            .map(String::as_str)
            .collect()
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.read.is_empty() && self.modified.is_empty()
    }
}

/// Reads the files one tool call touched into a [`FileOperations`].
pub trait FileOpExtractor: Send + Sync {
    /// Records the files `call` read or modified, if any.
    fn extract(&self, call: &ToolCall, ops: &mut FileOperations);
}

/// The default extractor; see the [module docs](self).
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultFileOpExtractor;

impl FileOpExtractor for DefaultFileOpExtractor {
    fn extract(&self, call: &ToolCall, ops: &mut FileOperations) {
        if call.invalid.is_some() {
            return;
        }
        let name = call.name.to_lowercase();
        let mutating = MUTATING_VERBS.iter().any(|verb| name.contains(verb));
        let mut record = |path: &str| {
            if mutating {
                ops.add_modified(path);
            } else {
                ops.add_read(path);
            }
        };
        for arg in PATH_ARGS {
            match call.arguments.get(arg) {
                Some(serde_json::Value::String(path)) => record(path),
                Some(serde_json::Value::Array(paths)) => {
                    paths.iter().filter_map(|p| p.as_str()).for_each(&mut record)
                }
                _ => {}
            }
        }
    }
}

/// Collects the file operations of every tool call in `messages`.
pub fn extract_file_operations(
    messages: &[Message],
    extractor: &dyn FileOpExtractor,
) -> FileOperations {
    let mut ops = FileOperations::default();
    for message in messages {
        if let Message::Assistant(assistant) = message {
            for call in &assistant.tool_calls {
                extractor.extract(call, &mut ops);
            }
        }
    }
    ops
}

/// Appends `<read-files>` / `<modified-files>` sections to `summary`; returns
/// it unchanged when `ops` is empty.
pub fn append_file_sections(summary: &str, ops: &FileOperations) -> String {
    let mut text = summary.trim_end().to_string();
    let read = ops.read_only();
    if !read.is_empty() {
        text.push_str(&format!("\n\n{READ_OPEN}{}{READ_CLOSE}", read.join("\n")));
    }
    let modified = ops.modified();
    if !modified.is_empty() {
        text.push_str(&format!(
            "\n\n{MODIFIED_OPEN}{}{MODIFIED_CLOSE}",
            modified.join("\n")
        ));
    }
    text
}

/// Splits the file sections [`append_file_sections`] wrote off `text`,
/// returning the remaining body and the operations they listed. Text without
/// sections comes back unchanged with an empty set.
pub fn split_file_sections(text: &str) -> (String, FileOperations) {
    let mut ops = FileOperations::default();
    let mut body = text.to_string();
    for (open, close, is_modified) in [
        (READ_OPEN, READ_CLOSE, false),
        (MODIFIED_OPEN, MODIFIED_CLOSE, true),
    ] {
        while let Some(start) = body.find(open) {
            let list_start = start + open.len();
            let Some(len) = body[list_start..].find(close) else {
                break;
            };
            for path in body[list_start..list_start + len].lines() {
                if is_modified {
                    ops.add_modified(path);
                } else {
                    ops.add_read(path);
                }
            }
            body.replace_range(start..list_start + len + close.len(), "");
        }
    }
    (body.trim().to_string(), ops)
}

#[cfg(test)]
#[path = "file_ops_tests.rs"]
mod tests;
