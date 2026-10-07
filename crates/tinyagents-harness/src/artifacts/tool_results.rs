//! Persist oversized tool outputs as artifacts the model can read back.
//!
//! Tool results enter the model context before the provider has seen them, so
//! this is the last cheap point to replace large raw output with a bounded
//! preview. The full, redacted body is written to disk so the host's
//! file-reading tool can inspect it later.
//!
//! Where it is written is the host's choice, through the two constructors:
//!
//! * [`ToolResultArtifactStore::new`] writes under
//!   `<action_dir>/artifacts/tool-results/` and hands the model a path relative
//!   to `action_dir`. Simple, but the files land in the directory the agent is
//!   working in: a coding agent editing a git checkout adds them to the
//!   project, and they show up in its diff.
//! * [`ToolResultArtifactStore::detached`] writes under a storage directory the
//!   host keeps outside the working tree and hands the model the **absolute**
//!   path. Use it whenever the action directory is someone's project.
//!
//! Distinct from [`offload_oversized_result`](super::offload_oversized_result),
//! which offloads a *worker's final result* under `outputs/`: this handles each
//! *tool result* mid-turn under `artifacts/tool-results/<session>/<tool>/`.
//!
//! What the host supplies, because the crate cannot decide it: the
//! [`ArtifactRedactor`] every body passes through before it is stored, the name
//! of its file-reading tool, the wrapper tool a read may arrive inside, and the
//! largest body its reader will open.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use super::policy::{ArtifactRedactor, Redacted};

const ARTIFACT_ROOT: &str = "artifacts/tool-results";

/// A read of a persisted artifact, recognised from a tool call's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRead {
    pub path: String,
    pub offset: usize,
}

/// The artifact a tool call reads, if any.
///
/// `read_tool` and `wrapper_tool` are the host's vocabulary, passed in because
/// tool names are not the crate's to decide (OpenHuman passes `file_read` and
/// `use_skill`).
///
/// Only a `read_tool` call counts, because only its result *is* the stored body:
/// `file_write`, `glob`, `list` or `apply_patch` can name an artifact path too,
/// and their output must still take the normal ladder. The read may arrive
/// wrapped in `wrapper_tool`, reported under that name
/// (`use_skill {"skill":"files","tool":"file_read","args":{"path":…}}`), so
/// `wrapper_tool` — and only it, the one tool whose result *is* the
/// wrapped tool's result — is followed into the tool it runs (#6284). Any
/// other tool that happens to carry `tool`/`args` fields is not a wrapper.
///
/// This recognises the relative `artifacts/tool-results/…` pointer of a store
/// built with [`ToolResultArtifactStore::new`]. A detached store hands out
/// absolute paths; use [`ToolResultArtifactStore::read_target`] for it.
pub fn artifact_read_target(
    tool_name: &str,
    args: &Value,
    read_tool: &str,
    wrapper_tool: &str,
) -> Option<ArtifactRead> {
    read_target_matching(
        tool_name,
        args,
        read_tool,
        wrapper_tool,
        &is_relative_artifact_path,
    )
}

/// A path component match: `artifacts/tool-results-backup/…` shares the
/// prefix but is not the artifact directory.
fn is_relative_artifact_path(path: &str) -> bool {
    path.trim_start_matches("./")
        .strip_prefix(ARTIFACT_ROOT)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// Whether `path` is absolute and lexically inside `dir`. A `..` component
/// disqualifies it outright: the read tool would resolve it somewhere else,
/// and the bytes it returns would not be this store's.
fn is_absolute_path_under(path: &str, dir: &Path) -> bool {
    let path = Path::new(path);
    path.is_absolute()
        && !path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        && path.starts_with(dir)
        && path != dir
}

/// Remove lexical `.` and `..` components without requiring the path to exist.
fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn read_target_matching(
    tool_name: &str,
    args: &Value,
    read_tool: &str,
    wrapper_tool: &str,
    is_artifact: &dyn Fn(&str) -> bool,
) -> Option<ArtifactRead> {
    if tool_name == wrapper_tool {
        let inner_tool = args.get("tool").and_then(Value::as_str)?;
        return read_target_matching(
            inner_tool,
            args.get("args")?,
            read_tool,
            wrapper_tool,
            is_artifact,
        );
    }
    if tool_name != read_tool {
        return None;
    }
    let path = args.get("path").and_then(Value::as_str)?;
    let under_root = is_artifact(path);
    // An absent or null offset starts at 0. A present one that is not a
    // non-negative integer that fits `usize` is not a read `file_read` serves
    // (it rejects it), so it is not an artifact read either; never reinterpret
    // it as 0.
    let offset = match args.get("offset") {
        None | Some(Value::Null) => 0,
        Some(value) => usize::try_from(value.as_u64()?).ok()?,
    };
    under_root.then(|| ArtifactRead {
        path: path.to_string(),
        offset,
    })
}

/// Bound one page of an artifact read to `budget_bytes`, naming the exact
/// `offset` the next read continues from. Never persists: re-persisting a read
/// of an artifact creates a new artifact whose preview is the same bounded
/// head, and the model can loop between previews without ever reaching the
/// body.
pub fn page_artifact_read(
    content: String,
    read: &ArtifactRead,
    budget_bytes: usize,
    read_tool: &str,
) -> String {
    if budget_bytes == 0 {
        return content;
    }
    // A page is useless without its continuation, so a budget too small to
    // carry one is raised to the floor a persisted envelope already takes for
    // the same reason (`MIN_ENVELOPE_ALLOWANCE_BYTES`). Every page therefore
    // fits `max(budget_bytes, MIN_ENVELOPE_ALLOWANCE_BYTES)` and advances.
    let budget_bytes = budget_bytes.max(MIN_ENVELOPE_ALLOWANCE_BYTES);
    if content.len() <= budget_bytes {
        return content;
    }
    let start = read.offset;
    let Some(total) = start.checked_add(content.len()) else {
        // Only reachable with an offset no real read carries (`file_read`
        // rejects offsets past its at-most-10-MiB file). Bound the result but
        // advertise no continuation, since none could advance.
        let cut = floor_char_boundary(&content, budget_bytes);
        return content[..cut].to_string();
    };
    let escaped_path = serde_json::to_string(&read.path).expect("serializing a string cannot fail");
    let with_path = |next: usize| {
        format!(
            "\n\n[artifact page: bytes {start}..{next} of {total}. Continue with {read_tool} {{\"path\":{escaped_path},\"offset\":{next}}}]"
        )
    };
    // Without the path (the caller already has it). At most ~100 bytes, so it
    // always leaves body room under the floor.
    let without_path = |next: usize| {
        format!(
            "\n\n[artifact page: bytes {start}..{next} of {total}. Continue with {read_tool} at \"offset\":{next}]"
        )
    };
    // Sized from the trailer this page will actually carry, not a fixed
    // reservation. `next` never has more digits than `total`, so a trailer
    // rendered with `total` is its longest form.
    let use_path = with_path(total).len() + 4 <= budget_bytes;
    let longest = if use_path {
        with_path(total).len()
    } else {
        without_path(total).len()
    };
    let cut = floor_char_boundary(&content, budget_bytes - longest);
    let trailer = if use_path {
        with_path(start + cut)
    } else {
        without_path(start + cut)
    };
    format!("{}{trailer}", &content[..cut])
}
const AGGREGATE_PREVIEW_BUDGET_BYTES: usize = 512;
/// #4469 item 6: floor for how tightly a persisted `[tool_result_preview]`
/// envelope may be bounded during aggregate spill. `allowed_len` can saturate to
/// `0` (or a handful of bytes) once earlier-spilled results have already consumed
/// the aggregate budget; bounding the envelope to that would return `""` — or a
/// header cut mid-line — discarding the `artifact_path` pointer the model needs
/// to `file_read` the full output. This floor keeps the envelope header (through
/// the `artifact_path` / `read_with` lines) intact even when the raw budget math
/// says zero; `apply_tool_result_budget` retains the head, so the pointer always
/// survives. Slightly overshooting the aggregate budget here is the correct
/// trade — a valid pointer is worth a few hundred bytes.
const MIN_ENVELOPE_ALLOWANCE_BYTES: usize = 512;
pub(super) const TRAILER_RESERVED: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BudgetOutcome {
    original_bytes: usize,
    final_bytes: usize,
    truncated: bool,
}

impl BudgetOutcome {
    fn unchanged(len: usize) -> Self {
        Self {
            original_bytes: len,
            final_bytes: len,
            truncated: false,
        }
    }
}

pub(super) fn apply_tool_result_budget(
    content: String,
    budget_bytes: usize,
) -> (String, BudgetOutcome) {
    let original_bytes = content.len();
    if budget_bytes == 0 || original_bytes <= budget_bytes {
        return (content, BudgetOutcome::unchanged(original_bytes));
    }

    let head_capacity = budget_bytes.saturating_sub(TRAILER_RESERVED).max(1);
    let mut cut = floor_char_boundary(&content, head_capacity);
    if cut == 0 {
        cut = content
            .char_indices()
            .next()
            .map(|(_, c)| c.len_utf8())
            .unwrap_or(0);
    }

    let dropped_bytes = original_bytes.saturating_sub(cut);
    let mut out = String::with_capacity(cut + TRAILER_RESERVED);
    out.push_str(&content[..cut]);
    // Say what was lost, and say what NOT to do about it (#6408).
    //
    // The previous wording ended "re-run with a narrower query to see the
    // rest", which is an instruction to repeat the call. For a listing tool
    // with no narrowing argument in reach — `GITHUB_LIST_PULL_REQUESTS` is the
    // reported case — the model has nothing to narrow, so it re-issues the
    // identical call, gets the identical truncation, and repeats until the
    // successful-repeat tracker halts the run: 8-14 calls per question, the
    // token and quota burn, and an "Incomplete" the user cannot explain.
    //
    // Truncation here is deterministic: the same call returns the same bytes
    // and the same cut. Saying so is what makes the retry stop, and giving the
    // totals lets the model judge whether the head it kept is enough to answer
    // from. Keep the `truncated by tool_result_budget` phrase — it is the
    // grep handle several suites and the runbooks match on.
    out.push_str(&format!(
        "\n\n[… {dropped_bytes} of {original_bytes} bytes truncated by tool_result_budget. \
         Repeating this call returns the same truncation — narrow the request \
         (filter, paginate, or request a smaller range) or answer from the {cut} bytes above …]"
    ));

    let final_bytes = out.len();
    (
        out,
        BudgetOutcome {
            original_bytes,
            final_bytes,
            truncated: true,
        },
    )
}

/// Writes oversized tool results to disk: under
/// `<action_dir>/artifacts/tool-results/` ([`Self::new`]), or under a storage
/// directory outside the working tree ([`Self::detached`]). Detached artifacts
/// live in a dedicated `tool-results` child namespace so pruning cannot touch
/// other host state.
///
/// The host supplies what the crate cannot decide: the redactor every body
/// passes through before it is stored (an artifact on disk is exactly as
/// readable as the result it replaces), the name of its file-reading tool
/// (quoted in the envelope so the model knows how to open the file), and the
/// largest body that tool will open.
#[derive(Debug, Clone)]
pub struct ToolResultArtifactStore {
    /// `action_dir` for [`StoreLayout::ActionRelative`], the storage directory
    /// itself for [`StoreLayout::Detached`].
    root: PathBuf,
    layout: StoreLayout,
    session_key: String,
    redactor: Arc<dyn ArtifactRedactor>,
    read_tool: String,
    max_readable_bytes: u64,
}

/// Where a store keeps its files and how it names them to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreLayout {
    /// `<root>/artifacts/tool-results/…`, named relative to `root` (the
    /// action directory). The original layout.
    ActionRelative,
    /// `<root>/…`, named by absolute path. `root` is a host-owned storage
    /// directory outside the working tree.
    Detached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedToolResult {
    pub output: String,
    pub path: String,
    pub original_bytes: usize,
    pub stored_bytes: usize,
    /// Whether the redactor rewrote anything before the body was stored.
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultArtifactOutcome {
    pub original_bytes: usize,
    pub final_bytes: usize,
    pub persisted: bool,
    pub artifact_path: Option<String>,
}

impl ToolResultArtifactOutcome {
    pub fn unchanged(len: usize) -> Self {
        Self {
            original_bytes: len,
            final_bytes: len,
            persisted: false,
            artifact_path: None,
        }
    }
}

impl ToolResultArtifactStore {
    /// `read_tool` is the host's file-reading tool and `max_readable_bytes` the
    /// largest body it will open; a body whose redacted form exceeds it is
    /// never stored, because the model could not read it back.
    ///
    /// Files go under `<action_dir>/artifacts/tool-results/` and the model is
    /// given a path relative to `action_dir`. When `action_dir` is a project
    /// the agent edits, prefer [`Self::detached`]: these files would otherwise
    /// become part of the project.
    pub fn new(
        action_dir: PathBuf,
        session_key: impl Into<String>,
        redactor: Arc<dyn ArtifactRedactor>,
        read_tool: impl Into<String>,
        max_readable_bytes: u64,
    ) -> Self {
        Self::with_layout(
            action_dir,
            StoreLayout::ActionRelative,
            session_key,
            redactor,
            read_tool,
            max_readable_bytes,
        )
    }

    /// A store that keeps its files in `storage_dir`, outside the working
    /// tree, and names each one to the model by its absolute path.
    ///
    /// The working tree is the agent's to change and, for a coding agent, the
    /// thing it is graded or reviewed on: an artifact written there is a stray
    /// file in the user's project, picked up by `git add -A` and shipped in the
    /// diff. A detached store never writes there. The absolute pointer resolves
    /// the same way from any working directory, so it stays readable when a
    /// turn's action directory changes, and a shell can open it as readily as
    /// the host's read tool can.
    ///
    /// Artifacts are kept under `storage_dir/tool-results/`, an owned namespace
    /// that pruning may remove stale session directories from. The host must
    /// let its read tool open paths under that namespace; that grant is policy
    /// and therefore the host's, not this crate's.
    pub fn detached(
        storage_dir: PathBuf,
        session_key: impl Into<String>,
        redactor: Arc<dyn ArtifactRedactor>,
        read_tool: impl Into<String>,
        max_readable_bytes: u64,
    ) -> Self {
        Self::try_detached(
            storage_dir,
            session_key,
            redactor,
            read_tool,
            max_readable_bytes,
        )
        .expect("detached artifact storage directory must be absolute and UTF-8 representable")
    }

    /// Fallible form of [`Self::detached`]. Returns an error when a relative
    /// root cannot be made absolute or when the root cannot be represented in
    /// the UTF-8 path string passed to the model's read tool.
    pub fn try_detached(
        storage_dir: PathBuf,
        session_key: impl Into<String>,
        redactor: Arc<dyn ArtifactRedactor>,
        read_tool: impl Into<String>,
        max_readable_bytes: u64,
    ) -> anyhow::Result<Self> {
        // The pointer is this path, so it must be absolute: a relative one would
        // resolve against whatever directory the reading tool works in.
        let storage_dir = if storage_dir.is_absolute() {
            storage_dir
        } else {
            std::env::current_dir()?.join(&storage_dir)
        };
        let storage_dir = normalize_absolute_path(&storage_dir.join("tool-results"));
        if storage_dir.to_str().is_none() {
            anyhow::bail!("detached artifact storage path is not valid UTF-8");
        }
        Ok(Self::with_layout(
            storage_dir,
            StoreLayout::Detached,
            session_key,
            redactor,
            read_tool,
            max_readable_bytes,
        ))
    }

    fn with_layout(
        root: PathBuf,
        layout: StoreLayout,
        session_key: impl Into<String>,
        redactor: Arc<dyn ArtifactRedactor>,
        read_tool: impl Into<String>,
        max_readable_bytes: u64,
    ) -> Self {
        Self {
            root,
            layout,
            session_key: sanitize_component(&session_key.into()),
            redactor,
            read_tool: read_tool.into(),
            max_readable_bytes,
        }
    }

    /// The root artifacts are written under: the action directory for a store
    /// from [`Self::new`], the storage directory for one from
    /// [`Self::detached`]. A caller choosing the wrong root produces a pointer
    /// the model cannot dereference, and that is only assertable from outside
    /// (#6483).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this store keeps its files outside the working tree
    /// ([`Self::detached`]).
    pub fn is_detached(&self) -> bool {
        self.layout == StoreLayout::Detached
    }

    /// The directory holding one subdirectory per session.
    fn sessions_dir(&self) -> PathBuf {
        match self.layout {
            StoreLayout::ActionRelative => self.root.join(ARTIFACT_ROOT),
            StoreLayout::Detached => self.root.clone(),
        }
    }

    /// The artifact a tool call reads, if it reads one of *this* store's
    /// artifacts.
    ///
    /// Like [`artifact_read_target`], which it extends: a relative
    /// `artifacts/tool-results/…` path is still recognised, so a transcript
    /// written before a host moved to a detached store keeps paging correctly,
    /// and a detached store additionally recognises an absolute path inside its
    /// storage directory, which is the pointer it hands out.
    pub fn read_target(
        &self,
        tool_name: &str,
        args: &Value,
        wrapper_tool: &str,
    ) -> Option<ArtifactRead> {
        read_target_matching(tool_name, args, &self.read_tool, wrapper_tool, &|path| {
            is_relative_artifact_path(path)
                || (self.is_detached() && is_absolute_path_under(path, &self.root))
        })
    }

    /// The host's file-reading tool, as quoted in every envelope.
    pub fn read_tool(&self) -> &str {
        &self.read_tool
    }

    /// Delete artifact directories for sessions other than this one that have
    /// not been touched within `max_age`.
    ///
    /// Nothing else removes artifacts (from the action workspace, or from a
    /// detached store's storage directory), so without a bound the directory
    /// grows for the life of the install — a connector returning 17-65 KB across 8-14 calls per question
    /// (#6408) writes a file per call. Trading a token-burn bug for a
    /// disk-growth bug is not a fix.
    ///
    /// Pruning on session START rather than on session end is deliberate. There
    /// is no single point where a session host ends — cached agents are evicted
    /// on fingerprint mismatch or poisoning, and a crash ends a session with no
    /// hook at all — so an end-of-session sweep would miss exactly the runs most
    /// likely to have left artifacts behind. An age sweep at start is
    /// self-healing instead: whatever the last run did, the next one tidies it.
    ///
    /// The current session is never pruned, no matter its age, so a long-lived
    /// session cannot delete artifacts the model may still be reading back.
    ///
    /// Best-effort by design: a failure here must not fail a turn. The caller
    /// logs and carries on, because the worst case is disk left uncollected,
    /// which the next session retries.
    pub fn prune_stale_sessions(&self, max_age: std::time::Duration) -> std::io::Result<u32> {
        let root = self.sessions_dir();
        let entries = match std::fs::read_dir(&root) {
            Ok(entries) => entries,
            // No artifact root yet is the common case on a first run.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };

        let now = std::time::SystemTime::now();
        let mut removed = 0;
        for entry in entries.flatten() {
            let name = entry.file_name();
            // Never prune the directory this store is actively writing into.
            if name.to_str() == Some(self.session_key.as_str()) {
                continue;
            }
            if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                continue;
            }
            let stale = newest_modified(&entry.path())
                .ok()
                .and_then(|modified| now.duration_since(modified).ok())
                .is_some_and(|age| age > max_age);
            if stale && std::fs::remove_dir_all(entry.path()).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// The pointer the model is given for one artifact: relative to the
    /// action directory for [`Self::new`], absolute for [`Self::detached`].
    pub fn path_for_read_tool(&self, tool_name: &str, call_id: Option<&str>) -> String {
        let relative = self.session_relative_path(tool_name, call_id);
        match self.layout {
            StoreLayout::ActionRelative => format!("{ARTIFACT_ROOT}/{relative}"),
            StoreLayout::Detached => self.root.join(relative).to_string_lossy().into_owned(),
        }
    }

    /// `<session>/<tool>/<call>.txt`, the part of an artifact's path below the
    /// sessions directory.
    fn session_relative_path(&self, tool_name: &str, call_id: Option<&str>) -> String {
        let call = call_id
            .map(sanitize_component)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(random_call_id);
        format!(
            "{}/{}/{}.txt",
            self.session_key,
            sanitize_component(tool_name),
            call
        )
    }

    /// Store `content` and return an envelope previewing it. When `content`
    /// would be unreadable once sanitized (see [`readable_body`]) and a
    /// `fallback` is given, the fallback is stored instead; when neither fits,
    /// this errors and the caller truncates inline rather than writing an
    /// artifact nobody can read.
    async fn persist(
        &self,
        tool_name: &str,
        call_id: Option<&str>,
        content: &str,
        fallback: Option<&str>,
        preview_budget_bytes: usize,
        reason: &str,
    ) -> anyhow::Result<PersistedToolResult> {
        let (content, sanitized) = readable_body(
            content,
            fallback,
            self.max_readable_bytes,
            self.redactor.as_ref(),
            &self.read_tool,
        )?;
        let read_tool = self.read_tool.as_str();
        let pointer = self.path_for_read_tool(tool_name, call_id);
        let absolute_path = match self.layout {
            StoreLayout::ActionRelative => self.root.join(&pointer),
            StoreLayout::Detached => PathBuf::from(&pointer),
        };
        assert_within_root(&self.root, &absolute_path)?;
        if let Some(parent) = absolute_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
            let canonical_root = tokio::fs::canonicalize(&self.root).await?;
            let canonical_parent = tokio::fs::canonicalize(parent).await?;
            if !canonical_parent.starts_with(&canonical_root) {
                anyhow::bail!(
                    "tool-result artifact parent escaped its root: {}",
                    parent.display()
                );
            }
        }
        tokio::fs::write(&absolute_path, sanitized.text.as_bytes()).await?;
        let relative_path = pointer;
        // Keeps the action-relative envelope byte-identical to what it was
        // before detached stores existed (`contract_expected.txt`).
        let location_note = match self.layout {
            StoreLayout::ActionRelative => {
                "Full scrubbed output was persisted under the action workspace."
            }
            StoreLayout::Detached => {
                "Full scrubbed output was persisted outside the working tree, at the absolute artifact_path above (not part of the project)."
            }
        };

        let (preview, preview_outcome) =
            apply_tool_result_budget(sanitized.text.clone(), preview_budget_bytes);
        let redaction_note = if sanitized.changed {
            " Credential/PII redaction was applied before storage and preview exposure."
        } else {
            ""
        };
        let truncation_note = if preview_outcome.truncated {
            format!(
                " Preview is bounded; {} stored bytes are available via {read_tool}.",
                sanitized.text.len()
            )
        } else {
            String::new()
        };

        let escaped_pointer =
            serde_json::to_string(&relative_path).expect("serializing a string cannot fail");
        let envelope = format!(
            "[tool_result_preview]\n\
             tool: {tool_name}\n\
             reason: {reason}\n\
             original_bytes: {}\n\
             stored_bytes: {}\n\
             artifact_path: {relative_path}\n\
             read_with: {read_tool} {{\"path\":{escaped_pointer}}} (a long read returns one page and names the \"offset\" to continue from)\n\
             notes: {location_note}{redaction_note}{truncation_note}\n\n\
             [preview]\n{preview}",
            content.len(),
            sanitized.text.len(),
        );

        Ok(PersistedToolResult {
            output: envelope,
            path: relative_path,
            original_bytes: content.len(),
            stored_bytes: sanitized.text.len(),
            redacted: sanitized.changed,
        })
    }
}

/// The body to store: `primary` if its sanitized form fits `limit`, else
/// `fallback` if *its* sanitized form does, else an error. Both checks are on
/// the sanitized size, the bytes actually written, because redaction can grow a
/// body (`+15551234567` becomes `[REDACTED_PII_PHONE]`): a raw body under the
/// limit can still produce an artifact the read tool refuses to open.
fn readable_body<'a>(
    primary: &'a str,
    fallback: Option<&'a str>,
    limit: u64,
    redactor: &dyn ArtifactRedactor,
    read_tool: &str,
) -> anyhow::Result<(&'a str, Redacted)> {
    let sanitized = redactor.redact(primary);
    if sanitized.text.len() as u64 <= limit {
        return Ok((primary, sanitized));
    }
    if let Some(fallback) = fallback {
        let sanitized_fallback = redactor.redact(fallback);
        if sanitized_fallback.text.len() as u64 <= limit {
            return Ok((fallback, sanitized_fallback));
        }
    }
    anyhow::bail!(
        "tool result would not be readable once stored: {} sanitized bytes exceed the {limit}-byte {read_tool} limit",
        sanitized.text.len()
    )
}

/// Persist an over-budget result and return its envelope.
///
/// `full_output` is the tool's output before any earlier stage rewrote it
/// (summarizer, TokenJuice). When given, *that* is what gets stored, so the
/// artifact holds what the tool returned rather than a compacted copy of it;
/// `content` still decides whether the budget was exceeded and is what the
/// model would otherwise have seen.
pub async fn apply_per_result_persistence(
    content: String,
    full_output: Option<String>,
    store: Option<&ToolResultArtifactStore>,
    tool_name: &str,
    call_id: Option<&str>,
    budget_bytes: usize,
) -> (String, ToolResultArtifactOutcome) {
    let original_bytes = content.len();
    if budget_bytes == 0 || original_bytes <= budget_bytes {
        return (
            content,
            ToolResultArtifactOutcome::unchanged(original_bytes),
        );
    }

    if let Some(store) = store {
        match store
            .persist(
                tool_name,
                call_id,
                full_output.as_deref().unwrap_or(&content),
                full_output.as_ref().map(|_| content.as_str()),
                budget_bytes,
                "per-result budget exceeded",
            )
            .await
        {
            Ok(persisted) => {
                let envelope_allowance = budget_bytes.max(envelope_budget_floor(&persisted.output));
                let (output, final_bytes) =
                    bound_text_to_budget(persisted.output, envelope_allowance);
                if final_bytes >= original_bytes {
                    // #4469 item 9: this branch does NOT fall back to inline
                    // truncation — the envelope is returned regardless, because it
                    // carries the `artifact_path` pointer to the full stored output
                    // (worth keeping even when the preview text nets no byte saving
                    // vs. the raw result). Log it as an observation only.
                    tracing::debug!(
                        "[agent][tool-result-artifacts] persisted envelope not smaller than raw result tool={} original_bytes={} final_bytes={} budget_bytes={} -- keeping envelope for its artifact_path pointer",
                        tool_name,
                        original_bytes,
                        final_bytes,
                        budget_bytes
                    );
                }
                tracing::info!(
                    "[agent][tool-result-artifacts] persisted oversized tool result tool={} original_bytes={} stored_bytes={} path={} redacted={}",
                    tool_name,
                    persisted.original_bytes,
                    persisted.stored_bytes,
                    persisted.path,
                    persisted.redacted
                );
                return (
                    output,
                    ToolResultArtifactOutcome {
                        // The size of what was stored, which `full_output` can
                        // make larger than `content`; the artifact index and its
                        // contents list read this number.
                        original_bytes: persisted.original_bytes,
                        final_bytes,
                        persisted: true,
                        artifact_path: Some(persisted.path),
                    },
                );
            }
            Err(err) => {
                tracing::warn!(
                    "[agent][tool-result-artifacts] persist failed tool={} original_bytes={} err={} — falling back to inline truncation",
                    tool_name,
                    original_bytes,
                    err
                );
            }
        }
    }

    // Reached two ways, and only one of them was audible: a persist that FAILED
    // warns just above, but a run with no artifact store configured falls
    // through to here silently — the oversized tail is discarded with nothing
    // recording that it happened. Say so, so "where did the rest of my search
    // result go" is answerable from the logs rather than by reading this
    // function.
    if store.is_none() {
        tracing::info!(
            "[agent][tool-result-artifacts] no artifact store configured; truncating oversized tool result inline tool={} original_bytes={} budget_bytes={} — the tail is discarded, not recoverable",
            tool_name,
            original_bytes,
            budget_bytes
        );
    }
    let (output, BudgetOutcome { final_bytes, .. }) =
        apply_tool_result_budget(content, budget_bytes);
    (
        output,
        ToolResultArtifactOutcome {
            original_bytes,
            final_bytes,
            persisted: false,
            artifact_path: None,
        },
    )
}

pub async fn spill_aggregate_tool_results(
    results: &mut [tinytools_agent::dialect::ToolOutcome],
    store: Option<&ToolResultArtifactStore>,
    budget_bytes: usize,
) {
    if budget_bytes == 0 {
        return;
    }
    let Some(store) = store else {
        return;
    };

    let mut total: usize = results.iter().map(|result| result.output.len()).sum();
    if total <= budget_bytes {
        return;
    }

    let mut indexes: Vec<usize> = (0..results.len()).collect();
    indexes.sort_by_key(|idx| std::cmp::Reverse(results[*idx].output.len()));

    for idx in indexes {
        if total <= budget_bytes {
            break;
        }
        let original = results[idx].output.clone();
        let original_len = original.len();
        let allowed_len = budget_bytes.saturating_sub(total.saturating_sub(original_len));
        let persisted_output = if looks_like_preview_envelope(&original) {
            Ok(PersistedToolResult {
                output: original.clone(),
                path: "<existing-preview>".to_string(),
                original_bytes: original_len,
                stored_bytes: original_len,
                redacted: false,
            })
        } else {
            store
                .persist(
                    &results[idx].name,
                    results[idx].tool_call_id.as_deref(),
                    &original,
                    None,
                    allowed_len.min(AGGREGATE_PREVIEW_BUDGET_BYTES),
                    "aggregate tool-result budget exceeded",
                )
                .await
        };
        match persisted_output {
            Ok(persisted) => {
                // #4469 item 6: never bound the preview envelope below the minimum
                // that preserves its `[tool_result_preview]` header + artifact
                // pointer — `allowed_len` can be 0 here, which would blank the
                // result and strip the `artifact_path` the model reads to recover
                // the full output.
                // `apply_tool_result_budget` keeps a head of budget minus its
                // trailer reserve. Include the complete rendered pointer/header
                // in that head, even when the path is an unusually long absolute
                // detached pointer.
                let envelope_allowance = allowed_len.max(envelope_budget_floor(&persisted.output));
                let (output, final_bytes) =
                    bound_text_to_budget(persisted.output, envelope_allowance);
                total = total
                    .saturating_sub(original_len)
                    .saturating_add(final_bytes);
                tracing::info!(
                    "[agent][tool-result-artifacts] aggregate spill tool={} original_bytes={} final_bytes={} total_bytes={} path={}",
                    results[idx].name,
                    original_len,
                    final_bytes,
                    total,
                    persisted.path
                );
                results[idx].output = output;
            }
            Err(err) => {
                tracing::warn!(
                    "[agent][tool-result-artifacts] aggregate spill failed tool={} bytes={} err={} -- falling back to inline budget trim",
                    results[idx].name,
                    original_len,
                    err
                );
                let (output, final_bytes) = bound_text_to_budget(original, allowed_len);
                total = total
                    .saturating_sub(original_len)
                    .saturating_add(final_bytes);
                results[idx].output = output;
            }
        }
    }
}

fn looks_like_preview_envelope(value: &str) -> bool {
    value.starts_with("[tool_result_preview]\n")
}

fn bound_text_to_budget(content: String, budget_bytes: usize) -> (String, usize) {
    if budget_bytes == 0 {
        return (String::new(), 0);
    }
    let (mut output, BudgetOutcome { final_bytes, .. }) =
        apply_tool_result_budget(content, budget_bytes);
    if final_bytes <= budget_bytes {
        return (output, final_bytes);
    }
    let cut = floor_char_boundary(&output, budget_bytes);
    output.truncate(cut);
    let final_bytes = output.len();
    (output, final_bytes)
}

fn envelope_budget_floor(envelope: &str) -> usize {
    let header_len = envelope
        .find("\nnotes:")
        .map(|end| end + 1)
        .unwrap_or(envelope.len());
    MIN_ENVELOPE_ALLOWANCE_BYTES.max(header_len.saturating_add(TRAILER_RESERVED + 1))
}

/// Round a byte index DOWN to the nearest UTF-8 character boundary.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut end = index;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// 32 lowercase hex characters, unique per call within and across processes for
/// naming an artifact whose tool call carried no id.
fn random_call_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let salt = COUNTER.fetch_add(1, Ordering::Relaxed);
    let half = |extra: u64| {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(nanos);
        hasher.write_u64(salt);
        hasher.write_u64(extra);
        hasher.finish()
    };
    format!("{:016x}{:016x}", half(1), half(2))
}

fn sanitize_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().min(80));
    for ch in value.chars().take(80) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn assert_within_root(root: &Path, path: &Path) -> anyhow::Result<()> {
    if path.starts_with(root) {
        return Ok(());
    }
    anyhow::bail!(
        "tool-result artifact path escaped its root: {}",
        path.display()
    );
}

/// Return the newest modification time in a session tree without following
/// symlinks. Nested tool writes must keep their containing session alive.
fn newest_modified(path: &Path) -> std::io::Result<std::time::SystemTime> {
    let metadata = std::fs::symlink_metadata(path)?;
    let mut newest = metadata.modified()?;
    if metadata.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_symlink() {
                continue;
            }
            if let Ok(modified) = newest_modified(&entry.path()) {
                newest = newest.max(modified);
            }
        }
    }
    Ok(newest)
}

#[cfg(test)]
#[path = "tool_results_tests.rs"]
mod test;
