//! Lossless durable transcript histories.
//!
//! This module deliberately exposes [`TranscriptMessage`] rather than a
//! provider or harness message. A transcript is an on-disk compatibility
//! boundary: it must preserve native tool calls, malformed raw arguments,
//! usage, thinking content, provider extensions and caller-owned metadata.
//! Converting it through a narrower runtime message here would make a later
//! replay silently lossy. Hosts perform any runtime conversion explicitly at
//! their own boundary.
//!
//! A history handle is bound to a transcript file. Thread and agent discovery
//! belong to [`TranscriptLocator`], because one thread can have several
//! transcript stems (for example a root agent and sub-agents).
//!
//! Every mutation is append-only: a reduced logical context is represented by
//! a `{"kind":"compaction","replacement":[…]}` record, never a destructive
//! rewrite. [`TranscriptHistory::clear`] is therefore an empty compaction.
//!

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::transcript::types::TranscriptMessage;
use fs2::FileExt;

use crate::transcript::{
    SessionAdoption, SessionRef, SessionTranscript, TranscriptMeta, TurnUsage,
    adopt_legacy_session_transcripts, find_latest_transcript, find_root_transcript_for_thread,
    find_root_transcript_for_thread_scoped, read_transcript, resolve_keyed_transcript_path,
    session_stem,
};

/// Upper bound on the compaction generations one session may accumulate.
///
/// Generation resolution probes `{stem}`, `{stem}.g1`, `{stem}.g2` … on disk
/// rather than consulting an index, so it needs a stop condition that holds
/// even if something in the directory is unexpected. A conversation that
/// compacts more than this many times has other problems.
const MAX_GENERATIONS: u32 = 4096;

/// One turn's worth of transcript write, borrowed.
///
/// The fields mirror the transcript writer's turn-append argument list one-for-one and
/// in order, so [`TranscriptHistory::append_turn`]'s forwarding is visually
/// checkable against the format's own signature. Nothing is transformed on the
/// way through; that is the entire correctness claim of this seam and
/// `append_turn_is_byte_identical_to_the_free_function` in the tests pins it.
///
/// `prev` is a field rather than handle state on purpose: the turn path tracks
/// the previously-persisted logical set in memory on `Agent`
/// (`persisted_transcript_messages`) precisely so it never has to re-read a
/// growing file, and a disk re-read is not a faithful substitute — see
/// `FileTranscriptHistory::write_logical_set_locked`.
pub struct TranscriptTurn<'a> {
    /// Logical message set already persisted, for the extension-vs-compaction diff.
    pub prev: &'a [TranscriptMessage],
    /// Logical message set after this turn.
    pub next: &'a [TranscriptMessage],
    /// `_meta` header to append after this turn's lines.
    pub meta: &'a TranscriptMeta,
    /// Usage + provenance attributed to the turn's last assistant row.
    pub turn_usage: Option<&'a TurnUsage>,
    /// Caller-provided request id, stamped on every line of the turn.
    pub request_id: Option<&'a str>,
    /// Tool declarations this ordinary turn was sent with. `None` records
    /// nothing and leaves the previous record in force (for exact-tool turns).
    pub tools: Option<&'a serde_json::Value>,
}

/// Display-only content produced before a turn stopped without a final answer.
///
/// This deliberately uses transcript-neutral fields. It is never added to a
/// model-context replay: the file writer records it as an interrupted message
/// line for the display projection only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptPartial {
    /// Visible assistant text accumulated before the interruption.
    pub content: String,
    /// Optional provider reasoning text associated with the partial.
    pub reasoning_content: Option<String>,
    /// Optional one-based engine iteration associated with the partial.
    pub iteration: Option<u32>,
}

impl TranscriptPartial {
    /// Creates a display-only partial with no provider-specific metadata.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            reasoning_content: None,
            iteration: None,
        }
    }
}

/// The seam a host turn path holds as `Arc<dyn TranscriptHistory>`.
///
/// `append_turn` is deliberately **sync**: `persist_session_transcript` is a
/// sync `&mut self` method and the whole write chain under it is sync, so an
/// async method here would ripple `.await` through the turn loop for no gain.
pub trait TranscriptHistory: TranscriptRead {
    /// Appends one turn, forwarding every argument to the format owner.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()>;

    /// Appends the logical turn and its optional display-only partial as one
    /// history operation.
    ///
    /// Implementors that cannot make the combined mutation atomic must reject
    /// a partial rather than persist either half. Existing implementors which
    /// only support logical turns remain source-compatible through this default.
    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        if partial.is_some() {
            anyhow::bail!("transcript history does not support atomic display partials");
        }
        self.append_turn(turn)
    }

    /// Returns the lossless model-context replay of this transcript.
    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>>;

    /// Appends one durable message while preserving all of its fields.
    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()>;

    /// Replaces the logical model context by appending a compaction record.
    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()>;

    /// Clears the logical model context by appending an empty compaction.
    fn clear(&self) -> anyhow::Result<()>;
}

/// The read half of a bound transcript — the seam the turn path's two resume
/// reads hold.
///
/// Split out of [`TranscriptHistory`] rather than added as one more method on it,
/// for a reason that is not stylistic: a *discovered* transcript can still be a
/// legacy `.md` file (see [`FileTranscriptHistory::opened_at`]), and
/// `append_transcript_turn` writes JSONL. Handing discovery results out as
/// `Arc<dyn TranscriptRead>` makes it impossible to `append_turn` into
/// one by construction, instead of by convention.
///
/// Sync for the same reason [`TranscriptHistory::append_turn`] is: both callers are
/// sync `&mut self` methods on `Agent`.
pub trait TranscriptRead: Send + Sync {
    /// The transcript file this handle is bound to.
    ///
    /// The turn path still needs the concrete path after the read:
    /// `maybe_shadow_read_session_store` takes `&Path`, and the dual-write
    /// mirror derives its record key from `file_stem()`.
    fn path(&self) -> &Path;

    /// The model-context replay of this transcript, `_meta` included, or
    /// `Ok(None)` when the file does not exist.
    ///
    /// Exactly [`read_transcript`], so compaction records have already replaced
    /// the accumulator and `interrupted: true` partials are already skipped —
    /// §3.1's "single most important constraint". Returning the whole
    /// [`SessionTranscript`] rather than messages alone is what lets the shadow
    /// read keep working through this seam.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>>;
}

/// Resolves transcripts by the two keys the turn path actually has, and binds
/// this session's own write handle.
///
/// One injected object covers the whole turn path: both resume reads and the
/// first-write bind. A host holds it as `Option<Arc<dyn TranscriptLocator>>`
/// and falls back to [`FileTranscriptLocator`] built from the *current*
/// `workspace_dir` — lazily, never frozen at build time,
/// because tests reassign `agent.workspace_dir` after `build()` and a
/// build-time locator would silently keep pointing at the old directory.
pub trait TranscriptLocator: Send + Sync {
    /// The durable destination this locator addresses, when it can name one.
    ///
    /// Two locators with equal, `Some` keys resolve every lookup and every
    /// bind to the same place, so a caller comparing bindings may treat them
    /// as interchangeable however they were allocated. `None` means "cannot
    /// say", and such a locator only ever matches itself.
    ///
    /// This exists because the guidance above tells a host to build the
    /// locator lazily from the *current* `workspace_dir` and never freeze it,
    /// which necessarily yields a fresh `Arc` per call. A caller that
    /// identified a locator by allocation would reject the very hosts that
    /// followed that instruction, so it identifies one by this key instead.
    fn destination_key(&self) -> Option<String> {
        None
    }

    /// Newest transcript for `agent_name` in this session's raw subtree,
    /// including the legacy `session_raw/DDMMYYYY/` + `.md` fallback.
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// Newest **root** transcript whose `_meta.thread_id` matches.
    ///
    /// Root-only on purpose: several transcripts share one thread id (every
    /// sub-agent spawned within it does), so a stem-keyed lookup would be
    /// ambiguous.
    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// [`Self::root_for_thread`], additionally scoped to `agent_id` when
    /// given — see
    /// [`transcript::find_root_transcript_for_thread_scoped`](crate::transcript::find_root_transcript_for_thread_scoped)
    /// for why. Defaults to the unscoped lookup so an implementor that never
    /// serves several distinct agents over the same `thread_id` (this test
    /// double, notably) does not have to know about agent scoping at all.
    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        let _ = agent_id;
        self.root_for_thread(thread_id)
    }

    /// Binds (creating on first write) this session's own write handle for
    /// `stem`, with `seed` used only when no file exists yet.
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>>;

    /// The newest generation of `session` that exists, or `session` itself when
    /// none has been written yet.
    ///
    /// A compaction seals a generation and opens the next
    /// ([`Self::begin_generation`]), so the head is the one a resume must load
    /// and append to. The default walks the successor chain through
    /// [`Self::session_exists`]; an implementor with an index may override it.
    fn head_generation(&self, session: &SessionRef) -> SessionRef {
        let mut head = session.clone();
        if !self.session_exists(&head) {
            return head;
        }
        while head.generation < MAX_GENERATIONS {
            let next = head.next_generation();
            if !self.session_exists(&next) {
                break;
            }
            head = next;
        }
        head
    }

    /// Whether `session` has a transcript on disk.
    fn session_exists(&self, session: &SessionRef) -> bool {
        self.read_session_transcript(session).is_some()
    }

    /// Every generation of `session` that exists, oldest first.
    ///
    /// A compaction seals a generation and opens the next, so a long
    /// conversation is a chain rather than one file. The model reads only the
    /// head ([`Self::head_generation`]); a host rendering or exporting the
    /// conversation wants the whole chain. Empty when nothing is written yet.
    fn session_chain(&self, session: &SessionRef) -> Vec<SessionRef> {
        let mut chain = Vec::new();
        let mut generation = session.first_generation();
        while generation.generation <= MAX_GENERATIONS && self.session_exists(&generation) {
            chain.push(generation.clone());
            generation = generation.next_generation();
        }
        chain
    }

    /// Reads `session`'s transcript, or `None` when it has none yet.
    ///
    /// Unlike [`Self::root_for_thread`] this is an exact lookup, not a
    /// newest-wins scan: one session resolves to one file, in every process and
    /// on every launch.
    ///
    /// Defaults to opening the stem the session names through
    /// [`Self::open_stem`] and reading it back. [`Self::open_stem`] alone is
    /// not sufficient — it binds a handle regardless of whether anything has
    /// ever been written there, so this default has to perform the read and
    /// report `None` unless the transcript actually exists, rather than
    /// reporting a handle for a file that was never created. An implementor
    /// with a cheaper existence check (a path probe, an index) should still
    /// override this.
    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        let stem = session_stem(session);
        let handle = self
            .open_stem(&stem, seed_meta_for_discovered(&stem))
            .ok()?;
        match handle.read_session() {
            Ok(Some(_)) => Some(handle as Arc<dyn TranscriptRead>),
            _ => None,
        }
    }

    /// Binds `session`'s own transcript for reading **and** appending.
    ///
    /// This is the method that closes the bug the whole session identity exists
    /// for: resume reads and the subsequent append address the same file, so a
    /// restart extends the conversation instead of re-materialising it into a
    /// fresh stem and orphaning the original.
    fn open_session(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.open_stem(&session_stem(session), seed)
    }

    /// Folds any pre-identity transcripts of `thread_id` into `session`, once.
    ///
    /// A conversation written before session identity existed is spread across
    /// one or more timestamped stems, of which resume only ever loaded the
    /// newest — so its opening turns became unreachable to the model. This
    /// recovers them the first time the session is resumed. Returns `Ok(None)`
    /// when the session already has a transcript or the thread has no legacy
    /// roots, which makes repeat calls harmless.
    ///
    /// Defaults to doing nothing, for locators that are not file-backed.
    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        let _ = (session, thread_id, seed);
        Ok(None)
    }

    /// Appends the display-only `partial` of an interrupted turn to the
    /// newest root transcript of `thread_id` (scoped to `agent_id` when
    /// given); returns whether there was one to append to.
    ///
    /// An interrupted **first** turn has no transcript yet, so there is
    /// nothing to append to and this returns `Ok(false)`. The model-context
    /// replay never includes a partial.
    ///
    /// Defaults to `Ok(false)`, for locators that keep no display partials.
    ///
    /// # Errors
    ///
    /// When a transcript exists but cannot be appended to.
    fn append_interrupted_partial(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let _ = (thread_id, agent_id, partial, request_id);
        Ok(false)
    }

    /// Seals `session` and binds its successor generation.
    ///
    /// Called when a turn's logical message set is no longer an extension of
    /// what is persisted — a compaction. Rewriting the sealed file in place
    /// would destroy the replaced turns; instead generation `n` is left
    /// byte-for-byte as it was and generation `n+1` takes the compacted set as
    /// its opening write, recording `n` as its parent. The conversation stays
    /// fully recoverable by walking the chain even though the model only sees
    /// the head.
    ///
    /// The returned handle is bound but empty: the caller writes the retained
    /// set through the ordinary turn path (`prev: &[]`), so usage, request ids
    /// and display partials are recorded exactly as on any other turn.
    ///
    /// Bounded by [`MAX_GENERATIONS`] — the same limit [`Self::head_generation`]
    /// and [`Self::session_chain`] stop probing at. Enforcing it here, at the
    /// only place a new generation is minted, is what keeps those two bounded
    /// scans complete: without it a chain could grow past what they are
    /// willing to walk, leaving its newest generation undiscoverable by resume
    /// and its head silently stuck on a stale, capped-off generation that the
    /// ordinary append path would then go on writing into.
    ///
    /// Defaults to sealing through [`Self::open_session`] and the trait's own
    /// existence check, which is enough for most implementors; a
    /// file-backed locator overrides it only to reuse an already-resolved
    /// path. Kept non-defaulted before this comment existed as a required
    /// method would have broken every external implementor the moment this
    /// method was added — this default is what restores that compatibility.
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        anyhow::ensure!(
            successor.generation <= MAX_GENERATIONS,
            "session generation limit reached"
        );
        anyhow::ensure!(
            !self.session_exists(&successor),
            "session generation already exists"
        );
        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        let handle = self.open_session(&successor, meta)?;
        Ok((successor, handle))
    }

    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        if let Some(transcript) = self.read_session_transcript(session)
            && let Some(transcript) = transcript.read_session()?
        {
            anyhow::ensure!(
                same_transcript_messages(&transcript.messages, baseline),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        } else {
            anyhow::ensure!(
                baseline.is_empty(),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        }
        self.begin_generation(session, seed)
    }

    /// Forks `session`'s head generation for edit or regenerate, without
    /// erasing history.
    ///
    /// This is [`Self::begin_generation`]'s compaction move, aimed at a
    /// different caller: instead of a summarizer replacing old turns with a
    /// digest, a host wants to edit a past message or regenerate the last
    /// answer. Both need the exact same guarantee compaction already
    /// provides — the current generation is sealed **untouched** on disk
    /// (nothing is written to it; that is the whole point of never rewriting
    /// a sealed file) and the next generation records it as parent — so this
    /// is built on the same primitive rather than a second, parallel one.
    ///
    /// Reads the current [`Self::head_generation`]'s messages, resolves
    /// `cut` against them, seals that head and opens its successor via
    /// [`Self::begin_generation`], and writes the retained prefix into the
    /// successor with [`TranscriptHistory::replace`] — the identical call a
    /// compaction makes to persist its own replacement set. Because the
    /// successor's parent is the sealed head exactly as `begin_generation`
    /// records it, [`Self::session_chain`] walks both generations, so the
    /// full pre-truncation history stays recoverable even though the model
    /// now reads only the truncated head.
    ///
    /// Returns the new generation's [`SessionRef`], its bound handle (already
    /// carrying the truncated messages), and the truncated messages
    /// themselves for the caller's own use (e.g. re-driving the model on the
    /// retained context).
    ///
    /// Fails if `session` has no transcript yet, or if `cut` is
    /// [`TruncateCut::BeforeMessageId`] naming an id absent from the head
    /// generation — silently falling back to some other cut point would risk
    /// truncating the wrong turn.
    fn truncate_into_next_generation(
        &self,
        session: &SessionRef,
        cut: TruncateCut,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(
        SessionRef,
        Arc<dyn TranscriptHistory>,
        Vec<TranscriptMessage>,
    )> {
        let head = self.head_generation(session);
        let head_read = self.read_session_transcript(&head).ok_or_else(|| {
            anyhow::anyhow!(
                "session {} has no transcript to truncate",
                head.session_id()
            )
        })?;
        let transcript = head_read.read_session()?.ok_or_else(|| {
            anyhow::anyhow!(
                "session {} has no transcript to truncate",
                head.session_id()
            )
        })?;
        let keep = cut.resolve(&transcript.messages)?;
        let truncated = transcript.messages[..keep].to_vec();

        let (successor, handle) =
            self.begin_generation_from_baseline(&head, seed, &transcript.messages)?;
        // Same call a compaction makes to persist its own replacement set —
        // see `a_compaction_seals_a_generation_and_leaves_it_untouched`.
        handle.replace(&truncated)?;
        Ok((successor, handle, truncated))
    }
}

/// Where to cut a session's head-generation messages when forking it with
/// [`TranscriptLocator::truncate_into_next_generation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruncateCut {
    /// Keep messages `[0, index)`; drop the message at `index` and everything
    /// after it. Clamped to the message count, so an out-of-range index keeps
    /// every message.
    BeforeIndex(usize),
    /// Keep everything before the message carrying this id. The id must name
    /// a message in the head generation — [`TranscriptMessage::id`] is only
    /// ever set by a host that assigns stable ids, so this is the most
    /// robust key to cut on when the caller has one; unlike an index, it
    /// cannot point at the wrong turn after an earlier truncation shifted
    /// everything else.
    BeforeMessageId(String),
    /// Drop the trailing assistant turn: everything strictly after the last
    /// `role == "user"` message, matching the `role == "assistant"` cutpoint
    /// convention already used across this crate's writer (e.g.
    /// `writer::append_transcript_turn`'s `last_assistant_idx`). Used for
    /// "regenerate the last answer." When there is no user message at all,
    /// every message is dropped.
    LastAssistantTurn,
}

impl TruncateCut {
    /// Resolves this cut to a keep-count (`messages[..keep]` survives)
    /// against the head generation's `messages`.
    fn resolve(&self, messages: &[TranscriptMessage]) -> anyhow::Result<usize> {
        match self {
            TruncateCut::BeforeIndex(index) => Ok((*index).min(messages.len())),
            TruncateCut::BeforeMessageId(id) => messages
                .iter()
                .position(|message| message.id.as_deref() == Some(id.as_str()))
                .ok_or_else(|| anyhow::anyhow!("no message with id `{id}` in the head generation")),
            TruncateCut::LastAssistantTurn => Ok(messages
                .iter()
                .rposition(|message| message.role == "user")
                .map(|index| index + 1)
                .unwrap_or(0)),
        }
    }
}

/// The default [`TranscriptLocator`]: real files under
/// `{workspace_dir}/session_raw`.
///
/// Thin by design — each method wraps exactly one `transcript::` free function
/// and changes nothing about it, so swapping the turn path onto the locator is
/// behaviour-preserving.
pub struct FileTranscriptLocator {
    workspace_dir: PathBuf,
}

impl FileTranscriptLocator {
    /// Builds a locator rooted at `workspace_dir` (i.e. it resolves
    /// `{workspace_dir}/session_raw/...`).
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }

    /// The workspace root this locator resolves `session_raw/` under.
    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }
}

impl TranscriptLocator for FileTranscriptLocator {
    /// The workspace root every lookup and bind resolves under, which is
    /// this locator's only field and therefore its whole identity.
    fn destination_key(&self) -> Option<String> {
        // The resolved `session_raw/` directory, not the workspace root: two
        // workspaces whose `session_raw` is a symlink to one directory share
        // their transcript files, so they must share the turn lock.
        Some(
            symlink_normalized_path(&self.workspace_dir.join("session_raw"))
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_latest_transcript(&self.workspace_dir, agent_name).or_else(|| {
            let dir = self.workspace_dir.join("session_raw");
            let mut best: Option<(String, PathBuf)> = None;
            for entry in fs::read_dir(dir).ok()?.flatten() {
                let candidate = entry.path();
                if candidate.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                    continue;
                }
                let Ok(transcript) = read_transcript(&candidate) else {
                    continue;
                };
                if transcript.meta.parent_session_id.is_none()
                    && (transcript.meta.agent_name == agent_name
                        || transcript.meta.agent_id.as_deref() == Some(agent_name))
                    && best
                        .as_ref()
                        .is_none_or(|(updated, _)| transcript.meta.updated > *updated)
                {
                    best = Some((transcript.meta.updated, candidate));
                }
            }
            best.map(|(_, path)| path).or_else(|| {
                crate::transcript::paths::find_latest_legacy_transcript(
                    &self.workspace_dir,
                    agent_name,
                )
            })
        })?;
        tracing::debug!(
            "[transcript-history] locator latest_for_agent agent={agent_name} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(agent_name),
        )))
    }

    fn append_interrupted_partial(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let Some(path) =
            find_root_transcript_for_thread_scoped(&self.workspace_dir, thread_id, agent_id)
        else {
            return Ok(false);
        };
        if partial.content.is_empty() {
            return Ok(false);
        }
        // Bind the lock handle to the current head so partial writes share
        // the generation and successor-reservation locks with head writers.
        let root_path = path;
        let path = head_generation_path(&root_path);
        let history =
            FileTranscriptHistory::opened_at(path.clone(), seed_meta_for_discovered(thread_id));
        history.with_write_locks(|| {
            anyhow::ensure!(
                head_generation_path(&root_path) == path,
                "transcript head advanced during partial append; retry"
            );
            crate::transcript::append_interrupted_partial(
                &path,
                &partial.content,
                request_id,
                partial.iteration,
                partial.reasoning_content.as_deref(),
            )?;
            tracing::debug!(
                "[transcript-history] locator appended interrupted partial thread={thread_id} chars={} path={}",
                partial.content.len(),
                path.display()
            );
            Ok(true)
        })
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_root_transcript_for_thread(&self.workspace_dir, thread_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread thread={thread_id} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        // Same cross-dir, newest-wins scan as `root_for_thread`, additionally
        // filtered on `_meta.agent_id` so one runtime agent's resume cannot
        // pick up a different agent's transcript for a caller-reused
        // `thread_id` — see `find_root_transcript_for_thread_scoped`.
        let path =
            find_root_transcript_for_thread_scoped(&self.workspace_dir, thread_id, agent_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread_scoped thread={thread_id} \
             agent_id={agent_id:?} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(Arc::new(FileTranscriptHistory::new(
            &self.workspace_dir,
            stem,
            seed,
        )?))
    }

    fn session_exists(&self, session: &SessionRef) -> bool {
        // A direct path probe, not a read: `head_generation` calls this once
        // per generation and only needs to know whether the file is there.
        // `is_file()` rather than `exists()`: a directory, FIFO or other
        // non-regular entry occupying the canonical path must not be
        // reported as an existing generation — reads/appends against it
        // would fail (or, for a directory, silently target the wrong thing)
        // downstream, and `head_generation`'s chain walk would stop at a
        // phantom "generation" that was never actually written.
        resolve_keyed_transcript_path(&self.workspace_dir, &session_stem(session))
            .is_ok_and(|path| path.is_file())
    }

    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        let stem = session_stem(session);
        let path = resolve_keyed_transcript_path(&self.workspace_dir, &stem).ok()?;
        if !path.is_file() {
            return None;
        }
        tracing::debug!(
            "[transcript-history] locator read_session session={stem} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(&stem),
        )))
    }

    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        adopt_legacy_session_transcripts(&self.workspace_dir, session, thread_id, seed)
    }

    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        begin_file_generation_with_baseline(&self.workspace_dir, session, seed, None)
    }

    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        begin_file_generation_with_baseline(&self.workspace_dir, session, seed, Some(baseline))
    }
}

/// Return a stable absolute key even before the workspace has been created.
/// `canonicalize` cannot resolve a path with a missing component, but the
/// in-process turn lock still needs equivalent relative and absolute paths to
/// identify the same destination.
fn absolute_normalized_path(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized.components().count() > 1
                    && !matches!(
                        normalized.components().next_back(),
                        Some(Component::RootDir)
                    )
                {
                    normalized.pop();
                }
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str());
            }
        }
    }
    normalized
}

/// Returns an absolute key with symlinks resolved in every existing prefix.
/// The final workspace directory may not exist yet, so canonicalizing the
/// whole path is not sufficient for locator identity.
fn symlink_normalized_path(path: &Path) -> PathBuf {
    enum OwnedComponent {
        CurDir,
        ParentDir,
        Prefix(std::ffi::OsString),
        RootDir,
        Normal(std::ffi::OsString),
    }

    fn owned_components(path: &Path) -> VecDeque<OwnedComponent> {
        path.components()
            .map(|component| match component {
                Component::CurDir => OwnedComponent::CurDir,
                Component::ParentDir => OwnedComponent::ParentDir,
                Component::Prefix(prefix) => {
                    OwnedComponent::Prefix(prefix.as_os_str().to_os_string())
                }
                Component::RootDir => OwnedComponent::RootDir,
                Component::Normal(name) => OwnedComponent::Normal(name.to_os_string()),
            })
            .collect()
    }

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let fallback = absolute_normalized_path(&absolute);
    let mut resolved = PathBuf::new();
    let mut pending = owned_components(&absolute);
    let mut symlink_hops = 0;

    while let Some(component) = pending.pop_front() {
        match component {
            OwnedComponent::CurDir => {}
            OwnedComponent::ParentDir => {
                if resolved.components().count() > 1
                    && !matches!(resolved.components().next_back(), Some(Component::RootDir))
                {
                    resolved.pop();
                }
            }
            OwnedComponent::Prefix(prefix) => resolved.push(prefix),
            OwnedComponent::RootDir => resolved.push(std::path::MAIN_SEPARATOR.to_string()),
            OwnedComponent::Normal(name) => {
                let candidate = resolved.join(&name);
                match fs::symlink_metadata(&candidate) {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        symlink_hops += 1;
                        if symlink_hops > 40 {
                            return fallback;
                        }
                        let target = match fs::read_link(&candidate) {
                            Ok(target) => target,
                            Err(_) => return fallback,
                        };
                        let mut replacement = if target.is_absolute() {
                            owned_components(&target)
                        } else {
                            owned_components(&resolved)
                        };
                        if !target.is_absolute() {
                            replacement.extend(owned_components(&target));
                        }
                        replacement.extend(pending);
                        pending = replacement;
                        resolved.clear();
                    }
                    _ => resolved.push(name),
                }
            }
        }
    }
    resolved
}

fn begin_file_generation_with_baseline(
    workspace_dir: &Path,
    session: &SessionRef,
    seed: TranscriptMeta,
    baseline: Option<&[TranscriptMessage]>,
) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
    let successor = session.next_generation();
    anyhow::ensure!(
        successor.generation <= MAX_GENERATIONS,
        "session {} has reached the {MAX_GENERATIONS}-generation compaction limit; \
         refusing to create generation {}",
        session.session_id(),
        successor.generation
    );
    let stem = session_stem(&successor);
    let path = resolve_keyed_transcript_path(workspace_dir, &stem)?;
    let parent_path = resolve_keyed_transcript_path(workspace_dir, &session_stem(session))?;
    let mut lock_paths = vec![parent_path.clone(), path.clone()];
    lock_paths.dedup();
    let mut locks = Vec::with_capacity(lock_paths.len());
    for lock_path in &lock_paths {
        let lock = generation_lock(lock_path)?;
        lock.lock_exclusive()?;
        locks.push((lock_path.clone(), lock));
    }
    let parent_lock = locks
        .iter()
        .position(|(locked_path, _)| locked_path == &parent_path)
        .map(|index| locks.remove(index).1);
    let successor_lock = locks
        .iter()
        .position(|(locked_path, _)| locked_path == &path)
        .map(|index| locks.remove(index).1)
        .expect("successor path lock acquired");
    if let Some(baseline) = baseline {
        let current = if parent_path.is_file() {
            read_transcript(&parent_path)?.messages
        } else {
            Vec::new()
        };
        anyhow::ensure!(
            same_transcript_messages(&current, baseline),
            "transcript baseline is stale for {}; reload the session before persisting",
            parent_path.display()
        );
    }
    anyhow::ensure!(
        !path_entry_exists(&path)?,
        "session generation {stem} already exists; refusing to overwrite a sealed transcript"
    );

    let mut meta = seed;
    meta.session_id = Some(successor.session_id());
    meta.parent_session_id = successor.parent_session_id();
    tracing::info!(
        "[transcript-history] sealed session={} and opened generation {} at {}",
        session.session_id(),
        successor.generation,
        path.display()
    );
    let mut history = FileTranscriptHistory::new(workspace_dir, &stem, meta)?;
    // The parent is protected while the successor slot is selected and
    // reserved. Later parent writers take the parent lock first, then fail
    // promptly on the still-absent, reserved successor slot.
    drop(parent_lock);
    *history.generation_reservation.get_mut().unwrap() = Some(GenerationReservation {
        _successor: successor_lock,
    });
    Ok((successor, Arc::new(history)))
}

fn path_entry_exists(path: &Path) -> anyhow::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub(crate) fn same_transcript_messages(
    left: &[TranscriptMessage],
    right: &[TranscriptMessage],
) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.same_row_as(b))
}

/// A placeholder `_meta` for a handle bound to an already-existing transcript.
///
/// `seed_meta` is consulted only when the file is **absent**, and a discovered
/// path exists by definition, so this value is never written. It exists because
/// [`FileTranscriptHistory`] is one type serving both roles; giving read-only
/// handles a `None` meta would mean an `Option` field every write path then has
/// to unwrap for no benefit.
pub(super) fn seed_meta_for_discovered(agent_name: &str) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: agent_name.to_string(),
        agent_id: None,
        agent_type: None,
        dispatcher: String::new(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// A lossless history backed by one `session_raw/{stem}.jsonl` transcript.
///
/// Construct with [`FileTranscriptHistory::new`] (workspace-rooted, i.e.
/// `{workspace}/session_raw/`). The `seed_meta` is used only when the
/// transcript file does not exist yet; for an existing file the authoritative
/// cumulative `_meta` is read back from disk so turn counts and token rollups
/// keep accumulating rather than resetting.
pub struct FileTranscriptHistory {
    /// Fully-resolved transcript file, fixed at construction.
    ///
    /// Resolved eagerly rather than derived per call from a `(workspace, stem)`
    /// pair: the old shape hardcoded `{workspace}/session_raw/`, which is the
    /// **wrong directory** for a canonical session and would have silently
    /// cross-written into the canonical session's transcripts the moment this
    /// handle was wired into the turn path.
    path: PathBuf,
    /// `_meta` used for the very first write, before a file exists.
    seed_meta: TranscriptMeta,
    /// Cross-process reservation held from generation selection through its
    /// first successful write. A crash releases the advisory lock.
    generation_reservation: Mutex<Option<GenerationReservation>>,
    /// Serializes callers sharing this handle while its first-write reservation
    /// is being consumed. The process-wide path lock alone is acquired too late
    /// to prevent two callers from both observing the reservation.
    write_serial: Mutex<()>,
}

struct GenerationReservation {
    _successor: File,
}

impl FileTranscriptHistory {
    fn acquire_write_lock(&self) -> anyhow::Result<Option<(File, File)>> {
        if self.generation_reservation.lock().unwrap().is_some() {
            return Ok(None);
        }
        let stem = self
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let (parent_stem, generation) = match stem.rsplit_once(".g") {
            Some((parent, suffix)) if suffix.parse::<u32>().is_ok() => {
                (parent, suffix.parse::<u32>().unwrap())
            }
            _ => (stem, 0),
        };
        let successor = self
            .path
            .with_file_name(format!("{parent_stem}.g{}.jsonl", generation + 1));
        // All writers and generation creators acquire parent before successor.
        // This lets ordinary writes serialize on the parent while a generation
        // reservation on an absent successor fails promptly instead of
        // blocking the generation's first write.
        let paths = vec![self.path.clone(), successor.clone()];
        let mut locks = Vec::with_capacity(paths.len());
        for lock_path in paths {
            let lock = generation_lock(&lock_path)?;
            if lock_path == successor && !path_entry_exists(&successor)? {
                // An absent successor with a held lock is reserved by a
                // generation creator, which needs its first write to release
                // that reservation. Do not wait here: this parent write would
                // otherwise deadlock the creator permanently.
                lock.try_lock_exclusive().map_err(|error| {
                    anyhow::anyhow!(
                        "successor generation is reserved for {}; retry after it commits: {error}",
                        self.path.display()
                    )
                })?;
            } else {
                // Use the same sorted acquisition order as generation creation
                // for locks that are not an unwritten successor reservation.
                lock.lock_exclusive()?;
            }
            locks.push((lock_path, lock));
        }
        let parent_lock = locks
            .iter()
            .position(|(path, _)| path == &self.path)
            .map(|index| locks.remove(index).1)
            .expect("parent path lock acquired");
        let successor_lock = locks
            .iter()
            .position(|(path, _)| path == &successor)
            .map(|index| locks.remove(index).1)
            .expect("successor path lock acquired");
        Ok(Some((successor_lock, parent_lock)))
    }

    fn finish_generation_reservation(&self, success: bool) -> anyhow::Result<()> {
        if success && let Some(reservation) = self.generation_reservation.lock().unwrap().take() {
            drop(reservation);
        }
        Ok(())
    }

    /// Binds a history handle to `{workspace_dir}/session_raw/{stem}.jsonl`.
    ///
    pub fn new(
        workspace_dir: impl AsRef<Path>,
        stem: &str,
        seed_meta: TranscriptMeta,
    ) -> anyhow::Result<Self> {
        let path = resolve_keyed_transcript_path(workspace_dir.as_ref(), stem)?;
        tracing::debug!(
            "[transcript-history] bound stem={stem} path={}",
            path.display()
        );
        Ok(Self {
            path,
            seed_meta,
            generation_reservation: Mutex::new(None),
            write_serial: Mutex::new(()),
        })
    }

    /// Binds a handle to an **already-discovered** transcript file, verbatim.
    ///
    /// Deliberately does **not** go through `resolve_keyed_transcript_path*`,
    /// which the two stem constructors above use. That helper `create_dir_all`s
    /// its parent and forces a `.jsonl` extension — both wrong for a discovered
    /// path: `find_latest_transcript` can still return a legacy `.md`
    /// file (`read_transcript` routes by extension), and re-resolving would
    /// mangle it into a sibling `.jsonl` that does not exist while creating
    /// stray directories on a pure read.
    ///
    /// Hand the result out as `Arc<dyn TranscriptRead>`, not
    /// `Arc<dyn TranscriptHistory>` — see [`TranscriptRead`]'s doc.
    pub fn opened_at(path: PathBuf, seed_meta: TranscriptMeta) -> Self {
        tracing::debug!(
            "[transcript-history] opened discovered path={}",
            path.display()
        );
        Self {
            path,
            seed_meta,
            generation_reservation: Mutex::new(None),
            write_serial: Mutex::new(()),
        }
    }

    /// This handle's transcript file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the current transcript, or `None` when no file exists yet.
    ///
    /// A missing transcript is the normal first-turn state, not an error.
    fn read(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            return Ok(None);
        }
        read_transcript(&self.path).map(Some)
    }

    /// The logical (model-context) message set currently on disk.
    ///
    /// Routes through [`read_transcript`], so compaction records have already
    /// replaced the accumulator and `interrupted: true` partials are skipped.
    fn persisted(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self.read()?.map(|t| t.messages).unwrap_or_default())
    }

    /// The `_meta` to write: the file's own cumulative meta when it exists,
    /// otherwise this handle's seed.
    ///
    /// Uses the existing durable metadata as a default when a caller does not
    /// caller-computed meta. The **turn path must never route through here** —
    /// it computes `turn_count` and the four token/cost rollups fresh each turn,
    /// and re-reading the file's `_meta` would freeze them at the previous
    /// turn's values, silently breaking `read_thread_usage_summary`.
    fn meta_for_write(&self) -> anyhow::Result<TranscriptMeta> {
        Ok(self
            .read()?
            .map(|t| t.meta)
            .unwrap_or_else(|| self.seed_meta.clone()))
    }
}

/// Opens the stable advisory-lock file associated with a transcript path.
/// The file is intentionally retained after unlock; deleting lock files can
/// split waiters across different inodes and invalidate mutual exclusion.
fn generation_lock(path: &Path) -> anyhow::Result<File> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let lock_root = parent.parent().unwrap_or(parent).join(".transcript-locks");
    let lock_dir = lock_root.join(parent.file_name().unwrap_or_default());
    fs::create_dir_all(&lock_dir)?;
    let lock_name = path.file_name().unwrap_or_default();
    let lock_path = lock_dir.join(lock_name);
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(Into::into)
}

/// A process-wide, per-path mutex serializing the read-modify-write sequence
/// [`FileTranscriptHistory::append`]/`replace`/`clear` run against one file.
///
/// [`SessionRef`]'s own doc names this as a supported shape: two cores in one
/// process sharing a workspace should both see and extend one conversation.
/// Without this, two `FileTranscriptHistory` instances bound to the same
/// path (a legitimate, common way to get there — `open_session` is called
/// fresh per `Session::resume`) can each read the file's current content,
/// compute a diff against that now-stale view, and write. Whichever finishes
/// its own read first computes a `next` that does not extend what the file
/// looks like by the time it *writes* — `append_transcript_turn_with_partial`
/// then reads that mismatch as "the context was reduced" and appends a
/// **compaction record** instead of a plain tail, and a compaction's
/// replacement value is what canonical reads return going forward. The
/// other write's whole contribution becomes unreachable, even though its
/// bytes are still physically on disk as a now-superseded line — a silent
/// lost update, not a crash.
///
/// Keyed by path rather than by `Arc<Mutex<_>>` identity because the two
/// racing instances are typically *separate* `FileTranscriptHistory` values,
/// not a shared handle. Entries are [`Weak`] and swept opportunistically so
/// the registry does not grow for the lifetime of a long-running host: once
/// every in-flight critical section for a path finishes, nothing keeps that
/// path's entry alive, and the next unrelated call reclaims the slot.
fn path_lock(path: &Path) -> Arc<Mutex<()>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.retain(|_, weak| weak.strong_count() > 0);
    if let Some(existing) = locks.get(path).and_then(Weak::upgrade) {
        return existing;
    }
    let fresh = Arc::new(Mutex::new(()));
    locks.insert(path.to_path_buf(), Arc::downgrade(&fresh));
    fresh
}

impl TranscriptRead for FileTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    /// Same call the free-function readers make, on the same path, with the
    /// same return type — so there is nothing left for the round trip to lose.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            tracing::debug!(
                "[transcript-history] read_session absent path={}",
                self.path.display()
            );
            return Ok(None);
        }
        let session = read_transcript(&self.path)?;
        tracing::debug!(
            "[transcript-history] read_session messages={} path={}",
            session.messages.len(),
            self.path.display()
        );
        Ok(Some(session))
    }
}

impl FileTranscriptHistory {
    /// The actual `append_turn` write. Assumes the caller already holds
    /// [`path_lock`] for [`Self::path`] — never call this directly; every
    /// public entry point below acquires the lock once and then routes
    /// through here (and [`Self::append_turn_with_partial_locked`]) so the
    /// lock is taken exactly once per call, never nested (this crate's
    /// `Mutex` is not reentrant).
    fn append_turn_locked(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        self.validate_turn_baseline(turn.prev)?;
        tracing::debug!(
            "[transcript-history] append_turn prev={} next={} usage={} request_id={:?} path={}",
            turn.prev.len(),
            turn.next.len(),
            turn.turn_usage.is_some(),
            turn.request_id,
            self.path.display()
        );
        crate::transcript::writer::append_transcript_turn_with_extras(
            &self.path,
            turn.prev,
            turn.next,
            turn.meta,
            turn.turn_usage,
            turn.request_id,
            crate::transcript::writer::AppendTranscriptExtras {
                partial: None,
                tools: turn.tools,
            },
        )?;
        Ok(())
    }

    /// [`Self::append_turn_locked`]'s counterpart for the display-partial
    /// variant. Same locking contract.
    fn append_turn_with_partial_locked(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        self.validate_turn_baseline(turn.prev)?;
        tracing::debug!(
            "[transcript-history] append_turn_with_partial prev={} next={} partial={} path={}",
            turn.prev.len(),
            turn.next.len(),
            partial.is_some(),
            self.path.display()
        );
        crate::transcript::writer::append_transcript_turn_with_extras(
            &self.path,
            turn.prev,
            turn.next,
            turn.meta,
            turn.turn_usage,
            turn.request_id,
            crate::transcript::writer::AppendTranscriptExtras {
                partial,
                tools: turn.tools,
            },
        )?;
        Ok(())
    }

    fn validate_turn_baseline(&self, prev: &[TranscriptMessage]) -> anyhow::Result<()> {
        if self.generation_reservation.lock().unwrap().is_some() || !self.path.is_file() {
            return Ok(());
        }
        let stem = self
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let (parent_stem, generation) = match stem.rsplit_once(".g") {
            Some((parent, suffix)) if suffix.parse::<u32>().is_ok() => {
                (parent, suffix.parse::<u32>().unwrap())
            }
            _ => (stem, 0),
        };
        let successor_stem = format!("{parent_stem}.g{}", generation + 1);
        let successor_path = self.path.with_file_name(format!("{successor_stem}.jsonl"));
        anyhow::ensure!(
            !successor_path.is_file(),
            "session generation {successor_stem} already exists; reload the session before persisting"
        );
        let disk = self.persisted()?;
        let same = disk.len() == prev.len()
            && disk
                .iter()
                .zip(prev)
                .all(|(left, right)| left.same_row_as(right));
        anyhow::ensure!(
            same,
            "transcript baseline is stale for {}; reload the session before persisting",
            self.path.display()
        );
        Ok(())
    }

    /// Writes `next` as the new logical set, diffing against what is
    /// persisted. Assumes the caller already holds [`path_lock`] for
    /// [`Self::path`] — see [`Self::append_turn_locked`]'s doc for why.
    ///
    /// Routes through [`Self::append_turn_locked`] so every write in this
    /// module — trait-driven and turn-path alike — funnels through one call
    /// to [`append_transcript_turn`], and the extension-vs-compaction
    /// decision stays with the format owner rather than drifting here.
    ///
    /// The `self.persisted()` disk re-read is what the generic trait path has
    /// to do, and is deliberately **not** what the turn path does.
    /// [`read_transcript`] reconstructs `TranscriptMessage`s from line records: the
    /// `failure` / `failure_detail` fields have been lifted out of
    /// `extra_metadata` and turn-usage fields hoisted to top-level line fields.
    /// Feeding that back in as `prev` would make `common_prefix_len` mismatch
    /// at the first such message, so the writer would emit a full compaction
    /// record — re-appending the entire message set — on every single turn.
    fn write_logical_set_locked(&self, next: &[TranscriptMessage]) -> anyhow::Result<()> {
        let prev = self.persisted()?;
        let meta = self.meta_for_write()?;
        self.append_turn_locked(TranscriptTurn {
            prev: &prev,
            next,
            meta: &meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
    }
}

impl FileTranscriptHistory {
    /// Runs `write` holding exactly the locks every other mutation of this
    /// file takes (handle serial, cross-process advisory lock, process-wide
    /// path lock), so an out-of-band writer can never interleave its bytes
    /// with a turn append or a generation being sealed.
    pub(super) fn with_write_locks<R>(
        &self,
        write: impl FnOnce() -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = write();
        drop(os_lock);
        result
    }
}

impl TranscriptHistory for FileTranscriptHistory {
    /// Pure forwarder: every argument reaches the transcript writer's turn append
    /// untouched, so the bytes this writes are identical to what the free
    /// function would have written at the call site.
    ///
    /// This — not [`TranscriptHistory::append`] — is the turn path's own
    /// write call (`Session::persist` in `tinyagents-runtime` calls
    /// [`TranscriptHistory::append_turn_with_partial`] directly), so the
    /// same [`path_lock`] serialization `append`/`replace`/`clear` need
    /// applies here too: two `FileTranscriptHistory` handles bound to the
    /// same successor generation (two compactions racing on
    /// `TranscriptLocator::begin_generation` for one session) would
    /// otherwise both see the file absent and both take the writer's
    /// create-fresh path, and whichever `fs::write` lands last would
    /// silently discard the other's retained set.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = self.append_turn_locked(turn);
        self.finish_generation_reservation(result.is_ok())?;
        drop(os_lock);
        result
    }

    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = self.append_turn_with_partial_locked(turn, partial);
        self.finish_generation_reservation(result.is_ok())?;
        drop(os_lock);
        result
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        self.persisted()
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut next = self.persisted()?;
        next.push(message);
        let result = self.write_logical_set_locked(&next);
        self.finish_generation_reservation(result.is_ok())?;
        drop(os_lock);
        result
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = self.write_logical_set_locked(messages);
        self.finish_generation_reservation(result.is_ok())?;
        drop(os_lock);
        result
    }

    fn clear(&self) -> anyhow::Result<()> {
        let _serial = self.write_serial.lock().unwrap_or_else(|p| p.into_inner());
        let os_lock = self.acquire_write_lock()?;
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.path.exists() {
            self.finish_generation_reservation(true)?;
            drop(os_lock);
            return Ok(());
        }
        let result = self.write_logical_set_locked(&[]);
        self.finish_generation_reservation(result.is_ok())?;
        drop(os_lock);
        result
    }
}

/// The newest generation of the transcript at `path`: the sibling
/// `{base}.g{n}.jsonl` with the largest `n`, or `path` itself when no successor
/// exists. `path` may name any generation of the chain.
fn head_generation_path(path: &Path) -> PathBuf {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return path.to_path_buf();
    };
    let base = match stem.rsplit_once(".g") {
        Some((base, generation)) if generation.parse::<u32>().is_ok() => base,
        _ => stem,
    };
    let Some(parent) = path.parent() else {
        return path.to_path_buf();
    };
    std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let candidate = entry.path();
            if candidate.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                return None;
            }
            let name = candidate.file_stem()?.to_str()?;
            let generation = if name == base {
                0
            } else {
                name.strip_prefix(base)?
                    .strip_prefix(".g")?
                    .parse::<u32>()
                    .ok()?
            };
            Some((generation, candidate))
        })
        .max_by_key(|(generation, _)| *generation)
        .map_or_else(|| path.to_path_buf(), |(_, head)| head)
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
