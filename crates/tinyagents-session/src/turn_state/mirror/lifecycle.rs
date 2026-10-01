//! Snapshot flush, interrupted-turn finalization, and the partial-answer
//! carry-over into the session transcript.

use super::TurnStateMirror;
use crate::turn_state::types::TurnLifecycle;

pub(super) const MIRROR_LOG_PREFIX: &str = "[threads:turn_state:mirror]";

impl TurnStateMirror {
    /// Mark the turn as `Interrupted` on the in-memory snapshot and
    /// flush. Called when the bridge exits without a `TurnCompleted`
    /// event (i.e. the agent loop errored out).
    pub fn finish(mut self) {
        if self.turn_completed {
            return;
        }
        // The turn driver settles this snapshot the moment the turn ends
        // (`TurnStateStore::settle_turn`), because this bridge can outlive its
        // turn by minutes — it only exits when its progress sender drops, and
        // for a cached per-thread session that waits for the *next* turn. If the
        // driver already recorded a terminal outcome, keep it: a late blind
        // `Interrupted` here would raise a retry banner on a turn that actually
        // succeeded and whose reply was delivered. The completion check and the
        // `Interrupted` write are one store operation, so a settle landing in
        // between cannot be overwritten.
        self.state.lifecycle = TurnLifecycle::Interrupted;
        self.state.active_tool = None;
        self.state.active_subagent = None;
        self.state.updated_at = chrono::Utc::now().to_rfc3339();
        match self.store.put_unless_completed(&self.state) {
            // Only a genuinely interrupted turn has a partial answer to carry
            // over.
            Ok(true) => self.persist_interrupted_partial(),
            // A settled turn's reply was already delivered and persisted;
            // appending it again would duplicate it in the session transcript.
            Ok(false) => tracing::debug!(
                "{MIRROR_LOG_PREFIX} turn already completed thread={} request={} — keeping it",
                self.state.thread_id,
                self.state.request_id
            ),
            // The stored outcome cannot be read, so it cannot be ruled out that
            // the turn completed: neither overwrite it nor append a partial.
            Err(err) => tracing::warn!(
                "{MIRROR_LOG_PREFIX} could not finalize turn thread={} request={}: {err} — leaving the stored snapshot as is",
                self.state.thread_id,
                self.state.request_id
            ),
        }
    }

    /// Append the partial streamed answer of an interrupted turn to the session
    /// transcript so the derived display view (Phase B) can surface it even
    /// after the live turn_state snapshot is gone. Display-only: the
    /// model-context reader skips `interrupted:true` lines.
    ///
    /// Guard: the root transcript file must already exist. An interrupted
    /// **first** turn has no session file yet (the harness has not persisted a
    /// turn), so there is nothing to append to — that case stays recoverable
    /// from the turn_state snapshot alone, as today. We log and skip it.
    fn persist_interrupted_partial(&self) {
        let partial = self.state.streaming_text.trim();
        if partial.is_empty() {
            return;
        }
        let thread_id = self.state.thread_id.trim();
        if thread_id.is_empty() {
            return;
        }
        let workspace_dir = self.store.workspace_dir();
        let Some(path) =
            crate::transcript::find_root_transcript_for_thread(workspace_dir, thread_id)
        else {
            tracing::debug!(
                "{MIRROR_LOG_PREFIX} no root transcript for thread={thread_id} yet — leaving interrupted partial ({} chars) in turn_state snapshot only",
                partial.len()
            );
            return;
        };
        let request_id = if self.state.request_id.is_empty() {
            None
        } else {
            Some(self.state.request_id.as_str())
        };
        let thinking = self.state.thinking.trim();
        let reasoning = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };
        match crate::transcript::append_interrupted_partial(
            &path,
            partial,
            request_id,
            Some(self.state.iteration),
            reasoning,
        ) {
            Ok(()) => tracing::debug!(
                "{MIRROR_LOG_PREFIX} appended interrupted partial ({} chars, thinking={} chars) for thread={thread_id} request_id={} to {}",
                partial.len(),
                thinking.len(),
                self.state.request_id,
                path.display()
            ),
            Err(err) => tracing::warn!(
                "{MIRROR_LOG_PREFIX} failed to append interrupted partial for thread={thread_id}: {err}"
            ),
        }
    }

    /// Write the in-memory snapshot through to the store. Best-effort: a failed
    /// write is logged, never fatal.
    ///
    /// A non-terminal flush never replaces a snapshot the turn driver already
    /// settled as `Completed`: the bridge can outlive its turn, and a late
    /// boundary flush would otherwise resurrect a finished turn.
    pub fn flush(&mut self) {
        let result = match self.state.lifecycle {
            TurnLifecycle::Completed => self.store.put(&self.state),
            _ => self.store.put_unless_completed(&self.state).map(|_| ()),
        };
        if let Err(err) = result {
            tracing::warn!(
                "{MIRROR_LOG_PREFIX} failed to persist snapshot for thread={}: {err}",
                self.state.thread_id
            );
        }
    }
}
