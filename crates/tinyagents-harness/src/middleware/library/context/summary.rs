//! How [`ContextCompressionMiddleware`] turns the messages a compaction folds
//! into one summary: split-turn aware, with file-operation lists appended.

use super::*;
use crate::summarization::{
    append_file_sections, extract_file_operations, split_file_sections, split_turn_start,
    summarize_split_turn,
};

impl ContextCompressionMiddleware {
    /// Summarizes `to_summarize` (the messages folded away) for a compaction
    /// that keeps `to_keep` verbatim.
    ///
    /// * A cut inside a turn gives that turn's prefix its own summary request
    ///   ([`summarize_split_turn`]) instead of size-halving.
    /// * Unless disabled, the files read and modified by the folded tool calls
    ///   are appended as `<read-files>` / `<modified-files>` sections, unioned
    ///   with the lists the previous summary carried. The previous summary
    ///   reaches the summarizer without its lists, and any the summarizer
    ///   echoes are dropped, so each list appears exactly once.
    pub(in crate::middleware::library) async fn summarize_batch(
        &self,
        to_summarize: &[Message],
        to_keep: &[Message],
        previous_summary: Option<String>,
    ) -> Result<SummaryRecord> {
        let mut ops = crate::summarization::FileOperations::default();
        let previous_summary = match (&self.file_ops, previous_summary) {
            (Some(_), Some(previous)) => {
                let (body, carried) = split_file_sections(&previous);
                ops = carried;
                Some(body)
            }
            (_, previous) => previous,
        };
        let mut record = summarize_split_turn(
            self.summarizer.as_ref(),
            to_summarize,
            self.split_turn_prefix
                .then(|| split_turn_start(to_summarize, to_keep))
                .flatten(),
            self.max_turn_tokens.unwrap_or(u64::MAX),
            previous_summary,
            crate::token_estimation::estimate_message_tokens,
        )
        .await?;
        if let Some(extractor) = &self.file_ops {
            ops.merge(&extract_file_operations(to_summarize, extractor.as_ref()));
            let (body, _) = split_file_sections(&record.summary.text());
            record.summary = Message::system(append_file_sections(&body, &ops));
        }
        Ok(record)
    }
}
