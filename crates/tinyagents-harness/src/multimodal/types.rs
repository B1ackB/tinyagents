//! Resolved, unconverted attachment data for host-controlled intake.

/// Validated attachment bytes and display metadata. Names are untrusted
/// display strings, never authorized filesystem destinations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAttachment {
    /// Decoded payload; transport gzip has already been removed once.
    pub bytes: Vec<u8>,
    /// Original display name when provided, otherwise `attachment`.
    pub name: String,
    /// Normalized MIME type; undetected bytes use `application/octet-stream`.
    pub mime: String,
    /// Decoded byte count, equal to `bytes.len()`.
    pub size_bytes: usize,
}

/// Whether a generic intake host admits types outside its file allowlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownMimePolicy {
    /// Require the configured allowlist, as legacy file resolution does.
    #[default]
    Reject,
    /// Admit arbitrary MIME types; size, source, and disable gates still apply.
    Accept,
}
