//! Archive inspection types and caller-selected resource caps.

use std::io::Read;

/// Supported archive wire formats; Office ZIP containers are documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    /// ZIP with stored or deflated members.
    Zip,
    /// Uncompressed TAR.
    Tar,
    /// Gzip-compressed TAR.
    TarGzip,
}

impl ArchiveFormat {
    /// Identify archive formats by MIME, signatures, and a display filename.
    /// Office MIME types and extensions take precedence over ZIP container magic.
    pub fn detect(name: &str, mime: &str, bytes: &[u8]) -> Option<Self> {
        let name = name.to_ascii_lowercase();
        let mime = mime
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if mime.starts_with("application/vnd.openxmlformats-officedocument.")
            || [".docx", ".xlsx", ".pptx"]
                .iter()
                .any(|ext| name.ends_with(ext))
        {
            return None;
        }
        if mime == "application/gzip" || bytes.starts_with(b"\x1f\x8b") {
            if is_tar_gzip(bytes) || name.ends_with(".tar.gz") || name.ends_with(".tgz") {
                return Some(Self::TarGzip);
            }
            return None;
        }
        if bytes.get(257..262) == Some(b"ustar") {
            return Some(Self::Tar);
        }
        if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
            return Some(Self::Zip);
        }
        if mime == "application/x-tar" {
            return Some(Self::Tar);
        }
        if mime == "application/zip" {
            return Some(Self::Zip);
        }
        if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
            return Some(Self::TarGzip);
        }
        if name.ends_with(".tar") {
            return Some(Self::Tar);
        }
        if name.ends_with(".zip") {
            return Some(Self::Zip);
        }
        None
    }
}

fn is_tar_gzip(bytes: &[u8]) -> bool {
    let mut header = [0; 512];
    flate2::read::GzDecoder::new(bytes)
        .take(header.len() as u64)
        .read_exact(&mut header)
        .is_ok()
        && &header[257..262] == b"ustar"
}

/// Inspection resource budget. Zero values are honored; values above the
/// safety ceilings are clamped before parsing. Limits concern listing and
/// validation, and never grant filesystem access.
#[derive(Debug, Clone)]
pub struct ArchiveLimits {
    /// Maximum compressed/input bytes (ceiling 50 MiB).
    pub max_input_bytes: usize,
    /// Maximum entries retained (ceiling 10,000).
    pub max_entries: usize,
    /// Maximum UTF-8 bytes per displayed name (ceiling 4096).
    pub max_name_bytes: usize,
    /// Maximum UTF-8 bytes across names (ceiling 1 MiB).
    pub max_total_name_bytes: usize,
    /// Maximum inflated ZIP member bytes or TAR stream bytes (ceiling 50 MiB).
    pub max_decompressed_bytes: usize,
}
impl Default for ArchiveLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 50 * 1024 * 1024,
            max_entries: 200,
            max_name_bytes: 1024,
            max_total_name_bytes: 64 * 1024,
            max_decompressed_bytes: 16 * 1024 * 1024,
        }
    }
}
impl ArchiveLimits {
    pub(crate) fn effective(&self) -> Self {
        Self {
            max_input_bytes: self.max_input_bytes.min(50 * 1024 * 1024),
            max_entries: self.max_entries.min(10_000),
            max_name_bytes: self.max_name_bytes.min(4096),
            max_total_name_bytes: self.max_total_name_bytes.min(1024 * 1024),
            max_decompressed_bytes: self.max_decompressed_bytes.min(50 * 1024 * 1024),
        }
    }
}

/// An entry's declared filesystem type. No links are followed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveEntryKind {
    /// Regular file.
    File,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
    /// Hard link.
    HardLink,
    /// Metadata, device, or another unsupported TAR kind.
    Other,
}
/// Metadata for one entry. Names are untrusted data, never extraction paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    /// Name recorded in the archive, including unsafe paths.
    pub name: String,
    /// Entry's declared kind.
    pub kind: ArchiveEntryKind,
    /// Uncompressed size declared by the header.
    pub declared_size: u64,
}
/// A complete or explicitly truncated listing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ArchiveListing {
    /// Entries inspected in archive order.
    pub entries: Vec<ArchiveEntry>,
    /// Which budget stopped inspection, if any.
    pub truncation: Option<ArchiveTruncation>,
}
/// Resource ceiling that stopped inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveTruncation {
    /// Entry count cap reached.
    Entries,
    /// Individual or aggregate displayed name byte cap reached.
    NameBytes,
    /// Inflated content/stream byte cap reached.
    DecompressedBytes,
}
/// Invalid or oversized archive input.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// Input exceeded the compressed/input byte budget.
    #[error("archive input exceeds byte limit")]
    InputTooLarge,
    /// Invalid ZIP headers or unsupported/encrypted member.
    #[error("invalid archive: {0}")]
    Invalid(String),
    /// Invalid TAR headers, compressed data, or ZIP CRC.
    #[error("invalid archive: {0}")]
    Io(#[from] std::io::Error),
}
