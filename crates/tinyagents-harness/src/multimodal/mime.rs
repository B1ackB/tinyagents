//! MIME detection for image and file attachments.
//!
//! Three signals, consulted in a deliberate order:
//!
//! 1. **The `Content-Type` header**, when one came from a fetch.
//! 2. **The file extension**, when the reference was a path.
//! 3. **Magic bytes.**
//!
//! Images and files order these differently, and the difference is not an
//! oversight. For images the header wins outright: an image server is
//! authoritative about what it served. For files the header wins only if it
//! names a format the allowlist could contain — a great many servers answer
//! `application/octet-stream` for everything, and taking that at face value
//! would degrade every fetched PDF to a metadata-only reference.
//!
//! Office containers are recognized from standard parts in the ZIP central
//! directory before generic ZIP headers are consulted. An extension remains a
//! fallback for incomplete containers; no member contents are extracted.

use std::path::Path;

use super::config::ALLOWED_IMAGE_MIME_TYPES;

/// `true` when `mime` is an image type the provider contract accepts.
pub fn is_allowed_image_mime(mime: &str) -> bool {
    ALLOWED_IMAGE_MIME_TYPES.contains(&mime)
}

/// Detect an image's MIME type: header, then extension, then magic bytes.
pub fn detect_image_mime(
    path: Option<&Path>,
    bytes: &[u8],
    header_content_type: Option<&str>,
) -> Option<String> {
    if let Some(header_mime) = header_content_type.and_then(normalize_content_type) {
        return Some(header_mime);
    }

    if let Some(path) = path
        && let Some(ext) = path.extension().and_then(|value| value.to_str())
        && let Some(mime) = image_mime_from_extension(ext)
    {
        return Some(mime.to_string());
    }

    image_mime_from_magic(bytes).map(ToString::to_string)
}

/// Detect a file's MIME type: Office container parts, known header, extension,
/// magic bytes, then a UTF-8 sniff. Generic ZIP/octet-stream headers defer to
/// Office extensions.
pub fn detect_file_mime(
    path: Option<&Path>,
    bytes: &[u8],
    header_content_type: Option<&str>,
) -> Option<String> {
    if let Some(mime) = office_container_mime(bytes) {
        return Some(mime.to_string());
    }
    // A generic ZIP/octet-stream header must not hide an Office extension.
    if header_content_type
        .and_then(normalize_content_type)
        .is_some_and(|mime| mime == "application/zip" || mime == "application/octet-stream")
        && let Some(ext) = path
            .and_then(|path| path.extension())
            .and_then(|ext| ext.to_str())
        && let Some(mime) = file_mime_from_extension(ext)
        && mime.starts_with("application/vnd.openxmlformats-officedocument.")
    {
        return Some(mime.to_string());
    }
    if let Some(header_mime) = header_content_type.and_then(normalize_content_type)
        && file_mime_known(&header_mime)
    {
        return Some(header_mime);
    }

    if let Some(path) = path
        && let Some(ext) = path.extension().and_then(|value| value.to_str())
        && let Some(mime) = file_mime_from_extension(ext)
    {
        return Some(mime.to_string());
    }

    if let Some(mime) = file_mime_from_magic(bytes) {
        return Some(mime.to_string());
    }

    if let Some(mime) = image_mime_from_magic(bytes) {
        return Some(mime.to_string());
    }
    if looks_like_utf8_text(bytes) {
        return Some("text/plain".to_string());
    }

    None
}

/// Generic attachments preserve explicit media types and recognize local media
/// before the legacy UTF-8 fallback. ZIP/octet-stream still allow Office probing.
pub(super) fn detect_attachment_mime(
    path: &Path,
    bytes: &[u8],
    header: Option<&str>,
) -> Option<String> {
    if let Some(mime) = header.and_then(normalize_content_type)
        && mime != "application/zip"
        && mime != "application/octet-stream"
    {
        return Some(mime);
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    let media = attachment_media_magic(bytes)
        .or_else(|| image_mime_from_magic(bytes))
        .or(match extension.as_deref() {
            Some("wav" | "wave") => Some("audio/wav"),
            Some("mp3") => Some("audio/mpeg"),
            Some("flac") => Some("audio/flac"),
            Some("ogg" | "oga") => Some("audio/ogg"),
            Some("opus") => Some("audio/opus"),
            Some("m4a" | "m4b" | "m4p") => Some("audio/mp4"),
            Some("avif") => Some("image/avif"),
            Some("heic") => Some("image/heic"),
            Some("heif") => Some("image/heif"),
            Some("aac") => Some("audio/aac"),
            Some("mp4" | "m4v") => Some("video/mp4"),
            Some("mov") => Some("video/quicktime"),
            Some("webm") => Some("video/webm"),
            Some("mkv") => Some("video/x-matroska"),
            Some("avi") => Some("video/x-msvideo"),
            _ => None,
        });
    media
        .map(ToString::to_string)
        .or_else(|| detect_file_mime(Some(path), bytes, header))
}

fn attachment_media_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        Some("audio/wav")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"AVI ") {
        Some("video/x-msvideo")
    } else if bytes.starts_with(b"ID3") {
        Some("audio/mpeg")
    } else if bytes.starts_with(b"fLaC") {
        Some("audio/flac")
    } else if bytes.starts_with(b"OggS") {
        Some("audio/ogg")
    } else if bytes.get(4..8) == Some(b"ftyp") {
        match bytes.get(8..12) {
            Some(b"M4A " | b"M4B " | b"M4P ") => Some("audio/mp4"),
            Some(b"qt  ") => Some("video/quicktime"),
            Some(b"avif" | b"avis") => Some("image/avif"),
            Some(b"heic" | b"heix") => Some("image/heic"),
            Some(b"mif1" | b"msf1") => Some("image/heif"),
            Some(b"isom" | b"iso2" | b"mp41" | b"mp42" | b"M4V " | b"avc1" | b"dash") => {
                Some("video/mp4")
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Strip parameters from a `Content-Type` header and lower-case it.
pub fn normalize_content_type(content_type: &str) -> Option<String> {
    let mime = content_type.split(';').next()?.trim().to_ascii_lowercase();
    if mime.is_empty() { None } else { Some(mime) }
}

/// Image MIME for a file extension.
pub fn image_mime_from_extension(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

/// File extension for a known image MIME — the inverse of
/// [`image_mime_from_extension`]. Used when naming a stashed attachment.
pub fn image_ext_from_mime(mime: &str) -> Option<&'static str> {
    match mime {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "image/bmp" => Some("bmp"),
        _ => None,
    }
}

/// Image MIME from magic bytes.
pub fn image_mime_from_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 8 && bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']) {
        return Some("image/png");
    }

    if bytes.len() >= 3 && bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }

    if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Some("image/gif");
    }

    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }

    if bytes.len() >= 2 && bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }

    None
}

/// `true` when a `Content-Type` header names a file format worth trusting over
/// the extension. Deliberately narrow — see the module header.
pub fn file_mime_known(mime: &str) -> bool {
    file_mime_from_extension(mime).is_some()
        || matches!(
            mime,
            "application/pdf"
                | "text/plain"
                | "text/csv"
                | "text/markdown"
                | "application/zip"
                | "application/x-tar"
                | "application/gzip"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/octet-stream"
        )
}

/// File MIME for a file extension.
pub fn file_mime_from_extension(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "pdf" => Some("application/pdf"),
        "txt" => Some("text/plain"),
        "md" | "markdown" => Some("text/markdown"),
        "csv" => Some("text/csv"),
        "zip" => Some("application/zip"),
        "tar" => Some("application/x-tar"),
        "gz" | "tgz" => Some("application/gzip"),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        _ => None,
    }
}

/// File MIME from magic bytes.
///
/// Returns `application/zip` for every OOXML container — see the module header
/// for why the extension has to discriminate.
pub fn file_mime_from_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 5 && bytes.starts_with(b"%PDF-") {
        return Some("application/pdf");
    }

    // OOXML formats (xlsx/docx/pptx) and plain zip all share the PK\x03\x04
    // ZIP local-file-header magic; without parsing the central directory we
    // cannot distinguish them, so callers must rely on the file extension for
    // OOXML vs zip discrimination.
    if bytes.len() >= 4 && bytes.starts_with(&[b'P', b'K', 0x03, 0x04]) {
        return Some("application/zip");
    }

    if bytes.starts_with(b"\x1f\x8b") {
        return Some("application/gzip");
    }
    if bytes.get(257..262) == Some(b"ustar") {
        return Some("application/x-tar");
    }
    None
}

/// Identify standard Office parts using bounded container metadata, without
/// reading document contents. Invalid/incomplete ZIPs retain extension-based
/// detection for compatibility. No archive member is extracted.
fn office_container_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() > 50 * 1024 * 1024 || !bytes.starts_with(b"PK\x03\x04") {
        return None;
    }
    if super::archive::zip_admission::zip_preflight(bytes)
        .ok()?
        .is_some()
    {
        return None;
    }
    let archive = super::archive::zip_admission::open_admitted_zip(bytes).ok()?;
    if archive.len() > 10_000 || archive.index_for_name("[Content_Types].xml").is_none() {
        return None;
    }
    for (part, mime) in [
        (
            "word/document.xml",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "xl/workbook.xml",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "ppt/presentation.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ] {
        if archive.index_for_name(part).is_some() {
            return Some(mime);
        }
    }
    None
}

/// `true` when `mime` names a format whose text is extracted by decoding the
/// bytes directly, with no parser involved.
pub fn is_extractable_text_mime(mime: &str) -> bool {
    matches!(mime, "text/plain" | "text/csv" | "text/markdown")
}

/// Crude UTF-8 sniff: the bytes parse as UTF-8 and contain at least one
/// non-control character. The last-resort fallback that lets an unlabeled
/// `.log` / `.ini` / source file still be recognised as text.
pub fn looks_like_utf8_text(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => text
            .chars()
            .any(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t')),
        Err(_) => false,
    }
}
