//! Allocation-free ZIP central-directory admission before the eager ZIP parser.

use super::archive_types::{ArchiveError, ArchiveTruncation};

fn invalid() -> ArchiveError {
    ArchiveError::Invalid("invalid ZIP central directory".into())
}

fn number(bytes: &[u8], offset: usize, width: usize) -> Result<u64, ArchiveError> {
    let slice = bytes
        .get(offset..offset.checked_add(width).ok_or_else(invalid)?)
        .ok_or_else(invalid)?;
    Ok(slice
        .iter()
        .enumerate()
        .fold(0, |value, (i, byte)| value | ((*byte as u64) << (8 * i))))
}

fn footer_offset(bytes: &[u8]) -> Result<usize, ArchiveError> {
    let end = (bytes.len().saturating_sub(65_557)..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| {
            bytes.get(i..i + 4) == Some(b"PK\x05\x06")
                && number(bytes, i + 20, 2).is_ok_and(|len| i + 22 + len as usize == bytes.len())
        })
        .ok_or_else(invalid)?;
    Ok(end)
}

/// Verify declared counts against available records and bound all metadata
/// before `ZipArchive::new` allocates. The safety ceilings are independent of
/// the caller's smaller listing budget so small listings retain their prefix.
/// Archives above a safety ceiling return an empty, truncated listing.
pub(super) fn zip_preflight(bytes: &[u8]) -> Result<Option<ArchiveTruncation>, ArchiveError> {
    let end = footer_offset(bytes)?;
    let mut count = number(bytes, end + 10, 2)?;
    if number(bytes, end + 8, 2)? != count
        || number(bytes, end + 4, 2)? != 0
        || number(bytes, end + 6, 2)? != 0
    {
        return Err(invalid());
    }
    let mut size = number(bytes, end + 12, 4)?;
    let mut directory_end = end;
    let mut relative = number(bytes, end + 16, 4)?;
    if end >= 20 && bytes.get(end - 20..end - 16) == Some(b"PK\x06\x07") {
        let offset = usize::try_from(number(bytes, end - 12, 8)?).map_err(|_| invalid())?;
        if bytes.get(offset..offset.saturating_add(4)) != Some(b"PK\x06\x06") {
            return Err(invalid());
        }
        let record_size = number(bytes, offset + 4, 8)?;
        if record_size < 44
            || (offset as u64)
                .checked_add(12)
                .and_then(|v| v.checked_add(record_size))
                != Some((end - 20) as u64)
        {
            return Err(invalid());
        }
        if record_size > 4 * 1024 * 1024 {
            return Ok(Some(ArchiveTruncation::NameBytes));
        }
        count = number(bytes, offset + 32, 8)?;
        if number(bytes, offset + 24, 8)? != count
            || number(bytes, offset + 16, 4)? != 0
            || number(bytes, offset + 20, 4)? != 0
            || number(bytes, end - 16, 4)? != 0
            || number(bytes, end - 4, 4)? != 1
        {
            return Err(invalid());
        }
        size = number(bytes, offset + 40, 8)?;
        directory_end = offset;
        relative = number(bytes, offset + 48, 8)?;
        for (legacy, sentinel, actual) in [
            (number(bytes, end + 10, 2)?, u16::MAX as u64, count),
            (number(bytes, end + 12, 4)?, u32::MAX as u64, size),
            (number(bytes, end + 16, 4)?, u32::MAX as u64, relative),
        ] {
            if legacy != sentinel && legacy != actual {
                return Err(invalid());
            }
        }
    } else if count == 65535 || size == u32::MAX as u64 {
        return Err(invalid());
    }
    let size = usize::try_from(size).map_err(|_| invalid())?;
    let start = directory_end.checked_sub(size).ok_or_else(invalid)?;
    if relative > start as u64 {
        return Err(invalid());
    }
    // The eager reader retries earlier footer candidates. Embedded footers in
    // admitted metadata are rejected; payload footers are hidden by the reader
    // below while it constructs its metadata index. This deliberately excludes
    // otherwise-valid names, extras and comments containing those byte sequences:
    // listing availability is sacrificed rather than allowing an eager fallback.
    for (i, signature) in bytes[start..].windows(4).enumerate() {
        let position = start + i;
        if (signature == b"PK\x05\x06" && position != end)
            || (signature == b"PK\x06\x06" && (directory_end == end || position != directory_end))
        {
            return Err(invalid());
        }
    }
    // Each directory entry requires at least 46 bytes. Check before comparing
    // the count to a cap: impossible hostile counts are malformed, not truncated.
    if count > (size / 46) as u64 {
        return Err(invalid());
    }
    if count > 10_000 {
        return Ok(Some(ArchiveTruncation::Entries));
    }
    // Extra fields and comments are also eagerly allocated by the ZIP reader.
    if size > 4 * 1024 * 1024 {
        return Ok(Some(ArchiveTruncation::NameBytes));
    }
    let mut cursor = start;
    let mut names = 0usize;
    for _ in 0..count {
        if bytes.get(cursor..cursor.saturating_add(4)) != Some(b"PK\x01\x02") {
            return Err(invalid());
        }
        let name_len = number(bytes, cursor + 28, 2)? as usize;
        let extra_len = number(bytes, cursor + 30, 2)? as usize;
        let comment_len = number(bytes, cursor + 32, 2)? as usize;
        let name_start = cursor + 46;
        let next = name_start + name_len + extra_len + comment_len;
        if next > directory_end {
            return Err(invalid());
        }
        let name = &bytes[name_start..name_start + name_len];
        // CP437 characters can expand to three UTF-8 bytes. UTF-8 names use
        // the exact lossy-decoding length (without allocating a String).
        let mut displayed = if number(bytes, cursor + 8, 2)? & 0x800 != 0 {
            std::str::from_utf8(name).map_or(name.len() * 3, str::len)
        } else {
            name.iter()
                .map(|byte| if *byte < 128 { 1 } else { 3 })
                .sum()
        };
        let mut field = name_start + name_len;
        let extra_end = field + extra_len;
        while field < extra_end {
            if extra_end - field < 4 {
                return Err(invalid());
            }
            let tag = number(bytes, field, 2)?;
            let len = number(bytes, field + 2, 2)? as usize;
            field += 4;
            if len > extra_end - field {
                return Err(invalid());
            }
            // The Unicode path extra field can replace the raw name. Bound
            // it even when its CRC/version would later cause it to be ignored.
            if tag == 0x7075 && len >= 5 {
                displayed = displayed.max(len - 5);
            }
            field += len;
        }
        names += displayed;
        if displayed > 4096 || names > 1024 * 1024 {
            return Ok(Some(ArchiveTruncation::NameBytes));
        }
        cursor = next;
    }
    if cursor != directory_end {
        return Err(invalid());
    }
    Ok(None)
}

/// A reader that hides payload footers while the ZIP parser indexes metadata.
/// This prevents fallback to unchecked footers embedded in a member's bytes.
pub(super) struct AdmittedReader<'a> {
    cursor: std::io::Cursor<&'a [u8]>,
    start: u64,
    indexing: std::rc::Rc<std::cell::Cell<bool>>,
}
impl std::io::Read for AdmittedReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let position = self.cursor.position();
        let read = std::io::Read::read(&mut self.cursor, buffer)?;
        if self.indexing.get() && position < self.start {
            let hidden = ((self.start - position).min(read as u64)) as usize;
            // Local header reads are needed during indexing. Hide only footer
            // magic, including signatures crossing a read-buffer boundary.
            for (index, byte) in buffer[..hidden].iter_mut().enumerate() {
                let offset = position as usize + index;
                if self
                    .cursor
                    .get_ref()
                    .get(offset..offset + 4)
                    .is_some_and(|magic| magic == b"PK\x05\x06" || magic == b"PK\x06\x06")
                {
                    *byte = 0;
                }
            }
        }
        Ok(read)
    }
}
impl std::io::Seek for AdmittedReader<'_> {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        std::io::Seek::seek(&mut self.cursor, position)
    }
}

/// Construct only after successful preflight. Pin the metadata offset and hide
/// all earlier member footer signatures until indexing finishes; payload reads then work
/// normally, including CRC and decompression validation.
pub(super) fn open_admitted_zip(
    bytes: &[u8],
) -> Result<zip::ZipArchive<AdmittedReader<'_>>, ArchiveError> {
    let end = footer_offset(bytes)?;
    let (directory_end, size, relative) =
        if end >= 20 && bytes.get(end - 20..end - 16) == Some(b"PK\x06\x07") {
            let offset = number(bytes, end - 12, 8)? as usize;
            (
                offset,
                number(bytes, offset + 40, 8)?,
                number(bytes, offset + 48, 8)?,
            )
        } else {
            (
                end,
                number(bytes, end + 12, 4)?,
                number(bytes, end + 16, 4)?,
            )
        };
    let start = (directory_end as u64)
        .checked_sub(size)
        .ok_or_else(invalid)?;
    let archive_offset = start.checked_sub(relative).ok_or_else(invalid)?;
    let indexing = std::rc::Rc::new(std::cell::Cell::new(true));
    let reader = AdmittedReader {
        cursor: std::io::Cursor::new(bytes),
        start,
        indexing: indexing.clone(),
    };
    let config = zip::read::Config {
        archive_offset: zip::read::ArchiveOffset::Known(archive_offset),
    };
    let archive = zip::ZipArchive::with_config(config, reader)
        .map_err(|error| ArchiveError::Invalid(error.to_string()))?;
    indexing.set(false);
    Ok(archive)
}
