use super::*;

#[test]
fn malformed_archives_are_errors() {
    for format in [
        ArchiveFormat::Zip,
        ArchiveFormat::Tar,
        ArchiveFormat::TarGzip,
    ] {
        assert!(inspect_archive(b"not an archive", format, &ArchiveLimits::default()).is_err());
    }
}

use std::io::{Cursor, Write};

fn zip_fixture(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in entries {
        writer
            .start_file(
                *name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn tar_fixture() -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(5);
    // Set path bytes directly: the writer correctly refuses traversal paths,
    // but the inspector must display a hostile archive without extracting it.
    header.as_mut_bytes()[..14].copy_from_slice(b"../outside.txt");
    header.set_cksum();
    builder.append(&header, &b"hello"[..]).unwrap();
    let mut link = tar::Header::new_gnu();
    link.set_path("link").unwrap();
    link.set_entry_type(tar::EntryType::Symlink);
    link.set_link_name("/etc/passwd").unwrap();
    link.set_size(0);
    link.set_cksum();
    builder.append(&link, &b""[..]).unwrap();
    builder.into_inner().unwrap()
}

#[test]
fn zip_listing_preserves_names_sizes_and_applies_each_budget() {
    let bytes = zip_fixture(&[("../outside.txt", b"hello"), ("normal.txt", b"world")]);
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).unwrap();
    assert_eq!(
        list.entries[0],
        ArchiveEntry {
            name: "../outside.txt".into(),
            kind: ArchiveEntryKind::File,
            declared_size: 5
        }
    );
    assert_eq!(list.entries.len(), 2);
    assert_eq!(list.truncation, None);
    let limits = ArchiveLimits {
        max_entries: 1,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert_eq!(list.entries.len(), 1);
    assert_eq!(list.truncation, Some(ArchiveTruncation::Entries));
    for limits in [
        ArchiveLimits {
            max_name_bytes: 3,
            ..ArchiveLimits::default()
        },
        ArchiveLimits {
            max_total_name_bytes: 3,
            ..ArchiveLimits::default()
        },
    ] {
        let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
        assert!(list.entries.is_empty());
        assert_eq!(list.truncation, Some(ArchiveTruncation::NameBytes));
    }
    assert!(matches!(
        inspect_archive(
            &bytes,
            ArchiveFormat::Zip,
            &ArchiveLimits {
                max_input_bytes: 2,
                ..ArchiveLimits::default()
            }
        ),
        Err(ArchiveError::InputTooLarge)
    ));
}

#[test]
fn tar_and_tar_gzip_list_traversal_and_links_without_extracting() {
    let bytes = tar_fixture();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&bytes).unwrap();
    let gzip = encoder.finish().unwrap();
    for (format, bytes) in [
        (ArchiveFormat::Tar, bytes.as_slice()),
        (ArchiveFormat::TarGzip, gzip.as_slice()),
    ] {
        let list = inspect_archive(bytes, format, &ArchiveLimits::default()).unwrap();
        assert_eq!(list.entries[0].name, "../outside.txt");
        assert_eq!(list.entries[0].declared_size, 5);
        assert_eq!(list.entries[1].kind, ArchiveEntryKind::Symlink);
        assert_eq!(list.truncation, None);
        let list = inspect_archive(
            bytes,
            format,
            &ArchiveLimits {
                max_decompressed_bytes: 100,
                ..ArchiveLimits::default()
            },
        )
        .unwrap();
        assert_eq!(list.truncation, Some(ArchiveTruncation::DecompressedBytes));
    }
}

#[test]
fn compressed_zip_expansion_is_capped_and_truncated_tar_is_an_error() {
    let bytes = zip_fixture(&[("bomb", &vec![0; 100_000])]);
    assert!(bytes.len() < 1000);
    let list = inspect_archive(
        &bytes,
        ArchiveFormat::Zip,
        &ArchiveLimits {
            max_decompressed_bytes: 1000,
            ..ArchiveLimits::default()
        },
    )
    .unwrap();
    assert_eq!(list.entries[0].declared_size, 100_000);
    assert_eq!(list.truncation, Some(ArchiveTruncation::DecompressedBytes));
    let tar = tar_fixture();
    assert!(inspect_archive(&tar[..514], ArchiveFormat::Tar, &ArchiveLimits::default()).is_err());
}

#[test]
fn archive_detection_does_not_mistake_office_documents_for_archives() {
    assert_eq!(
        ArchiveFormat::detect("table.xlsx", "application/zip", b"PK\x03\x04"),
        None
    );
    assert_eq!(
        ArchiveFormat::detect("archive.zip", "application/zip", b"PK\x03\x04"),
        Some(ArchiveFormat::Zip)
    );
    assert_eq!(
        ArchiveFormat::detect("archive.tgz", "application/gzip", b"\x1f\x8b"),
        Some(ArchiveFormat::TarGzip)
    );
    assert_eq!(
        ArchiveFormat::detect("archive.tar", "application/x-tar", b""),
        Some(ArchiveFormat::Tar)
    );
}

#[test]
fn office_container_contents_override_generic_zip_headers() {
    for (member, expected) in [
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
        let bytes = zip_fixture(&[
            ("[Content_Types].xml", b"<Types/>"),
            (member, b"<document/>"),
        ]);
        assert_eq!(
            super::super::mime::detect_file_mime(None, &bytes, Some("application/zip")).as_deref(),
            Some(expected)
        );
    }
}

#[test]
fn corrupt_zip_payloads_and_gzip_trailers_are_errors() {
    let mut bytes = zip_fixture(&[("file.txt", b"hello world")]);
    let start = zip::ZipArchive::new(Cursor::new(&bytes))
        .unwrap()
        .by_index(0)
        .unwrap()
        .data_start() as usize;
    bytes[start] ^= 0xff;
    assert!(inspect_archive(&bytes, ArchiveFormat::Zip, &ArchiveLimits::default()).is_err());
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&tar_fixture()).unwrap();
    let mut gzip = encoder.finish().unwrap();
    let checksum = gzip.len() - 8;
    gzip[checksum] ^= 0xff;
    assert!(inspect_archive(&gzip, ArchiveFormat::TarGzip, &ArchiveLimits::default()).is_err());
}

#[test]
fn exact_and_zero_listing_budgets_have_no_off_by_one() {
    let bytes = zip_fixture(&[("a", b"hello")]);
    let limits = ArchiveLimits {
        max_entries: 1,
        max_name_bytes: 1,
        max_total_name_bytes: 1,
        max_decompressed_bytes: 5,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert_eq!(list.truncation, None);
    assert_eq!(list.entries.len(), 1);
    let limits = ArchiveLimits {
        max_entries: 0,
        ..ArchiveLimits::default()
    };
    let list = inspect_archive(&bytes, ArchiveFormat::Zip, &limits).unwrap();
    assert!(list.entries.is_empty());
    assert_eq!(list.truncation, Some(ArchiveTruncation::Entries));
}
