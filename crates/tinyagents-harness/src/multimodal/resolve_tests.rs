use super::*;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::io::Write;

#[tokio::test]
async fn generic_resolution_retains_binary_bytes_and_decodes_only_transport_gzip() {
    let bytes = [0, 255, 7, 0];
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(&bytes).unwrap();
    let uri = format!(
        "data:application/gzip;original_mime=application/octet-stream;name=raw%20bytes.bin;base64,{}",
        STANDARD.encode(gzip.finish().unwrap())
    );
    let resolved = resolve_attachment(
        &uri,
        &FileLimits::default(),
        1024,
        &Client::new(),
        UnknownMimePolicy::Reject,
    )
    .await
    .unwrap();
    assert_eq!(resolved.bytes, bytes);
    assert_eq!(resolved.name, "raw bytes.bin");
    assert_eq!(resolved.mime, "application/octet-stream");
    assert_eq!(resolved.size_bytes, 4);
}

#[tokio::test]
async fn unknown_types_need_explicit_host_opt_in() {
    let uri = "data:application/x-private;name=a.custom;base64,AP8=";
    let limits = FileLimits::default();
    assert!(
        resolve_attachment(
            uri,
            &limits,
            1024,
            &Client::new(),
            UnknownMimePolicy::Reject
        )
        .await
        .is_err()
    );
    let resolved = resolve_attachment(
        uri,
        &limits,
        1024,
        &Client::new(),
        UnknownMimePolicy::Accept,
    )
    .await
    .unwrap();
    assert_eq!(resolved.mime, "application/x-private");
    assert_eq!(resolved.bytes, [0, 255]);
    assert!(
        resolve_file(uri, &limits, 1024, 1000, &Client::new(), &NoTextExtractor)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn disabled_files_and_transport_expansion_are_rejected() {
    let limits = FileLimits {
        max_files: 0,
        ..FileLimits::default()
    };
    assert!(
        resolve_attachment(
            "data:text/plain,hello",
            &limits,
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await
        .is_err()
    );
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(&vec![b'a'; 4096]).unwrap();
    let uri = format!(
        "data:application/gzip;original_mime=text/plain;base64,{}",
        STANDARD.encode(gzip.finish().unwrap())
    );
    assert!(
        resolve_attachment(
            &uri,
            &FileLimits::default(),
            100,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn archive_gzip_is_retained_and_nested_gzip_is_not_decoded_twice() {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"tar-like payload").unwrap();
    let original = encoder.finish().unwrap();
    let mut outer = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    outer.write_all(&original).unwrap();
    for uri in [
        format!(
            "data:application/gzip;name=a.tar.gz;base64,{}",
            STANDARD.encode(&original)
        ),
        format!(
            "data:application/gzip;original_mime=application/gzip;name=a.tar.gz;base64,{}",
            STANDARD.encode(outer.finish().unwrap())
        ),
    ] {
        let resolved = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(resolved.bytes, original);
        assert_eq!(resolved.name, "a.tar.gz");
        assert_eq!(resolved.mime, "application/gzip");
    }
}

#[tokio::test]
async fn local_unknown_bytes_preserve_legacy_rejection_and_generic_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("opaque.custom");
    tokio::fs::write(&path, [0, 255, 7]).await.unwrap();
    let source = path.to_str().unwrap();
    assert!(
        resolve_attachment(
            source,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Reject
        )
        .await
        .is_err()
    );
    let resolved = resolve_attachment(
        source,
        &FileLimits::default(),
        1024,
        &Client::new(),
        UnknownMimePolicy::Accept,
    )
    .await
    .unwrap();
    assert_eq!(resolved.bytes, [0, 255, 7]);
    assert_eq!(resolved.mime, "application/octet-stream");
    assert_eq!(resolved.name, "opaque.custom");
    assert!(matches!(
        resolve_attachment(
            source,
            &FileLimits::default(),
            2,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await,
        Err(MultimodalError::FileTooLarge { .. })
    ));
}

#[tokio::test]
async fn opt_in_retains_images_audio_video_and_native_documents_verbatim() {
    for (mime, bytes) in [
        ("image/png", &b"\x89PNG\r\n\x1a\n"[..]),
        ("audio/wav", &b"RIFF\0\0\0\0WAVE"[..]),
        ("video/mp4", &b"\0\0\0\x18ftypmp42"[..]),
        ("application/pdf", &b"%PDF-1.7"[..]),
    ] {
        let uri = format!("data:{mime};name=media;base64,{}", STANDARD.encode(bytes));
        let resolved = resolve_attachment(
            &uri,
            &FileLimits::default(),
            1024,
            &Client::new(),
            UnknownMimePolicy::Accept,
        )
        .await
        .unwrap();
        assert_eq!(resolved.mime, mime);
        assert_eq!(resolved.bytes, bytes);
    }
}

#[tokio::test]
async fn malformed_and_oversized_data_uri_payloads_fail_before_extraction() {
    for uri in [
        "data:text/plain;base64,%%%",
        "data:text/plain,bad%Q0",
        "data:application/gzip;original_mime=text/plain;base64,AP8=",
    ] {
        assert!(matches!(
            resolve_attachment(
                uri,
                &FileLimits::default(),
                1024,
                &Client::new(),
                UnknownMimePolicy::Accept
            )
            .await,
            Err(MultimodalError::InvalidFileMarker { .. })
        ));
    }
    assert!(matches!(
        resolve_attachment(
            "data:text/plain;base64,YWFhYWFhYWFhYWFh",
            &FileLimits::default(),
            2,
            &Client::new(),
            UnknownMimePolicy::Accept
        )
        .await,
        Err(MultimodalError::FileTooLarge { .. })
    ));
}
