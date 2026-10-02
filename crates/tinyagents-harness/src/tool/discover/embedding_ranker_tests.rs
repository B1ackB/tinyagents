use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tinytools::{RankCandidate, RankContext, ToolRanker};

use super::*;
use tinyinference_embeddings::{EmbeddingModel, Result as EmbedResult};

/// Embeds a text as a bag of three hand-picked words, so similarity is
/// deterministic and readable.
struct BagEmbedder {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl EmbeddingModel for BagEmbedder {
    fn name(&self) -> &str {
        "bag"
    }
    fn model_id(&self) -> &str {
        "bag-v1"
    }
    fn dimensions(&self) -> usize {
        3
    }
    async fn embed(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.to_ascii_lowercase();
                vec![
                    f32::from(u8::from(t.contains("message") || t.contains("ping"))),
                    f32::from(u8::from(t.contains("email") || t.contains("mail"))),
                    f32::from(u8::from(t.contains("file"))),
                ]
            })
            .collect())
    }
}

const SIGNATURE: &str = "provider=bag;model=bag-v1;dims=3";

fn candidates() -> Vec<RankCandidate> {
    vec![
        RankCandidate::new("SLACK_SEND_MESSAGE", "send a message to a channel")
            .with_family("slack"),
        RankCandidate::new("GMAIL_SEND_EMAIL", "send an email").with_family("gmail"),
        RankCandidate::new("file_read", "read a file"),
    ]
}

#[tokio::test]
async fn ranks_by_cosine_and_embeds_the_catalogue_once() {
    let embedder = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    let ranker = EmbeddingToolRanker::new(embedder.clone(), SIGNATURE);
    assert_eq!(ranker.kind(), "embedding");

    let hits = ranker
        .rank("ping alex", &RankContext::empty(), &candidates(), 2)
        .await
        .unwrap();
    assert_eq!(hits[0].key, "SLACK_SEND_MESSAGE");
    assert!(hits[0].confidence.is_none());
    assert_eq!(hits.len(), 2);
    // One batch for the catalogue plus one for the intent.
    assert_eq!(embedder.calls.load(Ordering::SeqCst), 2);

    let hits = ranker
        .rank("mail the report", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(hits[0].key, "GMAIL_SEND_EMAIL");
    // Only the intent was embedded this time.
    assert_eq!(embedder.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn disk_cache_round_trips_and_is_keyed_by_signature() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cache").join("tool_search_embeddings.json");
    let embedder = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    let ranker =
        EmbeddingToolRanker::new(embedder.clone(), SIGNATURE).with_disk_cache(path.clone());
    ranker
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert!(path.exists());

    let embedder2 = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    let warm = EmbeddingToolRanker::new(embedder2.clone(), SIGNATURE).with_disk_cache(path.clone());
    warm.rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(
        embedder2.calls.load(Ordering::SeqCst),
        1,
        "a warm cache embeds only the intent"
    );
}

#[tokio::test]
async fn empty_intent_is_rejected() {
    let ranker = EmbeddingToolRanker::new(
        Arc::new(BagEmbedder {
            calls: AtomicUsize::new(0),
        }),
        SIGNATURE,
    );
    assert!(
        ranker
            .rank("  ", &RankContext::empty(), &candidates(), 1)
            .await
            .is_err()
    );
}

/// A tool that appears later — a newly connected toolkit's actions, a
/// rewritten description — is embedded on its own; the rest is a cache hit.
#[tokio::test]
async fn a_new_or_changed_tool_is_embedded_incrementally() {
    let embedder = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    let ranker = EmbeddingToolRanker::new(embedder.clone(), SIGNATURE);
    ranker
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst),
        2,
        "catalogue + intent"
    );

    let mut grown = candidates();
    grown.push(RankCandidate::new("NOTION_CREATE_PAGE", "create a page").with_family("notion"));
    grown[2] = RankCandidate::new("file_read", "read a file from disk");
    ranker
        .rank("ping", &RankContext::empty(), &grown, 1)
        .await
        .unwrap();
    // One batch for the two unseen texts (the new tool and the changed one),
    // plus the intent — never the whole catalogue again.
    assert_eq!(embedder.calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        ranker.cache.read().unwrap().len(),
        5,
        "old and new descriptions both cached; a stale entry is harmless"
    );
}

/// A cache written under one embedding space is ignored under another.
#[tokio::test]
async fn a_disk_cache_written_under_another_signature_is_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("cache.json");
    let embedder = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    EmbeddingToolRanker::new(embedder, SIGNATURE)
        .with_disk_cache(path.clone())
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();

    let other = Arc::new(BagEmbedder {
        calls: AtomicUsize::new(0),
    });
    EmbeddingToolRanker::new(other.clone(), "provider=bag;model=bag-v2;dims=3")
        .with_disk_cache(path)
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(
        other.calls.load(Ordering::SeqCst),
        2,
        "a different signature re-embeds the catalogue and the intent"
    );
}
