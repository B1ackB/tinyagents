use super::*;

#[test]
fn rerank_errors_convert_to_embedding_errors_with_context() {
    let source = tinyinference_embeddings::Error::Rerank(
        tinyinference_embeddings::rerank::RerankError::ResponseTooLarge { limit: 17 },
    );

    let converted = TinyAgentsError::from(source);

    assert!(matches!(
        converted,
        TinyAgentsError::Embedding(message) if message.contains("17 bytes")
    ));
}
