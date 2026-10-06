from gaokao_agent.embeddings import ZhipuEmbeddings


def test_embed_documents_splits_batches_by_total_characters() -> None:
    embeddings = object.__new__(ZhipuEmbeddings)
    embeddings.batch_size = 32
    embeddings.max_batch_chars = 10
    embeddings.max_text_chars = 100
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(text))] for text in texts]

    embeddings._embed_adaptive = fake_embed  # type: ignore[method-assign]
    vectors = embeddings.embed_documents(["123456", "abcdef", "xy"])

    assert calls == [["123456"], ["abcdef", "xy"]]
    # 输出统一归一化，1 维正向向量都会变为单位向量。
    assert vectors == [[1.0], [1.0], [1.0]]


def test_embed_documents_splits_long_text_and_returns_one_vector() -> None:
    embeddings = object.__new__(ZhipuEmbeddings)
    embeddings.batch_size = 32
    embeddings.max_batch_chars = 100
    embeddings.max_text_chars = 4
    captured: list[str] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        captured.extend(texts)
        return [[1.0] for _ in texts]

    embeddings._embed_adaptive = fake_embed  # type: ignore[method-assign]
    vectors = embeddings.embed_documents(["123456789"])

    assert captured == ["1234", "5678", "9"]
    assert vectors == [[1.0]]
