import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from gaokao_agent.excel_loader import document_to_json
from gaokao_agent.indexer import (
    BM25_DB_FILE,
    DOCUMENTS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    build_bm25_database,
)
from gaokao_agent.retriever import HybridExcelRetriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "计算机" in text else [0.0, 1.0]


def test_hybrid_retriever_returns_evidence_with_scores(tmp_path: Path) -> None:
    documents = [
        Document(
            id="Sheet1!3",
            page_content="[Excel第3行]\n院校名称: 测试大学\n专业名称: 计算机科学与技术",
            metadata={"excel_row": 3, "record": {"院校名称": "测试大学", "专业名称": "计算机科学与技术", "科类": "物理"}},
        ),
        Document(
            id="Sheet1!4",
            page_content="[Excel第4行]\n院校名称: 示例学院\n专业名称: 汉语言文学",
            metadata={"excel_row": 4, "record": {"院校名称": "示例学院", "专业名称": "汉语言文学", "科类": "历史"}},
        ),
    ]
    with (tmp_path / DOCUMENTS_FILE).open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(document_to_json(document) + "\n")
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(matrix)
    faiss.write_index(index, str(tmp_path / INDEX_FILE))
    build_bm25_database(documents, tmp_path / BM25_DB_FILE)
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps({"document_count": 2, "embedding_model": "fake", "vector_enabled": True}),
        encoding="utf-8",
    )

    retriever = HybridExcelRetriever(
        index_dir=tmp_path,
        embeddings=FakeEmbeddings(),
        bm25_weight=0.45,
        vector_weight=0.55,
        top_k=1,
        candidate_k=2,
    )
    result = retriever.invoke("物理类计算机专业")

    assert result[0].metadata["excel_row"] == 3
    assert result[0].metadata["hybrid_score"] > 0
