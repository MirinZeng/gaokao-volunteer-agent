from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from rich.progress import Progress

from .embeddings import ZhipuEmbeddings
from .excel_loader import (
    ExcelAdmissionsLoader,
    document_from_json,
    document_search_text,
    document_to_json,
    infer_school_nature,
)
from .tokenizer import tokenize_zh


INDEX_FILE = "vectors.faiss"
DOCUMENTS_FILE = "documents.jsonl"
MANIFEST_FILE = "manifest.json"
BM25_DB_FILE = "bm25.sqlite3"
STRUCTURED_TABLE = "admissions"
CHECKPOINT_FILE = "embedding_checkpoint.npy"
CHECKPOINT_STATE_FILE = "embedding_checkpoint.json"


def _to_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("，", "")
    try:
        return int(float(text)) if text else None
    except ValueError:
        return None


def build_structured_database(documents: list[Document], database_path: Path) -> None:
    """把 Excel 关键字段落到有类型、有索引的 SQLite 表中。"""

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TABLE IF EXISTS admissions")
        connection.execute(
            """CREATE TABLE admissions (
                doc_index INTEGER PRIMARY KEY,
                excel_row INTEGER NOT NULL,
                year INTEGER,
                source_province TEXT,
                subject TEXT,
                batch TEXT,
                plan_nature TEXT,
                school_nature TEXT,
                school_code TEXT,
                school_name TEXT,
                group_code TEXT,
                group_name TEXT,
                major_code TEXT,
                major_name TEXT,
                major_full_name TEXT,
                selection_requirements TEXT,
                major_category TEXT,
                education_level TEXT,
                tuition INTEGER,
                plan_count INTEGER,
                city TEXT,
                province TEXT,
                min_score INTEGER,
                min_rank INTEGER,
                group_min_score INTEGER,
                group_min_rank INTEGER,
                effective_min_score INTEGER,
                effective_min_rank INTEGER,
                raw_record_json TEXT NOT NULL
            )"""
        )
        rows = []
        for doc_index, document in enumerate(documents):
            r = document.metadata.get("record", {})
            min_score = _to_int(r.get("最低分1"))
            min_rank = _to_int(r.get("最低位次1"))
            group_score = _to_int(r.get("专业组最低分1"))
            group_rank = _to_int(r.get("专业组最低位次1"))
            rows.append((
                doc_index, int(document.metadata["excel_row"]), _to_int(r.get("年份")),
                r.get("生源省"), r.get("科类"), r.get("批次"), r.get("计划性质"),
                infer_school_nature(r) or None,
                r.get("院校代码"), r.get("院校名称"), r.get("专业组代码"),
                r.get("专业组名称"), r.get("专业代码"), r.get("专业名称"),
                r.get("专业全称"), r.get("选科要求"),
                r.get("专业类") or r.get("专业类别"), r.get("专业层次"),
                _to_int(r.get("学费")), _to_int(r.get("计划人数")),
                r.get("城市"), r.get("所在省"), min_score, min_rank,
                group_score, group_rank, min_score or group_score, min_rank or group_rank,
                json.dumps(r, ensure_ascii=False, separators=(",", ":")),
            ))
        connection.executemany(
            "INSERT INTO admissions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        for statement in (
            "CREATE INDEX idx_admissions_subject_rank ON admissions(subject, effective_min_rank)",
            "CREATE INDEX idx_admissions_subject_score ON admissions(subject, effective_min_score)",
            "CREATE INDEX idx_admissions_school ON admissions(school_name)",
            "CREATE INDEX idx_admissions_school_nature ON admissions(school_nature)",
            "CREATE INDEX idx_admissions_city ON admissions(city)",
            "CREATE INDEX idx_admissions_major ON admissions(major_name)",
            "CREATE INDEX idx_admissions_group ON admissions(group_code)",
        ):
            connection.execute(statement)
        connection.commit()
        connection.execute("ANALYZE")
    finally:
        connection.close()


def structured_database_ready(database_path: Path) -> bool:
    if not database_path.exists():
        return False
    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (STRUCTURED_TABLE,)
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def rebuild_structured_database(index_dir: Path) -> int:
    documents_path = index_dir / DOCUMENTS_FILE
    if not documents_path.exists():
        raise FileNotFoundError(f"缺少文档索引：{documents_path}")
    with documents_path.open("r", encoding="utf-8") as stream:
        documents = [document_from_json(line) for line in stream if line.strip()]
    build_structured_database(documents, index_dir / BM25_DB_FILE)
    return len(documents)


def rebuild_search_database(index_dir: Path) -> int:
    """不重算向量，基于现有文档重建紧凑 BM25 与结构化表。"""

    documents_path = index_dir / DOCUMENTS_FILE
    if not documents_path.exists():
        raise FileNotFoundError(f"缺少文档索引：{documents_path}")
    with documents_path.open("r", encoding="utf-8") as stream:
        documents = [document_from_json(line) for line in stream if line.strip()]
    build_bm25_database(documents, index_dir / BM25_DB_FILE)
    return len(documents)


def build_bm25_database(documents: list[Document], database_path: Path) -> None:
    """将分词结果写入 SQLite FTS5；FTS5 的 bm25() 在查询时直接完成排序。"""

    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE VIRTUAL TABLE bm25_index USING fts5(doc_index UNINDEXED, tokens, tokenize='unicode61')"
        )
        batch: list[tuple[int, str]] = []
        for index, document in enumerate(documents):
            batch.append((index, " ".join(tokenize_zh(document_search_text(document)))))
            if len(batch) >= 500:
                connection.executemany(
                    "INSERT INTO bm25_index(doc_index, tokens) VALUES (?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO bm25_index(doc_index, tokens) VALUES (?, ?)",
                batch,
            )
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    build_structured_database(documents, database_path)


def build_bm25_only_index(
    loader: ExcelAdmissionsLoader,
    index_dir: Path,
) -> dict:
    """不调用任何 Embedding API，只建立文档库和 SQLite FTS5 BM25 索引。"""

    index_dir.mkdir(parents=True, exist_ok=True)
    summary = loader.summary()
    documents = loader.load()
    if not documents:
        raise ValueError("Excel 中没有可索引的数据行")

    with (index_dir / DOCUMENTS_FILE).open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(document_to_json(document) + "\n")
    build_bm25_database(documents, index_dir / BM25_DB_FILE)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": summary.path.name,
        "source_sha256": summary.sha256,
        "sheet": summary.sheet,
        "header_row": loader.header_row,
        "document_count": len(documents),
        "embedding_model": None,
        "embedding_dimension": None,
        "vector_enabled": False,
        "vector_metric": None,
        "bm25_backend": "sqlite_fts5",
        "structured_backend": "sqlite",
        "structured_schema_version": 1,
    }
    (index_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_index(
    loader: ExcelAdmissionsLoader,
    embeddings: ZhipuEmbeddings,
    index_dir: Path,
    batch_size: int,
    force: bool = False,
) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    summary = loader.summary()
    manifest_path = index_dir / MANIFEST_FILE

    if manifest_path.exists() and not force:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old_manifest.get("source_sha256") == summary.sha256
            and old_manifest.get("embedding_model") == embeddings.model
        ):
            required = [INDEX_FILE, DOCUMENTS_FILE, BM25_DB_FILE, MANIFEST_FILE]
            if all((index_dir / name).exists() for name in required):
                if not structured_database_ready(index_dir / BM25_DB_FILE):
                    rebuild_structured_database(index_dir)
                    old_manifest["structured_backend"] = "sqlite"
                    old_manifest["structured_schema_version"] = 1
                    manifest_path.write_text(
                        json.dumps(old_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return old_manifest

    documents = loader.load()
    if not documents:
        raise ValueError("Excel 中没有可索引的数据行")

    checkpoint_path = index_dir / CHECKPOINT_FILE
    checkpoint_state_path = index_dir / CHECKPOINT_STATE_FILE
    completed = 0
    matrix: np.ndarray | None = None
    if force:
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_state_path.unlink(missing_ok=True)
    elif checkpoint_path.exists() and checkpoint_state_path.exists():
        state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
        if (
            state.get("source_sha256") == summary.sha256
            and state.get("embedding_model") == embeddings.model
            and state.get("document_count") == len(documents)
        ):
            matrix = np.lib.format.open_memmap(checkpoint_path, mode="r+")
            completed = int(state.get("completed", 0))
            if matrix.shape[0] != len(documents) or completed > len(documents):
                raise RuntimeError("Embedding 检查点尺寸异常，请执行 ingest --force")

    with Progress() as progress:
        task = progress.add_task(
            f"生成 Embedding（{embeddings.model}）", total=len(documents), completed=completed
        )
        for start in range(completed, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            vectors = np.asarray(
                embeddings.embed_documents([document_search_text(document) for document in batch]),
                dtype="float32",
            )
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise RuntimeError("智谱 Embedding API 返回的向量数量或维度异常")
            if matrix is None:
                matrix = np.lib.format.open_memmap(
                    checkpoint_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(documents), vectors.shape[1]),
                )
            if vectors.shape[1] != matrix.shape[1]:
                raise RuntimeError("Embedding 维度发生变化，请执行 ingest --force")
            matrix[start : start + len(batch)] = vectors
            matrix.flush()
            completed = start + len(batch)
            checkpoint_state_path.write_text(
                json.dumps(
                    {
                        "source_sha256": summary.sha256,
                        "embedding_model": embeddings.model,
                        "document_count": len(documents),
                        "completed": completed,
                        "embedding_dimension": int(matrix.shape[1]),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            progress.update(task, completed=completed)

    assert matrix is not None
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_dir / INDEX_FILE))

    with (index_dir / DOCUMENTS_FILE).open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(document_to_json(document) + "\n")
    build_bm25_database(documents, index_dir / BM25_DB_FILE)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": summary.path.name,
        "source_sha256": summary.sha256,
        "sheet": summary.sheet,
        "header_row": loader.header_row,
        "document_count": len(documents),
        "embedding_model": embeddings.model,
        "embedding_dimension": int(matrix.shape[1]),
        "vector_enabled": True,
        "vector_metric": "cosine_similarity_via_normalized_inner_product",
        "bm25_backend": "sqlite_fts5",
        "structured_backend": "sqlite",
        "structured_schema_version": 1,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    del matrix
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_state_path.unlink(missing_ok=True)
    return manifest
