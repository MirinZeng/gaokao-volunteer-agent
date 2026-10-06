from pathlib import Path
import sqlite3

from openpyxl import Workbook

from gaokao_agent.excel_loader import ExcelAdmissionsLoader
from gaokao_agent.indexer import (
    BM25_DB_FILE,
    CHECKPOINT_FILE,
    CHECKPOINT_STATE_FILE,
    DOCUMENTS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    build_index,
)


class FakeBatchEmbeddings:
    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


def test_build_index_uses_incremental_checkpoint_and_cache(tmp_path: Path) -> None:
    excel_path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["说明", ""])
    sheet.append(["院校名称", "专业名称"])
    sheet.append(["甲大学", "计算机"])
    sheet.append(["乙大学", "法学"])
    workbook.save(excel_path)

    index_dir = tmp_path / "index"
    embeddings = FakeBatchEmbeddings()
    loader = ExcelAdmissionsLoader(excel_path, "Sheet1", 2)
    manifest = build_index(loader, embeddings, index_dir, batch_size=1)

    assert manifest["document_count"] == 2
    assert manifest["vector_enabled"] is True
    assert all(
        (index_dir / name).exists()
        for name in (INDEX_FILE, DOCUMENTS_FILE, BM25_DB_FILE, MANIFEST_FILE)
    )
    assert not (index_dir / CHECKPOINT_FILE).exists()
    assert not (index_dir / CHECKPOINT_STATE_FILE).exists()
    connection = sqlite3.connect(index_dir / BM25_DB_FILE)
    try:
        assert connection.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] == 2
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(admissions)").fetchall()
        }
        assert {
            "subject", "school_name", "school_nature", "major_name", "effective_min_rank"
        } <= columns
    finally:
        connection.close()
    calls_after_first_build = embeddings.calls

    cached_manifest = build_index(loader, embeddings, index_dir, batch_size=1)
    assert cached_manifest == manifest
    assert embeddings.calls == calls_after_first_build
