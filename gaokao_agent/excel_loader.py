from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from openpyxl import load_workbook


@dataclass(frozen=True)
class WorkbookSummary:
    path: Path
    sheet: str
    rows: int
    columns: int
    headers: tuple[str, ...]
    sha256: str


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# 只把真正有检索意义的字段送入 BM25/Embedding。完整 Excel 行仍保留在
# page_content/metadata.record 中供模型引用，避免长篇院校详情稀释核心语义。
SEARCH_FIELDS = (
    "年份", "生源省", "科类", "批次", "计划性质",
    "院校名称", "专业组名称", "专业组代码", "专业名称", "专业全称",
    "选科要求", "专业类", "专业类别", "专业层次", "城市", "所在省",
    "院校水平标签", "专业标签", "专业组定位",
    "最低分1", "最低位次1", "专业组最低分1", "专业组最低位次1",
)


def infer_school_nature(record: dict[str, Any]) -> str:
    """从“选择原因”的复合标签中规范化公办/民办；无法确认时返回空。"""

    labels = str(record.get("选择原因") or "")
    if "公办" in labels:
        return "公办"
    if "民办" in labels:
        return "民办"
    return ""


def document_search_text(document: Document) -> str:
    """为召回构造紧凑文本；不做任意字符切块，也不丢失完整证据。"""

    record = document.metadata.get("record", {})
    parts = [f"{field}: {record[field]}" for field in SEARCH_FIELDS if record.get(field)]
    if nature := infer_school_nature(record):
        parts.append(f"办学性质: {nature}")
    return "\n".join(parts) or document.page_content


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExcelAdmissionsLoader:
    """以只读模式加载 Excel，每个数据行转换为一条 LangChain Document。"""

    def __init__(self, path: Path, sheet: str = "Sheet1", header_row: int = 2):
        self.path = Path(path)
        self.sheet = sheet
        self.header_row = header_row

    def _worksheet(self):
        if not self.path.exists():
            raise FileNotFoundError(f"找不到 Excel 文件：{self.path}")
        workbook = load_workbook(self.path, read_only=True, data_only=True)
        if self.sheet not in workbook.sheetnames:
            workbook.close()
            raise ValueError(f"工作表 {self.sheet!r} 不存在，可用工作表：{workbook.sheetnames}")
        return workbook, workbook[self.sheet]

    def summary(self) -> WorkbookSummary:
        workbook, worksheet = self._worksheet()
        try:
            header_values = next(
                worksheet.iter_rows(
                    min_row=self.header_row,
                    max_row=self.header_row,
                    values_only=True,
                )
            )
            headers = tuple(_clean(value) for value in header_values)
            return WorkbookSummary(
                path=self.path,
                sheet=self.sheet,
                rows=max(worksheet.max_row - self.header_row, 0),
                columns=worksheet.max_column,
                headers=headers,
                sha256=file_sha256(self.path),
            )
        finally:
            workbook.close()

    def lazy_load(self) -> Iterator[Document]:
        workbook, worksheet = self._worksheet()
        try:
            raw_headers = next(
                worksheet.iter_rows(
                    min_row=self.header_row,
                    max_row=self.header_row,
                    values_only=True,
                )
            )
            headers = []
            seen: dict[str, int] = {}
            for column_index, raw_header in enumerate(raw_headers, start=1):
                base = _clean(raw_header) or f"未命名列{column_index}"
                seen[base] = seen.get(base, 0) + 1
                headers.append(base if seen[base] == 1 else f"{base}_{seen[base]}")

            for excel_row, values in enumerate(
                worksheet.iter_rows(min_row=self.header_row + 1, values_only=True),
                start=self.header_row + 1,
            ):
                record = {
                    header: cleaned
                    for header, value in zip(headers, values, strict=False)
                    if (cleaned := _clean(value))
                }
                if not record:
                    continue

                evidence_id = f"{self.sheet}!{excel_row}"
                lines = [f"[Excel第{excel_row}行]"]
                lines.extend(f"{key}: {value}" for key, value in record.items())
                yield Document(
                    id=evidence_id,
                    page_content="\n".join(lines),
                    metadata={
                        "source": self.path.name,
                        "sheet": self.sheet,
                        "excel_row": excel_row,
                        "evidence_id": evidence_id,
                        "record": record,
                    },
                )
        finally:
            workbook.close()

    def load(self) -> list[Document]:
        return list(self.lazy_load())


def document_to_json(document: Document) -> str:
    return json.dumps(
        {
            "id": document.id,
            "page_content": document.page_content,
            "metadata": document.metadata,
        },
        ensure_ascii=False,
    )


def document_from_json(line: str) -> Document:
    payload = json.loads(line)
    return Document(
        id=payload.get("id"),
        page_content=payload["page_content"],
        metadata=payload["metadata"],
    )
