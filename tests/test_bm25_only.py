from pathlib import Path

from openpyxl import Workbook

from gaokao_agent.excel_loader import ExcelAdmissionsLoader
from gaokao_agent.indexer import build_bm25_only_index
from gaokao_agent.retriever import HybridExcelRetriever


def test_bm25_only_index_needs_no_embeddings(tmp_path: Path) -> None:
    excel_path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["说明", ""])
    sheet.append(["院校名称", "专业名称", "科类"])
    sheet.append(["测试大学", "计算机科学与技术", "物理"])
    sheet.append(["示例学院", "汉语言文学", "历史"])
    workbook.save(excel_path)

    index_dir = tmp_path / "index"
    manifest = build_bm25_only_index(
        ExcelAdmissionsLoader(excel_path, "Sheet1", 2),
        index_dir,
    )
    retriever = HybridExcelRetriever(index_dir=index_dir, embeddings=None, top_k=1, candidate_k=5)
    documents = retriever.invoke("物理类计算机专业")

    assert manifest["vector_enabled"] is False
    assert documents[0].metadata["excel_row"] == 3
    assert documents[0].metadata["cosine_score"] == -1.0


def test_rank_search_expands_other_majors_in_same_group(tmp_path: Path) -> None:
    excel_path = tmp_path / "groups.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["说明", ""])
    sheet.append(["院校名称", "专业组代码", "专业名称", "科类", "最低位次1", "专业组最低位次1"])
    sheet.append(["甲大学", "501", "计算机科学与技术", "物理", 30000, 30500])
    sheet.append(["甲大学", "501", "土木工程", "物理", 32000, 30500])
    sheet.append(["甲大学", "501", "风景园林", "物理", 33000, 30500])
    sheet.append(["乙大学", "502", "软件工程", "物理", 29000, 29500])
    workbook.save(excel_path)

    index_dir = tmp_path / "index"
    build_bm25_only_index(ExcelAdmissionsLoader(excel_path, "Sheet1", 2), index_dir)
    retriever = HybridExcelRetriever(
        index_dir=index_dir,
        embeddings=None,
        rank_listing_k=1,
    )

    documents = retriever.invoke("江西物理类，位次30000左右，有哪些计算机专业？")
    same_group = [
        document.metadata["record"].get("专业名称")
        for document in documents
        if document.metadata.get("group_role") == "same_group_major"
    ]

    assert "土木工程" in same_group
    assert "风景园林" in same_group
    assert all(document.metadata.get("excel_row") for document in documents)
