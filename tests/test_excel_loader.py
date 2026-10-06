from pathlib import Path

from openpyxl import Workbook

from gaokao_agent.excel_loader import ExcelAdmissionsLoader


def test_loader_preserves_excel_row_and_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["说明", ""])
    sheet.append(["院校名称", "专业名称"])
    sheet.append(["测试大学", "计算机科学与技术"])
    workbook.save(path)

    documents = ExcelAdmissionsLoader(path, "Sheet1", 2).load()

    assert len(documents) == 1
    assert documents[0].metadata["excel_row"] == 3
    assert documents[0].metadata["record"]["院校名称"] == "测试大学"
    assert "[Excel第3行]" in documents[0].page_content

