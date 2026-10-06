from langchain_core.documents import Document

from gaokao_agent.excel_loader import document_search_text


def test_search_text_keeps_core_fields_and_drops_verbose_details() -> None:
    document = Document(
        page_content="完整证据",
        metadata={
            "record": {
                "院校名称": "测试大学",
                "专业名称": "计算机科学与技术",
                "最低位次1": "30000",
                "选择原因": "普通公办/一线城市/专业有硕士点",
                "院校专业实力详情": "很长的非核心介绍" * 1000,
            }
        },
    )
    text = document_search_text(document)
    assert "测试大学" in text
    assert "计算机科学与技术" in text
    assert "30000" in text
    assert "办学性质: 公办" in text
    assert "很长的非核心介绍" not in text
