from langchain_core.documents import Document

from gaokao_agent.agent import (
    _is_insufficient_balance,
    build_retrieval_fallback,
    format_documents,
    parse_intent_plan,
)


def _document() -> Document:
    return Document(
        page_content="[Excel第88行] 测试大学 计算机科学与技术",
        metadata={
            "excel_row": 88, "source": "data.xlsx", "sheet": "Sheet1",
            "record": {"院校名称": "测试大学", "专业名称": "计算机科学与技术", "最低分1": 580, "最低位次1": 30000},
            "rank_band": "稳", "bm25_score": 1.0, "cosine_score": -1.0, "hybrid_score": 1.0,
        },
    )


def test_detects_insufficient_balance() -> None:
    assert _is_insufficient_balance(RuntimeError("Error code: 402 - Insufficient Balance"))
    assert not _is_insufficient_balance(RuntimeError("timeout"))


def test_balance_fallback_returns_excel_records() -> None:
    result = build_retrieval_fallback([_document()], "bm25")
    assert result["status"] == "fallback_retrieval"
    assert result["citations"] == [88]
    assert result["blocks"][1]["rows"][0]["cells"][0] == "测试大学"
    assert "未经大模型分析" in result["warning"]


def test_rank_decision_label_is_injected_into_context() -> None:
    context = format_documents([_document()])
    assert "系统决策标签：稳" in context
    assert "位次比" not in context  # 测试文档没设置 ratio，不应伪造


def test_llm_intent_distinguishes_subject_group_from_major() -> None:
    plan = parse_intent_plan(
        '{"goal":"major_selection","target_major":"","target_school":"",'
        '"preferences":["就业"],"normalized_query":"江西物理类600分 专业选择",'
        '"interpretation":"物理组是科类，并非物理学专业"}'
    )
    assert plan is not None
    assert plan["goal"] == "major_selection"
    assert plan["target_major"] == ""
    assert "物理学专业" not in plan["normalized_query"]
