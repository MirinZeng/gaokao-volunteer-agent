from langchain_core.documents import Document

from gaokao_agent.agent import _parse_block, build_result_event


def test_result_event_has_stable_json_shape() -> None:
    documents = [
        Document(
            page_content="证据",
            metadata={
                "excel_row": 10,
                "source": "data.xlsx",
                "sheet": "Sheet1",
                "bm25_score": 0.8,
                "cosine_score": -1.0,
                "hybrid_score": 0.8,
            },
        )
    ]
    result = build_result_event("结论。[Excel第10行]", documents, "bm25")

    assert result["type"] == "result"
    assert result["status"] == "ok"
    assert result["validated"] is True
    assert result["citations"] == [10]
    assert result["grounding"] == "excel_supported"
    assert result["evidence"][0]["excel_row"] == 10


def test_invalid_citation_returns_validation_failed_json() -> None:
    documents = [Document(page_content="证据", metadata={"excel_row": 10})]
    result = build_result_event("虚构。[Excel第99行]", documents, "bm25")

    assert result["status"] == "validation_failed"
    assert result["validated"] is False
    assert result["answer"] == "当前模型输出未通过 Excel 证据引用校验，因此已阻止采用。"


def test_general_guidance_without_excel_citation_is_allowed() -> None:
    result = build_result_event("一般性建议（非Excel数据）：可以关注个人兴趣。", [], "bm25")

    assert result["status"] == "ok"
    assert result["validated"] is True
    assert result["grounding"] == "general_guidance"


def test_school_details_are_parsed_and_cited() -> None:
    block = _parse_block(
        '{"type":"section","title":"院校逐项分析","layout":"school_details",'
        '"items":[{"school":"测试大学","text":"位次接近，组内另有土木工程。","excel_row":10}]}'
    )
    documents = [Document(page_content="证据", metadata={"excel_row": 10})]

    assert block is not None
    result = build_result_event([block], "", documents, "bm25")
    assert result["validated"] is True
    assert result["citations"] == [10]
    assert result["blocks"][0]["items"][0]["school"] == "测试大学"
