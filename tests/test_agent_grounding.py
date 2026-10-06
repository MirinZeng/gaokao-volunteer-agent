from langchain_core.documents import Document

from gaokao_agent.agent import validate_citations


def test_citations_must_point_to_retrieved_excel_rows() -> None:
    documents = [Document(page_content="证据", metadata={"excel_row": 10})]

    assert validate_citations("结论。[Excel第10行]", documents)[0]
    assert validate_citations("一般性建议，没有引用", documents)[0]
    assert not validate_citations("虚构引用。[Excel第99行]", documents)[0]
