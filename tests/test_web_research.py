from gaokao_agent.web_research import WebResearcher, should_web_research


def test_web_search_is_only_triggered_for_fresh_or_policy_questions() -> None:
    assert should_web_research("2026年江西高考最新招生政策是什么？")
    assert should_web_research("南昌大学今年招生章程有变化吗？")
    assert not should_web_research("江西物理类30000位有哪些计算机专业？")
    assert should_web_research("我想想要保证就业呢？")
    assert should_web_research("哪些工科方向就业更好？", {"goal": "major_selection"})


def test_web_results_are_normalized_and_official_sources_first() -> None:
    researcher = WebResearcher(max_results=3)
    results = researcher._normalize([
        {"title": "第三方", "url": "https://example.com/a", "content": "摘要"},
        {"title": "高校招生网", "url": "https://zsb.example.edu.cn/a", "content": "公告"},
        {"title": "重复", "url": "https://example.com/a", "content": "重复"},
    ])
    assert len(results) == 2
    assert results[0].official is True
    assert results[0].title == "高校招生网"
