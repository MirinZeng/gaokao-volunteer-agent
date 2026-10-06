from gaokao_agent.intent import detect_intent, extract_rank


def test_extracts_common_rank_expressions() -> None:
    assert extract_rank("位次30000") == 30000
    assert extract_rank("30000位次左右") == 30000
    assert extract_rank("3万位") == 30000
    assert extract_rank("排名约3.2万") == 32000


def test_explicit_rank_with_subject_uses_numeric_route() -> None:
    intent = detect_intent("江西物理类，位次30000左右，有哪些计算机专业？", [], [])
    assert intent.kind == "rank"
    assert intent.subject == "物理"
    assert intent.rank_lookup.rank == 30000
    assert "计算机" in "".join(intent.remaining)


def test_rank_without_subject_is_not_silently_assumed() -> None:
    intent = detect_intent("30000位次计算机专业院校推荐", [], [])
    assert intent.kind == "semantic"
    assert intent.subject is None
