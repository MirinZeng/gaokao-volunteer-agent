from gaokao_agent.retriever import _minmax


def test_minmax_normalizes_candidate_scores() -> None:
    assert _minmax({1: 10.0, 2: 20.0}) == {1: 0.0, 2: 1.0}
    assert _minmax({1: 5.0, 2: 5.0}) == {1: 1.0, 2: 1.0}
    assert _minmax({1: 0.0, 2: 0.0}) == {1: 0.0, 2: 0.0}
