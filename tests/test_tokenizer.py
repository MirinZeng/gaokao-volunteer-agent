from gaokao_agent.tokenizer import tokenize_zh


def test_tokenizer_keeps_chinese_names_and_numbers() -> None:
    tokens = tokenize_zh("南昌大学 计算机科学与技术 最低位次30000")
    assert "南昌大学" in tokens
    assert "30000" in tokens
    assert "计算" in tokens

