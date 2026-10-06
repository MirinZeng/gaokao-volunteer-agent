from __future__ import annotations

import re

import jieba


_ALNUM = re.compile(r"[A-Za-z0-9_.+-]+")
_CJK_TOKEN = re.compile(r"^[\u3400-\u9fff]+$")
_ALLOWED_TOKEN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9_.+-]+$")


def core_tokens(text: str) -> list[str]:
    """只返回 jieba 分词结果，不做双字扩展。

    双字片段是**追加在列表末尾**的，会破坏 token 的原始邻接关系；
    而索引里 jieba 分词结果是原样连续存放的（双字片段同样追加在后面），
    所以构造"邻接词组查询"必须用这个函数，不能用 ``tokenize_zh``。
    """

    normalized = text.lower().strip()
    return [
        token.strip()
        for token in jieba.lcut(normalized)
        if token.strip() and _ALLOWED_TOKEN.fullmatch(token.strip())
    ]


def tokenize_zh(text: str) -> list[str]:
    """为中文志愿数据生成词、数字及中文双字片段，兼顾精确名词与召回率。"""

    normalized = text.lower().strip()
    tokens = [
        token.strip()
        for token in jieba.lcut(normalized)
        if token.strip() and _ALLOWED_TOKEN.fullmatch(token.strip())
    ]
    tokens.extend(_ALNUM.findall(normalized))
    # 只为分词后的中短中文词补双字片段，避免给 3.9 万条长记录生成海量重复 token。
    for token in tuple(tokens):
        if _CJK_TOKEN.fullmatch(token) and 2 < len(token) <= 12:
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens
