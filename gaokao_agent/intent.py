"""查询意图识别。

把查询分成两类，交给不同的检索策略：

- ``entity``：查询里明确点名了某所院校（如「江西应用科技学院 计算机」）。
  院校范围一旦确定，就在该校文档内做**纯 BM25** 排序，好处有三：

  1. 精确 —— 不会混进别的学校；
  2. 省一次 Embedding 推理（本地 GPU 上约 50~100 ms）；
  3. 避免向量的"语义相近但并非同一实体"噪声。实测混合检索会把
     「江西应用工程职业学院」（余弦 0.7167）顶到「江西应用科技学院」前面，
     而后者在库里明明有 67 条记录。

- ``semantic``：用户只描述需求（如「我想学编程」），没有点名实体，
  此时向量检索才是主力，走 BM25 + 向量的混合检索。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .tokenizer import core_tokens

# 两类词对排序没有区分度，参与 MATCH 只会拖慢查询并引入噪声：
#   1. 提问虚词（有/哪些/什么…）
#   2. 字段名（专业/院校/计划…）—— 它们出现在几乎每一条记录里
STOPWORDS: frozenset[str] = frozenset(
    {
        "有", "哪些", "哪个", "什么", "怎么", "如何", "是", "的", "了", "吗", "呢",
        "请", "我", "想", "要", "可以", "能不能", "介绍", "一下", "情况", "以及",
        "和", "与", "或", "这", "那", "多少", "几", "怎么样", "还有", "给", "帮",
        "报", "读", "考", "上", "去", "来", "学校", "这所", "该校", "它",
        "参考", "推荐", "建议", "看看", "了解", "了解下", "相关", "方向",
        # 校名里的通用后缀：几乎所有文档都含「大学/学院」，留着只会稀释 BM25
        "大学", "学院", "能上", "考上", "我考", "考了", "想去", "想上", "报考",
        "专业", "院校", "名称", "代码", "计划", "人数", "学制", "学费", "选科",
        "要求", "组内", "备注",
    }
)


@dataclass(frozen=True)
class QueryIntent:
    """一次查询的意图判定结果。"""

    kind: str  # "entity" | "rank" | "city" | "semantic"
    schools: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    remaining: tuple[str, ...] = ()
    # 分数 → 位次换算（kind == "rank" 时填充）
    score: int | None = None
    subject: str | None = None
    rank_lookup: Any = None

    @property
    def is_entity(self) -> bool:
        return self.kind == "entity"

    @property
    def is_city(self) -> bool:
        return self.kind == "city"

    @property
    def is_rank(self) -> bool:
        return self.kind == "rank"


@dataclass(frozen=True)
class ExplicitRank:
    """用户直接给出的位次；接口与一分一段表的 RankLookup 保持一致。"""

    subject: str
    rank: int
    score: int | None = None
    exact: bool = True
    note: str = "用户直接提供的位次"


# 「600分」「考了612分」「600 分」：限定 3 位数且必须带「分」字，
# 避免把专业代码、计划人数、位次里的数字误判成分数。
SCORE_PATTERN = re.compile(r"(?<!\d)(\d{3})\s*分")
SCORE_MIN, SCORE_MAX = 100, 750
RANK_PATTERNS = (
    re.compile(r"(?:位次|排名)\s*(?:是|为|约|大约|大概|在)?\s*(\d+(?:\.\d+)?)\s*(万)?"),
    re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(万)?\s*(?:名|位)?\s*(?:左右|上下|附近)?\s*(?:的)?\s*(?:位次|排名)\s*(?:左右|上下|附近)?"),
    re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(万)\s*(?:名|位)(?!次)"),
)

# 只认多字表述。单字「理」「文」绝对不能放进来 ——
# 「管理」「理论」「专业管理」里全是「理」，会大面积误判。
SUBJECT_WORDS: tuple[tuple[str, str], ...] = (
    ("物理类", "物理"),
    ("物理组", "物理"),
    ("物理", "物理"),
    ("理科", "物理"),
    ("历史类", "历史"),
    ("历史组", "历史"),
    ("历史", "历史"),
    ("文科", "历史"),
)


def detect_subject(query: str) -> str | None:
    for word, subject in SUBJECT_WORDS:
        if word in query:
            return subject
    return None


def extract_score(query: str) -> int | None:
    match = SCORE_PATTERN.search(query)
    if not match:
        return None
    score = int(match.group(1))
    return score if SCORE_MIN <= score <= SCORE_MAX else None


def extract_rank(query: str) -> int | None:
    for pattern in RANK_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        value = float(match.group(1))
        if match.group(2):
            value *= 10000
        rank = int(round(value))
        return rank if rank > 0 else None
    return None


def _drop_consumed(tokens, consumed: set[str]) -> tuple[str, ...]:
    return tuple(
        token
        for token in dict.fromkeys(tokens)
        if token not in consumed and token not in STOPWORDS
    )


def detect_intent(
    query: str,
    school_names: list[str] | tuple[str, ...],
    city_names: list[str] | tuple[str, ...] = (),
    score_rank: Any = None,
) -> QueryIntent:
    """识别查询意图，优先级：院校实体 > 分数位次 > 城市 > 语义检索。

    院校名与城市名都采用**最长匹配优先**：否则问「南昌大学共青学院」时会同时命中
    「南昌大学」，问「南昌大学」时又会同时命中城市「南昌」，把范围错误地放大。
    """

    tokens = core_tokens(query)
    score = extract_score(query)
    explicit_rank = extract_rank(query)
    subject = detect_subject(query)

    # 数值位次优先走确定性筛选，绝不能退化成语义相似度检索。
    # 位次必须绑定科类；科类缺失时不猜测，由上层结合历史或追问。
    if explicit_rank is not None and subject is not None:
        consumed = {str(explicit_rank), "位次", "排名", "名", "位", "左右", "附近"}
        for word, mapped in SUBJECT_WORDS:
            if mapped == subject and word in query:
                consumed.update(core_tokens(word))
        return QueryIntent(
            kind="rank",
            subject=subject,
            rank_lookup=ExplicitRank(subject=subject, rank=explicit_rank),
            remaining=_drop_consumed(tokens, consumed),
        )

    # 分数 + 科类同样优先：先查一分一段表，再按位次筛选。
    if score is not None and subject is not None and score_rank is not None:
        lookup = score_rank.lookup(score, subject)
        if lookup is not None:
            consumed: set[str] = {str(score), "分"}
            for word, mapped in SUBJECT_WORDS:
                if mapped == subject and word in query:
                    consumed.update(core_tokens(word))
            return QueryIntent(
                kind="rank",
                score=score,
                subject=subject,
                rank_lookup=lookup,
                remaining=_drop_consumed(tokens, consumed),
            )

    # 1) 点名了院校 → 实体检索（纯 BM25，范围锁死在该校）
    matched = [name for name in school_names if name and name in query]
    if matched:
        longest = [
            name
            for name in matched
            if not any(name != other and name in other for other in matched)
        ]
        consumed: set[str] = set()
        for name in longest:
            consumed.update(core_tokens(name))
        return QueryIntent(
            kind="entity",
            schools=tuple(sorted(longest)),
            remaining=_drop_consumed(tokens, consumed),
        )

    # 2) 报了分数 + 科类 → 先换算位次，再按位次选院校。
    #    科类是必须的：位次是"该科类内"的名次，物理类 600 分是 8985 名，
    #    历史类 600 分却是 2199 名，差了 4 倍，绝不能混用。
    # 3) 点名了城市 → 城市检索（用 FTS5 词组硬过滤「城市」字段，再按其余词排序）
    matched_cities = [name for name in city_names if name and name in query]
    if matched_cities:
        longest_cities = [
            name
            for name in matched_cities
            if not any(name != other and name in other for other in matched_cities)
        ]
        consumed = set()
        for name in longest_cities:
            consumed.update(core_tokens(name))
        return QueryIntent(
            kind="city",
            cities=tuple(sorted(longest_cities)),
            remaining=_drop_consumed(tokens, consumed),
        )

    return QueryIntent(kind="semantic", remaining=_drop_consumed(tokens, set()))
