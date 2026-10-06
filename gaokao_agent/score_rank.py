"""江西省一分一段表：分数 → 位次 的**确定性**换算。

**为什么不做 Embedding**：一分一段表是一张确定性对照表 —— 600 分对应哪个位次是唯一
确定的，这不是语义相似度问题。走向量检索只会返回"语义相近"的片段，模型完全可能读错
行；而位次一旦错一位，冲稳保的判断就整体偏了。所以正确做法是：把 PDF 抽成结构化 CSV，
用 O(1) 查表换算，再把**换算结果**作为事实注入提示词，而不是把原始表丢给模型自己看。

数据来源（用户提供的官方 PDF，各 21 页，3 列：分数 / 人数 / 累计人数）：
  - 江西省2025年普通高考物理类一分一段表.pdf → data/score_rank/物理.csv
  - 江西省2025年普通高考历史类一分一段表.pdf → data/score_rank/历史.csv

「累计人数」即**位次**（该分数及以上的考生人数）。

关于表中跳过的分数：官方表不列 0 人上线的分数段（实测物理类跳过 117、101 分），
这类分数的位次等于**紧邻其上**那一段的累计人数 —— 因为「≥117 分」与「≥118 分」
是同一批人。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# 用户的自然表述 → 数据里的科类字段值
SUBJECT_ALIASES: dict[str, str] = {
    "物理": "物理",
    "物理类": "物理",
    "物理组": "物理",
    "理科": "物理",
    "理": "物理",
    "历史": "历史",
    "历史类": "历史",
    "历史组": "历史",
    "文科": "历史",
    "文": "历史",
}


@dataclass(frozen=True)
class RankLookup:
    """一次分数 → 位次换算的结果。"""

    subject: str
    score: int
    rank: int
    people: int
    exact: bool
    note: str

    def describe(self) -> str:
        parts = [f"2025年江西{self.subject}类 {self.score}分 → 位次约 {self.rank:,}"]
        if self.people:
            parts.append(f"该分数段 {self.people} 人")
        if self.note:
            parts.append(self.note)
        return "，".join(parts)


class ScoreRankTable:
    """一个科类的一分一段表。"""

    def __init__(
        self,
        subject: str,
        scores: list[int],
        people: list[int],
        ranks: list[int],
    ):
        # scores 降序；ranks 为累计人数（即位次）
        self.subject = subject
        self.scores = scores
        self.people = people
        self.ranks = ranks
        self.max_score = scores[0]
        self.min_score = scores[-1]
        self.total = ranks[-1]
        self.top_is_open = True  # 第一段形如「676及以上」

        # 官方表会跳过 0 人上线的分数段，这里补齐：该分数的位次沿用紧邻其上的那一段。
        # 从高分往低分填，才能保证 score+1 已经算好。
        self._rank_by_score: dict[int, int] = {}
        self._people_by_score: dict[int, int] = {}
        filled: dict[int, int] = {}
        present = dict(zip(scores, ranks))
        for score in range(self.max_score, self.min_score - 1, -1):
            if score in present:
                filled[score] = present[score]
            else:
                filled[score] = filled[score + 1]
        self._rank_by_score = filled
        self._people_by_score = dict(zip(scores, people))
        self._present = set(scores)

    @classmethod
    def load(cls, path: Path, subject: str) -> "ScoreRankTable":
        scores: list[int] = []
        people: list[int] = []
        ranks: list[int] = []
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scores.append(int(row["分数"]))
                people.append(int(row["人数"]))
                ranks.append(int(row["累计人数"]))
        if not scores:
            raise ValueError(f"一分一段表为空：{path}")
        return cls(subject, scores, people, ranks)

    def rank_for(self, score: int) -> RankLookup | None:
        """分数 → 位次。分数低于表内最低分段时返回 None。"""

        if score > self.max_score:
            # 高于最高分段：只能确定"位次在最高段的累计人数以内"
            return RankLookup(
                subject=self.subject,
                score=score,
                rank=self.ranks[0],
                people=self.people[0],
                exact=False,
                note=f"高于表内最高分段，位次在第 {self.ranks[0]:,} 名以内",
            )

        if score < self.min_score:
            return None

        rank = self._rank_by_score[score]
        exact = score in self._present
        return RankLookup(
            subject=self.subject,
            score=score,
            rank=rank,
            people=self._people_by_score.get(score, 0),
            exact=exact,
            note="" if exact else "该分数无人上线，位次沿用上一分段",
        )

    def rank_range(self, low_score: int, high_score: int) -> tuple[int, int] | None:
        """分数区间 → 位次区间，用于「600分左右能上什么」。"""

        top = self.rank_for(high_score)
        bottom = self.rank_for(low_score)
        if top is None or bottom is None:
            return None
        return top.rank, bottom.rank


class ScoreRankRegistry:
    """按科类管理多张一分一段表。"""

    def __init__(self, tables: dict[str, ScoreRankTable]):
        self.tables = tables

    @classmethod
    def load(cls, directory: Path) -> "ScoreRankRegistry":
        tables: dict[str, ScoreRankTable] = {}
        directory = Path(directory)
        for subject in ("物理", "历史"):
            path = directory / f"{subject}.csv"
            if path.exists():
                tables[subject] = ScoreRankTable.load(path, subject)
        return cls(tables)

    def table(self, subject: str) -> ScoreRankTable | None:
        return self.tables.get(SUBJECT_ALIASES.get(subject.strip(), subject.strip()))

    def lookup(self, score: int, subject: str) -> RankLookup | None:
        table = self.table(subject)
        return table.rank_for(score) if table else None

    @property
    def subjects(self) -> list[str]:
        return sorted(self.tables)
