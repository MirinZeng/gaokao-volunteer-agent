from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr
from .excel_loader import document_from_json, document_search_text, infer_school_nature
from .indexer import BM25_DB_FILE, DOCUMENTS_FILE, INDEX_FILE, MANIFEST_FILE
from .intent import QueryIntent, detect_intent
from .tokenizer import core_tokens, tokenize_zh


_RANK_QUERY_NOISE = {
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东",
    "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海",
    "宁夏", "新疆", "物理", "历史", "物理类", "历史类", "物理组", "历史组", "理科",
    "文科", "类", "组", "左右", "附近", "专业", "院校", "推荐", "参考",
}


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-12:
        constant = 0.0 if abs(high) < 1e-12 else 1.0
        return {key: constant for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _to_int(value: Any) -> int | None:
    """把 Excel 里的位次/分数单元格转成 int（容忍千分位、全角、空白、小数点）。"""

    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("，", "").replace(" ", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


class HybridExcelRetriever(BaseRetriever):
    """LangChain Retriever：BM25 与归一化向量内积（余弦相似度）加权融合。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_dir: Path
    embeddings: Embeddings | None = None
    bm25_weight: float = 0.45
    vector_weight: float = 0.55
    top_k: int = 8
    candidate_k: int = 40
    # 邻接词组命中（整串校名等精确实体）的额外加权基数
    phrase_bonus: float = 0.15
    # 实体意图且无额外检索词时的证据上限（整校列举必须尽量给全）
    entity_listing_k: int = 80
    # 城市意图的证据上限（如「南昌有哪些软件工程专业」实测涉及 35 所院校 / 94 条记录）
    city_listing_k: int = 40
    # 位次意图的证据上限（按冲/稳/保分桶后轮转取样）
    rank_listing_k: int = 40
    # 冲稳保分档：k = 该校最低位次 / 你的位次
    #   k < 1  → 该校去年录取名次比你更靠前 → 你够不到 → 冲
    #   k ≈ 1  → 稳
    #   k > 1  → 去年录到比你靠后的名次 → 你有余量 → 保
    rank_reach_lo: float = 0.70
    rank_reach_hi: float = 0.90
    rank_match_hi: float = 1.10
    rank_guarantee_hi: float = 1.25
    rank_safety_hi: float = 1.40
    # 一分一段表（分数 → 位次 的确定性换算）
    score_rank: Any = None
    # 交叉编码器重排器（可选）。召回是双塔结构、精度有限，重排能让真正相关的记录上来。
    reranker: Any = None
    # 送去重排的候选数量：太小则重排无从发挥，太大则每个查询要多跑很多次前向
    rerank_candidates: int = 60

    _documents: list[Document] = PrivateAttr(default_factory=list)
    _index: Any = PrivateAttr(default=None)
    _manifest: dict[str, Any] = PrivateAttr(default_factory=dict)
    # 院校名 -> 文档下标；实体意图直接用它把检索范围锁死到某所学校
    _school_docs: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _school_names: list[str] = PrivateAttr(default_factory=list)
    # 城市名 -> 文档下标；城市意图用它做二次校验（硬过滤交给 FTS5 的字段词组）
    _city_docs: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _city_names: list[str] = PrivateAttr(default_factory=list)
    # 科类 -> [(最低位次, 文档下标)]，按位次升序
    _rank_index: dict[str, list[tuple[int, int]]] = PrivateAttr(default_factory=dict)
    _group_docs: dict[tuple[str, str, str], list[int]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        documents_path = self.index_dir / DOCUMENTS_FILE
        bm25_path = self.index_dir / BM25_DB_FILE
        manifest_path = self.index_dir / MANIFEST_FILE
        missing = [
            str(path)
            for path in (documents_path, bm25_path, manifest_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("索引尚未建立，缺少：" + "、".join(missing))

        with documents_path.open("r", encoding="utf-8") as stream:
            self._documents = [document_from_json(line) for line in stream if line.strip()]
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self._manifest.get("vector_enabled", True):
            index_path = self.index_dir / INDEX_FILE
            if not index_path.exists():
                raise FileNotFoundError(f"向量索引缺失：{index_path}")
            if self.embeddings is None:
                raise RuntimeError("向量索引已启用，但没有配置 Embedding 客户端")
            self._index = faiss.read_index(str(index_path))
            if self._index.ntotal != len(self._documents):
                raise RuntimeError("FAISS 向量数量与文档数量不一致，请重新执行 ingest --force")

        # 预建院校名 -> 文档下标，供实体意图直接锁定检索范围。
        for index, document in enumerate(self._documents):
            record = document.metadata.get("record", {})
            name = record.get("院校名称")
            if name:
                self._school_docs.setdefault(str(name), []).append(index)
            city = record.get("城市")
            if city and str(city).strip():
                self._city_docs.setdefault(str(city).strip(), []).append(index)
            subject = str(record.get("科类", "")).strip()
            school = str(record.get("院校名称", "")).strip()
            group = str(record.get("专业组代码") or record.get("专业组名称") or "").strip()
            if subject and school and group:
                self._group_docs.setdefault((subject, school, group), []).append(index)
        # 按长度倒序，便于最长匹配优先
        self._school_names = sorted(self._school_docs, key=len, reverse=True)
        self._city_names = sorted(self._city_docs, key=len, reverse=True)

        # 预建「科类 -> [(最低位次, 文档下标)]」并按位次升序排好，供位次检索二分定位。
        # 用最近一年（最低位次1），缺失时退回专业组层面的位次。
        for index, document in enumerate(self._documents):
            record = document.metadata.get("record", {})
            subject = str(record.get("科类", "")).strip()
            if not subject:
                continue
            rank = _to_int(record.get("最低位次1"))
            if rank is None:
                rank = _to_int(record.get("专业组最低位次1"))
            if rank is None or rank <= 0:
                continue
            self._rank_index.setdefault(subject, []).append((rank, index))
        for subject in self._rank_index:
            self._rank_index[subject].sort()

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def plan(self, query: str) -> QueryIntent:
        """把查询解析成意图。

        公开出来是为了让 Agent 能读到分数→位次的换算结果（用于注入提示词），
        同时保持**无状态** —— 不把结果存在实例上，避免并发请求互相覆盖。
        """

        return detect_intent(query, self._school_names, self._city_names, self.score_rank)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # 意图路由：院校实体 → 分数位次 → 城市 → 混合检索
        intent = self.plan(query)
        if intent.is_entity:
            return self._entity_search(query, intent)
        if intent.is_rank:
            return self._rank_search(query, intent)
        if intent.is_city:
            return self._city_search(query, intent)
        return self._hybrid_search(query)

    def _rank_search(self, query: str, intent: QueryIntent) -> list[Document]:
        """按位次选院校。

        先用一分一段表把分数**确定性地**换成位次，再挑位次相近的院校专业，
        按冲/稳/保分桶后轮转取样，保证三档都有代表。

        判据 k = 该校最近一年最低位次 / 你的位次：
          k < 0.90         → 冲（该校去年录到比你更靠前的名次，你够不到）
          0.90 ≤ k < 1.10  → 稳
          1.10 ≤ k ≤ 1.60  → 保（去年录到比你靠后的名次，你有余量）

        科类必须一致：位次是"该科类内"的名次，物理类 600 分是 8985 名、
        历史类 600 分却是 2199 名，差了 4 倍，混用会得出完全错误的结论。
        """

        lookup = intent.rank_lookup
        subject = intent.subject or ""
        if lookup is None:
            return []

        user_rank = max(int(lookup.rank), 1)
        # 数值筛选必须由结构化 SQLite 完成，不能依赖向量相似度或让模型猜范围。
        lower_rank = max(int(user_rank * self.rank_reach_lo), 1)
        upper_rank = max(int(user_rank * self.rank_safety_hi), lower_rank)
        requested_nature = "公办" if "公办" in query else "民办" if "民办" in query else None
        requested_cities = [city for city in self._city_names if city and city in query]
        # 只有明确说“学费/每年预算”才做硬过滤；笼统的家庭总预算交给模型解释，避免误杀。
        budget_match = re.search(
            r"(?:学费|每年预算|年预算)\s*(?:不超过|控制在|最多|约|为|是)?\s*(\d+(?:\.\d+)?)\s*(万|元)?",
            query,
        )
        max_tuition = None
        if budget_match:
            max_tuition = int(float(budget_match.group(1)) * (10000 if budget_match.group(2) == "万" else 1))
        connection = sqlite3.connect(self.index_dir / BM25_DB_FILE)
        try:
            sql = (
                "SELECT doc_index, effective_min_rank FROM admissions "
                "WHERE subject = ? AND effective_min_rank BETWEEN ? AND ?"
            )
            params: list[Any] = [subject, lower_rank, upper_rank]
            if requested_nature:
                sql += " AND school_nature = ?"
                params.append(requested_nature)
            if requested_cities:
                placeholders = ",".join("?" for _ in requested_cities)
                sql += f" AND city IN ({placeholders})"
                params.extend(requested_cities)
            if max_tuition is not None:
                sql += " AND tuition IS NOT NULL AND tuition <= ?"
                params.append(max_tuition)
            sql += " ORDER BY effective_min_rank"
            index_rows = [
                (int(rank), int(index))
                for index, rank in connection.execute(sql, params).fetchall()
            ]
        finally:
            connection.close()
        if not index_rows:
            return []

        banded: list[tuple[str, float, int, int]] = []
        for rank, index in index_rows:
            ratio = rank / user_rank
            if ratio < self.rank_reach_lo or ratio > self.rank_safety_hi:
                continue
            if ratio < self.rank_reach_hi:
                band = "冲"
            elif ratio < self.rank_match_hi:
                band = "稳"
            elif ratio < self.rank_guarantee_hi:
                band = "保"
            else:
                band = "垫"
            banded.append((band, ratio, index, rank))

        remaining = [token for token in intent.remaining if token]
        if not banded:
            # 数值边界是硬约束：宁可返回空证据，也不能为了“有答案”把相差数倍的
            # 院校塞进结果。否则 3 万位会出现 10 万位“保底”这类误导。
            return []
        else:
            # 组内排序：有剩余检索词（如「600分 想学计算机」）就按 BM25 专业相关度，
            # 否则按位次接近程度 —— 越贴近你的位次越是"稳"，优先后展示。
            # “江西”之类生源地词存在于几乎每行，不能让它把医学、金融等无关专业
            # 一起召回。将真正的专业偏好做双字扩展，并把 BM25 未命中的记录剔除；
            # 数值范围与专业相关性由此成为两个独立硬门槛。
            focus_text = " ".join(token for token in remaining if token not in _RANK_QUERY_NOISE)
            focus_tokens = [
                token for token in dict.fromkeys(tokenize_zh(focus_text))
                if len(token) >= 2 and token not in _RANK_QUERY_NOISE
            ]
            relevance = self._bm25_over(focus_tokens, [index for _, _, index, _ in banded])
            if focus_tokens:
                banded = [item for item in banded if item[2] in relevance]
            if relevance:
                banded.sort(key=lambda item: relevance[item[2]], reverse=True)
            else:
                banded.sort(key=lambda item: abs(item[1] - 1.0))

        if not banded:
            return []

        band_of = {index: band for band, _, index, _ in banded}
        ratio_of = {index: ratio for _, ratio, index, _ in banded}
        rank_of = {index: rank for _, _, index, rank in banded}

        def band_school_key(index: int) -> tuple:
            return (band_of.get(index, ""), str(self._record_of(index).get("院校名称", "")))

        picked = self._diversify(
            [index for _, _, index, _ in banded], self.rank_listing_k, band_school_key
        )

        results: list[Document] = []
        selected_meta: dict[int, tuple[str, float, int]] = {}
        for index in picked:
            document = self._build_document(index, "rank_bm25", 0.0, 0.0, -1.0, None)
            document.metadata["rank_band"] = band_of.get(index, "")
            document.metadata["rank_ratio"] = round(ratio_of.get(index, 0.0), 4)
            document.metadata["doc_rank"] = rank_of.get(index)
            document.metadata["group_role"] = "candidate"
            results.append(document)
            selected_meta[index] = (
                band_of.get(index, ""), ratio_of.get(index, 0.0), rank_of.get(index, 0)
            )

        # 对排名最靠前的候选专业组展开组内专业。使用紧凑文本，保留真实 Excel 行号，
        # 让模型能判断调剂风险，而不是只看到组里的“计算机”一行。
        seen = set(picked)
        group_count = 0
        for parent_index in picked:
            record = self._record_of(parent_index)
            group = str(record.get("专业组代码") or record.get("专业组名称") or "").strip()
            key = (subject, str(record.get("院校名称", "")).strip(), group)
            siblings = self._group_docs.get(key, [])
            if not group or len(siblings) <= 1:
                continue
            group_count += 1
            band, ratio, doc_rank = selected_meta[parent_index]
            for sibling_index in siblings:
                if sibling_index in seen:
                    continue
                seen.add(sibling_index)
                sibling = self._build_document(
                    sibling_index, "rank_group_context", 0.0, 0.0, -1.0, None
                )
                sibling.page_content = (
                    f"[Excel第{sibling.metadata['excel_row']}行]\n"
                    + document_search_text(sibling)
                )
                sibling.metadata.update({
                    "rank_band": band,
                    "rank_ratio": round(ratio, 4),
                    "doc_rank": doc_rank,
                    "group_role": "same_group_major",
                })
                results.append(sibling)
                if len(results) >= 100:
                    return results
            if group_count >= 12:
                break
        return results

    def _city_search(self, query: str, intent: QueryIntent) -> list[Document]:
        """城市检索：用 FTS5 词组硬过滤「城市」字段，再按其余检索词排序。

        关键点是复用文档自身的字段结构：记录里 ``城市: 南昌`` 会被 jieba 切成相邻的
        ``城市`` ``南昌`` 两个 token，于是词组 ``"城市 南昌"`` 就等价于「城市字段 = 南昌」，
        不必把几千个 doc_index 传进 SQL（实测 8 ms vs 38 ms）。

        实测「南昌有哪些软件工程专业」：全库 ``"软件工程"`` 命中 4613 条，其中南昌 94 条、
        涉及 35 所院校；而混合检索的 top_k=8 只给出了 5 所。
        """

        city = intent.cities[0]
        city_set = set(self._city_docs.get(city, ()))
        if not city_set:
            return []

        remaining = [token for token in intent.remaining if token]
        expression = f'"城市 {city}"'
        if remaining:
            # 关键词在查询与文档里的 jieba 切分可能不一致（如查询得到「计算机专业」，
            # 文档里却是「计算机」「专业」），所以这里用 tokenize_zh 把双字片段也展开，
            # 提高命中率。城市那一半是 AND 硬过滤，多出来的词只影响城市内的排序。
            alternatives_tokens = list(dict.fromkeys(tokenize_zh(" ".join(remaining))))
            alternatives = " OR ".join(
                '"' + token.replace('"', '""') + '"' for token in alternatives_tokens
            )
            expression = f"{expression} AND ({alternatives})"

        connection = sqlite3.connect(self.index_dir / BM25_DB_FILE)
        try:
            rows = connection.execute(
                "SELECT doc_index, bm25(bm25_index) AS score FROM bm25_index "
                "WHERE bm25_index MATCH ? ORDER BY score LIMIT ?",
                (expression, self.rerank_candidates * 2),
            ).fetchall()
        finally:
            connection.close()

        # 双保险：词组匹配已保证属于该城市，这里再核对一次 doc_index 归属
        ranked = [(-float(score), int(index)) for index, score in rows if int(index) in city_set]

        if not ranked:
            # 词组没命中（城市写法不一致等）时退回：直接在该城市的记录里按院校轮转
            return [
                self._build_document(index, "city_bm25", 0.0, 0.0, -1.0, None)
                for index in self._diversify(
                    sorted(city_set), self.city_listing_k, self._school_key
                )
            ]

        # 顺序很重要：先重排（把真正相关的顶上来），再按院校轮转取样。
        # 只轮转不重排的话相关性会丢；只重排不轮转的话 40 条可能全来自同一所学校
        # （实测华东交通大学一所就有 10 条软件工程）。
        reranked = self._apply_rerank(query, ranked, self.rerank_candidates)
        score_of = {index: score for score, index, _ in reranked}
        rerank_of = {index: rerank for _, index, rerank in reranked}
        ordered = self._diversify(
            [index for _, index, _ in reranked], self.city_listing_k, self._school_key
        )
        return [
            self._build_document(
                index,
                "city_bm25",
                score_of.get(index, 0.0),
                score_of.get(index, 0.0),
                -1.0,
                rerank_of.get(index),
            )
            for index in ordered
        ]

    def _entity_search(self, query: str, intent: QueryIntent) -> list[Document]:
        """实体检索：范围锁死在命中的院校内，只做 BM25，不调用 Embedding。

        两条分支：
          - 有剩余检索词（如「江西应用科技学院 计算机」）：校内 BM25 排序取 top_k；
          - 无剩余检索词（如「江西应用科技学院有哪些专业」）：按专业组轮转取样给全。
            后者很关键：该校 67 条分属 6 个专业组，若只取文档顺序前 8 条会全部落在
            第101组，模型于是回答"未提供其他专业组"——对证据正确，对数据是错的。
        """

        entity_docs: list[int] = []
        for name in intent.schools:
            entity_docs.extend(self._school_docs.get(name, ()))
        entity_docs = sorted(set(entity_docs))
        if not entity_docs:
            return []

        remaining = [token for token in intent.remaining if token]
        if not remaining:
            # 无剩余检索词 = 用户在问"这所学校有哪些专业"：按专业组轮转给全。
            # 这里刻意不重排 —— 重排会打乱专业组覆盖，而且 80 条要跑 80 次前向。
            return [
                self._build_document(index, "entity_bm25", 0.0, 0.0, -1.0, None)
                for index in self._diversify(entity_docs, self.entity_listing_k, self._group_key)
            ]

        # 有剩余检索词：校内 BM25 粗排后交给重排器精排。
        # MATCH 只用"除去校名之外的剩余词"——校名已经通过 doc_index 锁定，
        # 若把它也塞进 MATCH，命中数会膨胀到全库级别（实测 2770 ms → 50 ms）。
        match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in remaining)
        placeholders = ",".join("?" * len(entity_docs))
        connection = sqlite3.connect(self.index_dir / BM25_DB_FILE)
        try:
            rows = connection.execute(
                f"SELECT doc_index, bm25(bm25_index) AS score FROM bm25_index "
                f"WHERE bm25_index MATCH ? AND doc_index IN ({placeholders}) "
                f"ORDER BY score LIMIT ?",
                (match_query, *entity_docs, self.rerank_candidates),
            ).fetchall()
        finally:
            connection.close()

        ranked: list[tuple[float, int]] = [(-float(score), int(index)) for index, score in rows]
        seen = {index for _, index in ranked}
        # 补齐到 rerank_candidates：给重排器机会把"BM25 没匹配上但其实相关"的校内记录捞回来
        for index in entity_docs:
            if len(ranked) >= self.rerank_candidates:
                break
            if index not in seen:
                ranked.append((0.0, index))
                seen.add(index)
        ranked.sort(key=lambda item: item[0], reverse=True)

        return [
            self._build_document(index, "entity_bm25", score, score, -1.0, rerank_score)
            for score, index, rerank_score in self._apply_rerank(query, ranked, self.top_k)
        ]

    def _record_of(self, index: int) -> dict:
        return self._documents[index].metadata.get("record", {})

    def _group_key(self, index: int) -> tuple:
        record = self._record_of(index)
        return (
            str(record.get("科类", "")),
            str(record.get("批次", "")),
            str(record.get("专业组名称") or record.get("专业组代码") or ""),
        )

    def _school_key(self, index: int) -> tuple:
        return (str(self._record_of(index).get("院校名称", "")),)

    def _bm25_over(self, tokens: list[str], candidates: list[int]) -> dict[int, float]:
        """在给定文档子集内做 BM25 打分（用于实体/位次路径的组内排序）。"""

        if not tokens or not candidates:
            return {}
        match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        placeholders = ",".join("?" * len(candidates))
        connection = sqlite3.connect(self.index_dir / BM25_DB_FILE)
        try:
            rows = connection.execute(
                f"SELECT doc_index, bm25(bm25_index) AS score FROM bm25_index "
                f"WHERE bm25_index MATCH ? AND doc_index IN ({placeholders}) "
                f"ORDER BY score LIMIT ?",
                (match_query, *candidates, len(candidates)),
            ).fetchall()
        finally:
            connection.close()
        return {int(index): -float(score) for index, score in rows}

    def _apply_rerank(
        self, query: str, ranked: list[tuple[float, int]], limit: int
    ) -> list[tuple[float, int, float | None]]:
        """对前 ``rerank_candidates`` 个候选做交叉编码器重排。

        返回 ``(召回分数, 文档下标, 重排分数)``。没配重排器时重排分数为 None。
        """

        if self.reranker is None or not ranked:
            return [(score, index, None) for score, index in ranked[:limit]]

        head = ranked[: self.rerank_candidates]
        try:
            order = self.reranker.rerank(query, [self._documents[index] for _, index in head])
        except Exception:  # noqa: BLE001 - 重排失败绝不能拖垮整个检索
            return [(score, index, None) for score, index in ranked[:limit]]

        return [
            (head[position][0], head[position][1], rerank_score)
            for position, rerank_score in order[:limit]
        ]

    def _build_document(
        self,
        index: int,
        route: str,
        hybrid_score: float,
        bm25_score: float,
        cosine_score: float,
        rerank_score: float | None,
    ) -> Document:
        source = self._documents[index]
        metadata = dict(source.metadata)
        metadata["school_nature"] = infer_school_nature(metadata.get("record", {})) or None
        metadata.update(
            {
                "hybrid_score": round(hybrid_score, 6),
                "bm25_score": round(bm25_score, 6),
                "cosine_score": round(cosine_score, 6),
                "rerank_score": None if rerank_score is None else round(rerank_score, 6),
                "retrieval_route": route,
            }
        )
        return Document(id=source.id, page_content=source.page_content, metadata=metadata)

    def _diversify(self, docs: list[int], limit: int, key: Any) -> list[int]:
        """按 key 分桶后轮转取样，保证每个桶都被覆盖。

        - 实体列举用 (科类, 批次, 专业组)：否则结果会整个落进同一个专业组，
          模型据此回答"该校只有第101组"。
        - 城市查询用 院校名称：否则 40 条可能全来自同一所学校。
        - 位次查询用 (冲稳保档位, 院校名称)：保证三档都有代表、且不挤在一所学校。

        docs 的传入顺序会保留为桶内顺序（即相关性顺序）。key 接收的是**文档下标**。
        """

        buckets: dict[tuple, list[int]] = {}
        for index in docs:
            buckets.setdefault(key(index), []).append(index)

        ordered = list(buckets.values())
        picked: list[int] = []
        position = 0
        while len(picked) < limit:
            added = False
            for bucket in ordered:
                if position < len(bucket):
                    picked.append(bucket[position])
                    added = True
                    if len(picked) >= limit:
                        break
            if not added:
                break
            position += 1
        return picked

    def _hybrid_search(self, query: str) -> list[Document]:
        mentioned_subjects = [subject for subject in ("物理", "历史") if subject in query]
        exact_subject = mentioned_subjects[0] if len(mentioned_subjects) == 1 else None
        expansion = 3 if exact_subject else 1
        candidate_k = min(self.candidate_k * expansion, len(self._documents))

        query_tokens = list(dict.fromkeys(tokenize_zh(query)))
        core = core_tokens(query)
        bm25_raw: dict[int, float] = {}
        phrase_len: dict[int, int] = {}
        connection = sqlite3.connect(self.index_dir / BM25_DB_FILE)
        try:
            # (1) 邻接子词组查询。
            # 索引里 token 按文档顺序以空格分隔、jieba 结果原样相邻，所以
            # 「院校名称: 江西应用科技学院」存成 … 院校名称 江西 应用 科技 学院 …。
            # jieba 会把校名切成 江西/应用/科技/学院 这类高频词，只用 OR 查询会被
            # "科技学院"类噪声淹没（实测 OR 命中全库 39086 条、目标 0 条）。
            # 这里对所有长度 2~5 的**连续子序列**做词组查询，因此即使实体名嵌在
            # 「我想报江西应用科技学院」这样的长句里，也能精确锁定。
            # 只取长度 >= 3 的连续子序列：长度为 2 的词组（如「江西 应用」）重合度太高，
            # 会给大量无关院校加权，实测反而把精确命中的目标校挤出前 8。
            for span in range(min(len(core), 5), 2, -1):
                for start in range(len(core) - span + 1):
                    expression = (
                        '"' + " ".join(t.replace('"', '""') for t in core[start : start + span]) + '"'
                    )
                    for index, score in connection.execute(
                        "SELECT doc_index, bm25(bm25_index) AS score "
                        "FROM bm25_index WHERE bm25_index MATCH ? ORDER BY score LIMIT ?",
                        (expression, candidate_k),
                    ).fetchall():
                        index = int(index)
                        value = -float(score)
                        if value > bm25_raw.get(index, float("-inf")):
                            bm25_raw[index] = value
                        if span > phrase_len.get(index, 0):
                            phrase_len[index] = span

            # (2) OR 兜底：保证召回率。单独使用时排序质量差，但能补上词组未覆盖的条目。
            match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in query_tokens)
            if match_query:
                for index, score in connection.execute(
                    "SELECT doc_index, bm25(bm25_index) AS score "
                    "FROM bm25_index WHERE bm25_index MATCH ? ORDER BY score LIMIT ?",
                    (match_query, candidate_k),
                ).fetchall():
                    # SQLite FTS5 的 bm25() 越小越相关（负数）；取相反数后越大越好。
                    bm25_raw.setdefault(int(index), -float(score))
        finally:
            connection.close()

        vector_raw: dict[int, float] = {}
        if self._index is not None and self.embeddings is not None:
            query_vector = np.asarray([self.embeddings.embed_query(query)], dtype="float32")
            faiss.normalize_L2(query_vector)
            vector_scores, vector_indices = self._index.search(query_vector, candidate_k)
            vector_raw = {
                int(index): float(score)
                for score, index in zip(vector_scores[0], vector_indices[0], strict=False)
                if index >= 0
            }

        bm25 = _minmax(bm25_raw)
        vector = _minmax(vector_raw)
        combined: list[tuple[float, int]] = []
        for index in set(bm25) | set(vector):
            record = self._documents[index].metadata.get("record", {})
            if exact_subject and str(record.get("科类", "")) != exact_subject:
                continue
            if vector:
                score = self.bm25_weight * bm25.get(index, 0.0) + self.vector_weight * vector.get(index, 0.0)
            else:
                score = bm25.get(index, 0.0)
            # 邻接词组命中是很强的"精确实体"信号（例如整串校名），命中越长越可信：
            # span=3 给 1 份、span=4 给 2 份、span>=5 给 3 份加权。
            # 不加这个权重的话，0.55×余弦 会压过 0.45×BM25，把语义相近但并非同一实体的
            # "江西应用工程职业学院"排到"江西应用科技学院"前面。
            if index in phrase_len:
                score += self.phrase_bonus * min(phrase_len[index] - 2, 3)
            combined.append((score, index))
        # 排序分两层：先看是否有 span>=4 的邻接词组命中，再比融合分。
        # span>=4 的连续命中基本等同于"用户报出了完整实体名"（例如整串校名
        # 「江西 应用 科技 学院」），必须优先于"语义相近但并非同一实体"的结果
        # （如「广州应用科技学院」只命中 span=3 的「应用 科技 学院」，
        #  「江西应用技术职业学院」则完全靠向量分），否则目标校会被挤出前 8。
        combined.sort(
            key=lambda item: (1 if phrase_len.get(item[1], 0) >= 4 else 0, item[0]),
            reverse=True,
        )

        selected = self._apply_rerank(query, combined, self.rerank_candidates)
        # 保留"整串实体名命中"的分层优先：重排只在层内决定顺序，否则重排可能把
        # 语义相近但并非同一实体的记录又顶到正主前面。
        selected.sort(
            key=lambda item: (
                1 if phrase_len.get(item[1], 0) >= 4 else 0,
                item[2] if item[2] is not None else item[0],
            ),
            reverse=True,
        )
        return [
            self._build_document(
                index,
                "hybrid_bm25_cosine",
                score,
                bm25.get(index, 0.0),
                vector_raw.get(index, -1.0),
                rerank_score,
            )
            for score, index, rerank_score in selected[: self.top_k]
        ]
