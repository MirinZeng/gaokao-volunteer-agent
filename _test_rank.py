"""测试分数 → 位次 → 冲稳保选校的完整链路。"""

import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from gaokao_agent.cli import _retriever
from gaokao_agent.config import get_settings

retriever = _retriever(get_settings())
print(f"一分一段表: {retriever.score_rank.subjects if retriever.score_rank else '未加载'}")
print(f"位次索引  : " + ", ".join(
    f"{subject} {len(rows)} 条" for subject, rows in retriever._rank_index.items()
))
print()

CASES = [
    "我考了600分，物理类，能上什么大学",
    "历史类550分 想学汉语言文学",
    "物理类620分想学计算机",
    "600分能上什么学校",
    "江西师范大学有哪些专业",
    "南昌有哪些软件工程专业",
]

for query in CASES:
    intent = retriever.plan(query)
    started = time.time()
    documents = retriever.invoke(query)
    elapsed = (time.time() - started) * 1000
    route = documents[0].metadata.get("retrieval_route") if documents else "-"

    print("=" * 76)
    print(f"查询：{query}")
    print("=" * 76)
    print(f"  意图={intent.kind}  科类={intent.subject}  分数={intent.score}  剩余词={list(intent.remaining)}")
    if intent.rank_lookup:
        print(f"  换算：{intent.rank_lookup.describe()}")
    print(f"  路由={route}  返回 {len(documents)} 条  ({elapsed:.0f} ms)")

    if route == "rank_bm25":
        bands = Counter(d.metadata.get("rank_band") for d in documents)
        print(f"  冲稳保分布: {dict(bands)}")
        print(f"  {'档':<4}{'院校':<22}{'专业':<20}{'该校位次':>9}{'k':>7}")
        for document in documents[:8]:
            record = document.metadata.get("record", {})
            print(
                f"  {document.metadata.get('rank_band',''):<4}"
                f"{str(record.get('院校名称',''))[:20]:<22}"
                f"{str(record.get('专业名称',''))[:18]:<20}"
                f"{document.metadata.get('doc_rank',0):>9,}"
                f"{document.metadata.get('rank_ratio',0):>7.2f}"
            )
    print()
