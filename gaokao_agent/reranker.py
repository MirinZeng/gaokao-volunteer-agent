"""本地 BAAI/bge-reranker-v2-m3 交叉编码器重排。

**为什么需要重排**：召回阶段（BM25 / 向量）本质是"双塔"结构 —— query 与 doc 各自
独立编码再算相似度。它快、能扫全库，但 query 端完全不知道 doc 里具体写了什么，
所以容易把"看起来像"的记录排上来。

重排阶段把 query 与 doc **拼成一对**送进交叉编码器，让模型的注意力在两者之间
自由交叉，能捕捉细粒度语义交互，因此通常能把真正相关的记录顶到前面。
代价是每个候选都要跑一次前向，所以只对前 N 个候选做重排（默认 60），而不是全库。

实测配置：``max_length=512`` 会把每条记录截断到前 ~400 个中文字符。对本项目的数据
这已经覆盖了「院校名称 / 专业名称 / 选科 / 批次」等用于判断相关性的关键字段
（它们都在记录开头），而尾部字段（城市、所在省、历年分数）本身不参与重排判断。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document


class BGEM3Reranker:
    """sentence-transformers CrossEncoder 的薄封装。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = True,
    ):
        # 延迟导入：没装 torch 时不影响纯智谱后端。
        import torch
        from sentence_transformers import CrossEncoder

        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = bool(use_fp16) and str(device).startswith("cuda")
        model_kwargs = {"torch_dtype": torch.float16} if self.use_fp16 else {}
        self.model = CrossEncoder(
            self.model_path,
            device=device,
            max_length=max_length,
            model_kwargs=model_kwargs,
        )

    def rerank(self, query: str, documents: Sequence[Document]) -> list[tuple[int, float]]:
        """返回 ``(documents 下标, 相关度分数)``，按分数降序。

        只返回下标与分数，不重排 Document 本身 —— 调用方自己决定截断到几条、
        以及要把分数写进哪个 metadata 字段。
        """

        if not documents:
            return []
        pairs = [(query, document.page_content) for document in documents]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return sorted(
            ((position, float(score)) for position, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
