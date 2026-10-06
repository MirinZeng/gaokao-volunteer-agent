from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from zhipuai import APIRequestFailedError, ZhipuAI


def _is_retryable_api_error(error: BaseException) -> bool:
    if isinstance(error, APIRequestFailedError):
        status_code = getattr(error, "status_code", None)
        return status_code in {408, 409, 429, 500, 502, 503, 504}
    return True


def _unit_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    """多条向量取平均并归一化为单位向量；归一化后内积即余弦相似度。"""

    array = np.asarray(vectors, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    mean = array.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-12:
        mean = mean / norm
    return mean.tolist()


class ZhipuEmbeddings(Embeddings):
    """智谱 embedding API 的 LangChain Embeddings 适配器。

    智谱 embedding-3 的上下文窗口为 8K tokens，单条输入超限会被 400/1210 拒绝。
    仅按"文本条数"二分无法切开单条文本本身，因此这里做三层处理：

    1. 先把每条文本切成不超过 ``max_text_chars`` 的块（块大小，语义上不再是截断长度）；
    2. 所有块统一按 ``batch_size`` / ``max_batch_chars`` 组批调用 API；
    3. 同一条文本的多个块向量取平均并归一化，最终**仍然每条输入返回一个向量**，
       从而保持 documents.jsonl、BM25 的 doc_index 与 FAISS 行号三者的一一对应；
    4. 兜底：若某个块仍被 400 拒绝，就对该块自身二分递归再平均。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        batch_size: int = 32,
        max_batch_chars: int = 12000,
        max_text_chars: int = 3000,
    ):
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_batch_chars = max_batch_chars
        self.max_text_chars = max_text_chars

    @retry(
        retry=retry_if_exception(_is_retryable_api_error),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def _split_text(self, text: str) -> list[str]:
        """把一条文本切成若干块，避免单条超过模型上下文窗口。"""

        limit = max(int(self.max_text_chars), 1)
        if len(text) <= limit:
            return [text]
        return [text[start : start + limit] for start in range(0, len(text), limit)]

    def _embed_adaptive(self, texts: list[str]) -> list[list[float]]:
        """调用一批文本；被 400 拒绝时先在文本之间二分，只剩一条时再切开该条文本。"""

        try:
            return self._embed_batch(texts)
        except APIRequestFailedError as error:
            if getattr(error, "status_code", None) != 400:
                raise
            if len(texts) > 1:
                middle = len(texts) // 2
                return self._embed_adaptive(texts[:middle]) + self._embed_adaptive(texts[middle:])
            # 单条文本仍被拒：切开它自己，再对两半的向量取平均。
            text = texts[0]
            head = len(text) // 2
            if head >= 1:
                left = self._embed_adaptive([text[:head]])
                right = self._embed_adaptive([text[head:]])
                return [_unit_mean([left[0], right[0]])]
            raise RuntimeError(
                "智谱 Embedding 拒绝了长度为 1 的输入，请检查该 Excel 行内容。"
            ) from error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        chunks: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            for chunk in self._split_text(text):
                chunks.append((index, chunk))

        grouped: dict[int, list[list[float]]] = {index: [] for index in range(len(texts))}
        pending: list[str] = []
        pending_index: list[int] = []
        pending_chars = 0

        def flush() -> None:
            nonlocal pending, pending_index, pending_chars
            if not pending:
                return
            for index, vector in zip(
                pending_index, self._embed_adaptive(pending), strict=True
            ):
                grouped[index].append(vector)
            pending = []
            pending_index = []
            pending_chars = 0

        for index, chunk in chunks:
            would_overflow = pending and (
                len(pending) >= self.batch_size
                or pending_chars + len(chunk) > self.max_batch_chars
            )
            if would_overflow:
                flush()
            pending.append(chunk)
            pending_index.append(index)
            pending_chars += len(chunk)
        flush()

        return [_unit_mean(grouped[index]) for index in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


class BGEM3Embeddings(Embeddings):
    """本地 BAAI/bge-m3 适配器（sentence-transformers，可用 GPU）。

    bge-m3 上下文 8192 tokens，与智谱 embedding-3 同级，输出 1024 维。
    沿用与 ZhipuEmbeddings 相同的"切块 + 块向量平均"策略：
    既避免超长文本被静默截断，又保证**每条输入只返回一个向量**，
    从而不破坏 documents.jsonl / BM25 doc_index / FAISS 行号三者的一一对应。

    注意：``self.model`` 是写入 manifest 的模型标识字符串（indexer.py 依赖它），
    真正的模型对象放在 ``self.encoder``。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        batch_size: int = 16,
        max_text_chars: int = 3000,
        max_seq_length: int = 8192,
        use_fp16: bool = True,
        model_id: str = "bge-m3",
    ):
        # 延迟导入：未安装 torch 时不影响使用智谱后端。
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = model_id
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        # RTX 20 系（Turing）的 fp16 张量核心比 fp32 快约 7 倍（实测 13.2 vs 1.9 条/秒），
        # 且逐条余弦相似度最低 0.999804，检索排序无影响；CPU 上保持 fp32。
        self.use_fp16 = bool(use_fp16) and str(device).startswith("cuda")
        model_kwargs = {"torch_dtype": torch.float16} if self.use_fp16 else {}
        self.encoder = SentenceTransformer(
            self.model_path, device=device, model_kwargs=model_kwargs
        )
        self.encoder.max_seq_length = max_seq_length

    def _split_text(self, text: str) -> list[str]:
        limit = max(int(self.max_text_chars), 1)
        if len(text) <= limit:
            return [text]
        return [text[start : start + limit] for start in range(0, len(text), limit)]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.encoder.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        chunks: list[str] = []
        owner: list[int] = []
        for index, text in enumerate(texts):
            for chunk in self._split_text(text):
                chunks.append(chunk)
                owner.append(index)

        grouped: dict[int, list[np.ndarray]] = {index: [] for index in range(len(texts))}
        if chunks:
            vectors = self._encode(chunks)
            for position, index in enumerate(owner):
                grouped[index].append(vectors[position])

        return [_unit_mean(grouped[index]) for index in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0].tolist()
