from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """项目配置；优先读取环境变量，其次读取项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: SecretStr = Field(default=SecretStr(""))
    zhipu_api_key: SecretStr = Field(default=SecretStr(""))
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_temperature: float = 0.4
    # DeepSeek 思考模式：思维链通过 reasoning_content 返回，前端可展示。
    # effort 实测档位：minimal/low → low，medium/high/xhigh → high，max/ultra → max。
    # 注意思考模式下 temperature 会被忽略（传入不报错但不生效）。
    deepseek_thinking: bool = True
    deepseek_reasoning_effort: str = "high"
    zhipu_embedding_model: str = "embedding-3"
    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_search_timeout: int = 15

    # Embedding 后端选择：
    #   zhipu  = 智谱云端 embedding-3（2048 维，按 token 收费）
    #   bge_m3 = 本地 BAAI/bge-m3（1024 维，免费、可离线、支持 GPU）
    embedding_backend: str = "zhipu"
    bge_model_path: Path = Path(r"D:\models\bge-m3")
    bge_device: str = "cuda"
    bge_batch_size: int = 16
    # fp16 在 Turing 及以上 GPU 上约快 7 倍，精度损失可忽略（实测余弦 ≥0.9998）
    bge_use_fp16: bool = True

    excel_path: Path = Path("江西-2025高报数据参考-家长版.xlsx")
    index_dir: Path = Path("data/index")
    excel_sheet: str = "Sheet1"
    excel_header_row: int = 2

    bm25_weight: float = 0.45
    vector_weight: float = 0.55
    # 混合检索最终交给模型的证据条数。原为 8，配合重排上调到 12。
    top_k: int = 12
    # 融合前的候选池：BM25 取前 N、向量取前 N，并集后参与融合与重排。
    # 原为 40，上调到 60 —— 重排要有足够候选才有发挥空间。
    retrieval_candidates: int = 60
    # 邻接词组命中（如整串校名）的额外加权基数：
    # span=3 加 1 份、span=4 加 2 份、span>=5 加 3 份
    retrieval_phrase_bonus: float = 0.15

    # ---- 交叉编码器重排 ----
    # 召回是"双塔"结构精度有限，重排把 query 与 doc 拼在一起过模型的交叉编码器，
    # 能显著提升真正相关记录的排名。代价是每个候选一次前向。
    rerank_enabled: bool = True
    reranker_model_path: Path = Path(r"D:\models\bge-reranker-v2-m3")
    rerank_candidates: int = 60
    rerank_batch_size: int = 16
    # 512 会截断到记录前 ~400 中文字符，刚好覆盖院校名称/专业名称/选科/批次，
    # 而尾部字段（城市、历年分数）本身不参与相关性判断。调大可提高精度但更慢。
    rerank_max_length: int = 512
    rerank_use_fp16: bool = True
    # 实体意图且没有额外检索词时（如「某某学院有哪些专业」）返回的最大证据条数。
    # 这不是 top_k：整校记录必须尽量给全，否则模型会误判"表中没有其他专业组"。
    # 实测上限 80 可完整覆盖 99% 的院校（中位数仅 13 条），只有 1% 的大校会触顶。
    retrieval_entity_listing_k: int = 80
    # 城市意图的证据上限（如「南昌有哪些软件工程专业」实测涉及 35 所院校 / 94 条记录，
    # 旧的 top_k=8 只够给出 5 所）
    retrieval_city_listing_k: int = 40

    # ---- 分数 → 位次 换算与按位次选院校 ----
    # 一分一段表目录（由官方 PDF 抽取而来的 CSV：物理.csv / 历史.csv）
    score_rank_dir: Path = Path("data/score_rank")
    rank_listing_k: int = 40
    # 冲稳保分档，判据 k = 该校最近一年最低位次 / 你的位次
    rank_reach_lo: float = 0.70
    rank_reach_hi: float = 0.90
    rank_match_hi: float = 1.10
    # 1.10~1.25 为保，1.25~1.40 为垫；总上限仍严格控制在 1.40。
    rank_guarantee_hi: float = 1.25
    # 保底不能拉得过远：3 万位最多看到约 4.2 万位，绝不扩到 10 万位。
    rank_safety_hi: float = 1.40
    embedding_batch_size: int = 32
    embedding_max_batch_chars: int = 12000
    # 单条文本的切块大小（不是截断长度）。智谱 embedding-3 上下文窗口为 8K tokens，
    # 中文约 1.7 tokens/字符，实测单条超过约 4200 字符即报 400/1210，故取 3000 留足余量。
    embedding_max_text_chars: int = 3000

    @field_validator(
        "excel_path", "index_dir", "bge_model_path", "reranker_model_path", "score_rank_dir",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> "Settings":
        if self.embedding_backend not in {"zhipu", "bge_m3"}:
            raise ValueError("EMBEDDING_BACKEND 只能是 zhipu 或 bge_m3")
        if abs(self.bm25_weight + self.vector_weight - 1.0) > 1e-6:
            raise ValueError("BM25_WEIGHT 与 VECTOR_WEIGHT 之和必须为 1")
        if min(self.bm25_weight, self.vector_weight) < 0:
            raise ValueError("检索权重不能为负数")
        if not 0 <= self.deepseek_temperature <= 2:
            raise ValueError("DEEPSEEK_TEMPERATURE 必须位于 0 到 2 之间")
        if self.top_k < 1 or self.retrieval_candidates < self.top_k:
            raise ValueError("RETRIEVAL_CANDIDATES 必须不小于 TOP_K，且 TOP_K 至少为 1")
        if (
            self.excel_header_row < 1
            or self.embedding_batch_size < 1
            or self.embedding_max_batch_chars < 1
            or self.embedding_max_text_chars < 1
        ):
            raise ValueError("表头行号和批大小必须为正整数")
        return self

    def require_deepseek_key(self) -> str:
        value = self.deepseek_api_key.get_secret_value().strip()
        if not value or value == "YOUR_DEEPSEEK_API_KEY_HERE":
            raise RuntimeError("请先在 .env 中填写 DEEPSEEK_API_KEY")
        return value

    def require_zhipu_key(self) -> str:
        value = self.zhipu_api_key.get_secret_value().strip()
        if not value or value == "YOUR_ZHIPU_API_KEY_HERE":
            raise RuntimeError("请先在 .env 中填写 ZHIPU_API_KEY")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
