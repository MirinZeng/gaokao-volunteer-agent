from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import AdmissionsRAGAgent
from .config import Settings, get_settings
from .embeddings import BGEM3Embeddings, ZhipuEmbeddings
from .reranker import BGEM3Reranker
from .score_rank import ScoreRankRegistry
from .web_research import WebResearcher
from .excel_loader import ExcelAdmissionsLoader
from .indexer import (
    MANIFEST_FILE,
    build_bm25_only_index,
    build_index,
    rebuild_search_database,
    rebuild_structured_database,
)
from .retriever import HybridExcelRetriever


app = typer.Typer(no_args_is_help=True, help="高考志愿填报混合检索 RAG Agent")
console = Console()


def _emit_json(event: dict) -> None:
    """输出单行 UTF-8 JSON；每一行都可被日志系统独立解析。"""
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def _settings() -> Settings:
    return get_settings()


def _embeddings(settings: Settings) -> ZhipuEmbeddings | BGEM3Embeddings:
    if settings.embedding_backend == "bge_m3":
        # 本地模型，不需要任何 API Key，也不产生费用。
        return BGEM3Embeddings(
            model_path=settings.bge_model_path,
            device=settings.bge_device,
            batch_size=settings.bge_batch_size,
            max_text_chars=settings.embedding_max_text_chars,
            use_fp16=settings.bge_use_fp16,
        )
    return ZhipuEmbeddings(
        api_key=settings.require_zhipu_key(),
        model=settings.zhipu_embedding_model,
        batch_size=settings.embedding_batch_size,
        max_batch_chars=settings.embedding_max_batch_chars,
        max_text_chars=settings.embedding_max_text_chars,
    )


def _reranker(settings: Settings) -> BGEM3Reranker | None:
    if not settings.rerank_enabled:
        return None
    return BGEM3Reranker(
        model_path=settings.reranker_model_path,
        device=settings.bge_device,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
        use_fp16=settings.rerank_use_fp16,
    )


def _score_rank(settings: Settings) -> ScoreRankRegistry | None:
    """加载一分一段表。文件不存在时返回 None，位次路由自动停用。"""

    registry = ScoreRankRegistry.load(settings.score_rank_dir)
    return registry if registry.tables else None


def _retriever(settings: Settings) -> HybridExcelRetriever:
    manifest_path = settings.index_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError("索引尚未建立，请先运行 ingest 或 ingest --bm25-only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = _embeddings(settings) if manifest.get("vector_enabled", True) else None
    return HybridExcelRetriever(
        index_dir=settings.index_dir,
        embeddings=embeddings,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
        top_k=settings.top_k,
        candidate_k=settings.retrieval_candidates,
        phrase_bonus=settings.retrieval_phrase_bonus,
        entity_listing_k=settings.retrieval_entity_listing_k,
        city_listing_k=settings.retrieval_city_listing_k,
        rank_listing_k=settings.rank_listing_k,
        rank_reach_lo=settings.rank_reach_lo,
        rank_reach_hi=settings.rank_reach_hi,
        rank_match_hi=settings.rank_match_hi,
        rank_guarantee_hi=settings.rank_guarantee_hi,
        rank_safety_hi=settings.rank_safety_hi,
        score_rank=_score_rank(settings),
        reranker=_reranker(settings),
        rerank_candidates=settings.rerank_candidates,
    )


def _agent(settings: Settings) -> AdmissionsRAGAgent:
    agent = AdmissionsRAGAgent.create(
        retriever=_retriever(settings),
        api_key=settings.require_deepseek_key(),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=settings.deepseek_temperature,
        reasoning_effort=settings.deepseek_reasoning_effort,
        thinking=settings.deepseek_thinking,
    )
    if settings.web_search_enabled:
        agent.web_researcher = WebResearcher(
            tavily_api_key=settings.tavily_api_key.get_secret_value(),
            max_results=settings.web_search_max_results,
            timeout=settings.web_search_timeout,
        )
    return agent


@app.command("inspect-excel")
def inspect_excel() -> None:
    """检查 Excel 的工作表、行列数与表头。"""
    settings = _settings()
    summary = ExcelAdmissionsLoader(
        settings.excel_path,
        settings.excel_sheet,
        settings.excel_header_row,
    ).summary()
    console.print(f"文件：{summary.path}")
    console.print(f"工作表：{summary.sheet}；数据行：{summary.rows:,}；字段数：{summary.columns}")
    console.print("表头：" + "、".join(summary.headers))
    console.print(f"SHA-256：{summary.sha256}")


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", help="忽略现有向量索引并重新生成"),
    bm25_only: bool = typer.Option(False, "--bm25-only", help="只建立 BM25，不调用智谱 Embedding"),
) -> None:
    """读取 Excel，调用智谱 Embedding API 并建立本地混合检索索引。"""
    settings = _settings()
    loader = ExcelAdmissionsLoader(
        settings.excel_path,
        settings.excel_sheet,
        settings.excel_header_row,
    )
    if bm25_only:
        manifest = build_bm25_only_index(loader=loader, index_dir=settings.index_dir)
    else:
        manifest = build_index(
            loader=loader,
            embeddings=_embeddings(settings),
            index_dir=settings.index_dir,
            batch_size=settings.embedding_batch_size,
            force=force,
        )
    console.print(Panel(json.dumps(manifest, ensure_ascii=False, indent=2), title="索引已就绪"))


@app.command()
def search(query: str, limit: int = typer.Option(8, min=1, max=30)) -> None:
    """只显示混合检索证据，不调用 DeepSeek。"""
    settings = _settings()
    retriever = _retriever(settings)
    retriever.top_k = limit
    documents = retriever.invoke(query)
    table = Table(title=f"Excel 混合检索：{query}")
    table.add_column("行号", justify="right")
    table.add_column("综合", justify="right")
    table.add_column("BM25", justify="right")
    table.add_column("余弦", justify="right")
    table.add_column("院校")
    table.add_column("专业")
    table.add_column("科类")
    for document in documents:
        record = document.metadata.get("record", {})
        table.add_row(
            str(document.metadata.get("excel_row", "")),
            f"{document.metadata.get('hybrid_score', 0):.3f}",
            f"{document.metadata.get('bm25_score', 0):.3f}",
            f"{document.metadata.get('cosine_score', 0):.3f}",
            str(record.get("院校名称", "")),
            str(record.get("专业名称", "")),
            str(record.get("科类", "")),
        )
    console.print(table)


@app.command("build-structured")
def build_structured() -> None:
    """基于现有 documents.jsonl 重建结构化 SQLite，不重算 Embedding。"""
    settings = _settings()
    count = rebuild_structured_database(settings.index_dir)
    console.print(f"结构化招生数据库已建立：{count:,} 条记录")


@app.command("rebuild-search")
def rebuild_search() -> None:
    """重建紧凑 BM25 与结构化 SQLite，不重算 Embedding。"""
    settings = _settings()
    count = rebuild_search_database(settings.index_dir)
    manifest_path = settings.index_dir / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "bm25_text_profile": "core_fields_v1",
            "structured_backend": "sqlite",
            "structured_schema_version": 1,
        })
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    console.print(f"紧凑 BM25 与结构化数据库已重建：{count:,} 条记录")


@app.command()
def ask(question: str) -> None:
    """单次提问，以 JSON Lines 流式输出。"""
    settings = _settings()
    for event in _agent(settings).stream_answer(question):
        _emit_json(event)


@app.command()
def chat() -> None:
    """启动连续对话；每次回答以 JSON Lines 流式输出。"""
    settings = _settings()
    agent = _agent(settings)
    console.print("高考志愿 Agent 已启动。输入 exit 或 quit 结束。")
    while True:
        question = console.input("\n[bold cyan]你：[/bold cyan]").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            for event in agent.stream_answer(question):
                _emit_json(event)
        except Exception as exc:  # CLI 边界：展示可操作错误，不隐藏异常类型。
            console.print(f"[bold red]{type(exc).__name__}：{exc}[/bold red]")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8000, min=1, max=65535, help="监听端口"),
) -> None:
    """启动 FastAPI，并托管已经构建的 Vue 前端。"""
    import uvicorn

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if not frontend_dist.exists():
        raise FileNotFoundError("尚未构建 Vue 前端，请先在 frontend 目录运行 npm run build")
    console.print(f"前端地址：http://{host}:{port}")
    uvicorn.run("gaokao_agent.api:app", host=host, port=port)


def main() -> None:
    try:
        app()
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]{type(exc).__name__}：{exc}[/bold red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
