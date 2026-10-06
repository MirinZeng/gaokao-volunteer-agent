from __future__ import annotations

import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cli import _agent
from .config import get_settings
from .memory import ConversationMemory


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    web_search: bool = True
    session_id: str | None = None


class SessionCreateRequest(BaseModel):
    title: str = "新的志愿咨询"


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=40)


_agent_build_lock = threading.Lock()
_memory = ConversationMemory(get_settings().index_dir.parent / "memory.sqlite3")


@lru_cache(maxsize=1)
def _build_agent():
    return _agent(get_settings())


def get_agent():
    """线程安全地取 Agent 单例。

    **不能直接给这个函数加 lru_cache**：lru_cache 在调用期间会释放锁，于是缓存被
    填充之前，每个并发请求都会各自重建一次 Agent —— 也就是各自把 Embedding 和重排
    模型重新加载一遍。实测 3 个并发请求会让 4 份模型同时挤进 6GB 显存，单次响应
    从 2 秒劣化到 47~121 秒。这里用显式锁把构建串行化：只有第一个请求真正构建，
    其余等在锁上，随后直接拿缓存。
    """

    with _agent_build_lock:
        return _build_agent()


def _warm_up_agent() -> None:
    """后台预热：提前把索引、Embedding、重排模型加载好。

    否则用户问的第一个问题要多等约 30 秒（实测冷启动 28.97s vs 热态 2.00s）。
    预热失败不影响服务启动，首个真实请求会重新抛出可读的错误。
    """

    try:
        get_agent()
        print("[warmup] 索引与模型预热完成", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"[warmup] 预热失败：{type(error).__name__}: {error}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up_agent, daemon=True).start()
    yield


app = FastAPI(title="高考志愿 Agent API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    # Vite 端口被占用时会自动顺延（5174、5175…），也可能被其他本地静态服务托管；
    # 放行所有本机来源，避免前端回退直连后端时被 CORS 拦掉。
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    manifest_path = settings.index_dir / "manifest.json"
    index_ready = manifest_path.exists()
    retrieval_mode = None
    embedding_model = None
    embedding_dimension = None
    vector_enabled = False
    document_count = None
    if index_ready:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vector_enabled = bool(manifest.get("vector_enabled", True))
        retrieval_mode = "hybrid_bm25_cosine" if vector_enabled else "bm25"
        embedding_model = manifest.get("embedding_model")
        embedding_dimension = manifest.get("embedding_dimension")
        document_count = manifest.get("document_count")

    # 索引里的向量必须与当前后端是同一个模型，否则查询向量与库内向量不在同一空间。
    expected_model = (
        "bge-m3" if settings.embedding_backend == "bge_m3" else settings.zhipu_embedding_model
    )
    embedding_matches = not embedding_model or embedding_model == expected_model

    return {
        "status": "ok",
        "index_ready": index_ready,
        "retrieval_mode": retrieval_mode,
        "model": settings.deepseek_model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_expected_model": expected_model,
        "embedding_matches": embedding_matches,
        "vector_enabled": vector_enabled,
        "document_count": document_count,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "bge_device": settings.bge_device if settings.embedding_backend == "bge_m3" else None,
        "bge_use_fp16": settings.bge_use_fp16 if settings.embedding_backend == "bge_m3" else None,
        "chunk_chars": settings.embedding_max_text_chars,
        "web_search_enabled": settings.web_search_enabled,
        "web_search_provider": "tavily" if settings.tavily_api_key.get_secret_value() else "duckduckgo",
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    session = _memory.ensure_session(request.session_id)
    session_id = str(session["id"])
    memory_context = _memory.context(session_id)
    question = request.message.strip()
    _memory.add_message(session_id, "user", question)
    if int(session.get("message_count") or 0) == 0:
        _memory.rename_session(session_id, question[:24])

    def generate() -> Iterator[str]:
        final_event: dict[str, Any] | None = None
        intent_plan: dict[str, Any] | None = None
        try:
            for event in get_agent().stream_answer(
                question, web_enabled=request.web_search, memory_context=memory_context
            ):
                event["session_id"] = session_id
                if event.get("type") == "start":
                    intent_plan = event.get("intent")
                    _memory.update_profile(session_id, intent_plan)
                if event.get("type") in {"result", "error"}:
                    final_event = event
                yield json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        except Exception as error:
            message = str(error)
            if "insufficient balance" in message.lower() or "error code: 402" in message.lower():
                message = "DeepSeek API 余额不足，请充值或更换有效 API Key。"
            final_event = {
                "type": "error", "status": "failed", "message": message,
                "session_id": session_id,
            }
            yield json.dumps(
                final_event,
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
        finally:
            if final_event is not None:
                content = str(final_event.get("answer") or final_event.get("message") or "")
                if final_event.get("blocks"):
                    content = json.dumps(final_event["blocks"], ensure_ascii=False)
                _memory.add_message(session_id, "assistant", content, final_event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    return {"sessions": _memory.list_sessions()}


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    return _memory.create_session(request.title)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        return {
            "session": _memory.get_session(session_id),
            "messages": _memory.messages(session_id),
            "memory": _memory.context(session_id),
        }
    except KeyError:
        return {"session": None, "messages": [], "memory": None}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, request: SessionRenameRequest) -> dict[str, Any]:
    _memory.rename_session(session_id, request.title)
    return _memory.get_session(session_id)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    _memory.delete_session(session_id)
    return {"deleted": True}


# ------------------------------ 索引进度可视化 ------------------------------

DASHBOARD_FILE = Path(__file__).resolve().parent / "static" / "progress.html"
_PROGRESS_HISTORY: deque[tuple[float, int]] = deque(maxlen=400)
_PROGRESS_WINDOW_SECONDS = 300


def _ingest_progress() -> dict[str, Any]:
    """读取 Embedding 检查点，给出进度、速率与预计剩余时间。"""

    settings = get_settings()
    index_dir = settings.index_dir
    state_path = index_dir / "embedding_checkpoint.json"
    manifest_path = index_dir / "manifest.json"

    manifest_data: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_data = {}
    vector_enabled = bool(manifest_data.get("vector_enabled"))

    progress: dict[str, Any] = {
        "state": "idle",
        "completed": 0,
        "total": 0,
        "percent": 0.0,
        "rate": 0.0,
        "eta_seconds": None,
        "model": None,
        "dimension": None,
        "chunk_chars": settings.embedding_max_text_chars,
        "backend": settings.embedding_backend,
        "vector_enabled": vector_enabled,
        "checkpoint_age_seconds": None,
        "history": [],
    }

    if not state_path.exists():
        # 构建成功后 indexer 会清理检查点文件；此时依据 manifest 判断索引是否已就绪，
        # 否则仪表盘会错误地显示"尚未开始"。
        if vector_enabled:
            count = int(manifest_data.get("document_count") or 0)
            progress.update(
                {
                    "state": "finished",
                    "completed": count,
                    "total": count,
                    "percent": 100.0,
                    "model": manifest_data.get("embedding_model"),
                    "dimension": manifest_data.get("embedding_dimension"),
                }
            )
        return progress

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return progress

    total = int(state.get("document_count") or 0)
    completed = int(state.get("completed") or 0)
    progress["completed"] = completed
    progress["total"] = total
    progress["percent"] = round(completed / total * 100, 2) if total else 0.0
    progress["model"] = state.get("embedding_model")
    progress["dimension"] = state.get("embedding_dimension")

    # 检查点每批（32 条）重写一次；超过 90 秒没动就认为已暂停或中断。
    age = time.time() - state_path.stat().st_mtime
    progress["checkpoint_age_seconds"] = round(age, 1)
    if total and completed >= total:
        progress["state"] = "finished"
    elif age < 90:
        progress["state"] = "running"
    else:
        progress["state"] = "paused"

    if progress["state"] != "running":
        return progress

    # 采样历史只在本接口被轮询时推进，因此仪表盘打开后曲线才开始积累。
    now = time.time()
    if not _PROGRESS_HISTORY or _PROGRESS_HISTORY[-1][1] != completed:
        _PROGRESS_HISTORY.append((now, completed))
    while _PROGRESS_HISTORY and now - _PROGRESS_HISTORY[0][0] > _PROGRESS_WINDOW_SECONDS:
        _PROGRESS_HISTORY.popleft()

    if len(_PROGRESS_HISTORY) >= 2:
        origin = _PROGRESS_HISTORY[0][0]
        span = _PROGRESS_HISTORY[-1][0] - origin
        delta = _PROGRESS_HISTORY[-1][1] - _PROGRESS_HISTORY[0][1]
        if span > 1 and delta > 0:
            rate = delta / span
            progress["rate"] = round(rate, 2)
            progress["eta_seconds"] = round((total - completed) / rate, 1)
        progress["history"] = [
            {"t": round(stamp - origin, 1), "done": count}
            for stamp, count in _PROGRESS_HISTORY
        ]
    return progress


@app.get("/api/ingest-progress")
def ingest_progress() -> dict[str, Any]:
    """Embedding 索引构建进度，供 /progress 仪表盘轮询。"""

    return _ingest_progress()


@app.get("/progress", response_class=HTMLResponse)
def progress_dashboard() -> HTMLResponse:
    """实时索引进度仪表盘。"""

    if DASHBOARD_FILE.exists():
        return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>缺少 gaokao_agent/static/progress.html</h1>", status_code=404)


frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
