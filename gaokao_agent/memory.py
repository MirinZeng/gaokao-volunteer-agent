from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


PROFILE_FIELDS = (
    "province", "year", "batch", "subject", "score", "rank", "cities",
    "majors", "rejected_majors", "budget", "risk_preference", "adjustment",
    "accept_private", "accept_joint_program", "priorities",
)


class ConversationMemory:
    """SQLite 四层会话记忆：元数据、结构化档案、滚动摘要、滑动窗口。"""

    def __init__(self, path: Path, window_messages: int = 12, summary_chars: int = 4000):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window_messages = max(4, int(window_messages))
        self.summary_chars = max(1000, int(summary_chars))
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_until INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);
                """
            )

    def create_session(self, title: str = "新的志愿咨询") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (session_id, title[:40] or "新的志愿咨询", now, now),
            )
            connection.execute(
                "INSERT INTO profiles(session_id,profile_json,updated_at) VALUES(?,?,?)",
                (session_id, "{}", now),
            )
            connection.execute(
                "INSERT INTO summaries(session_id,summary,summarized_until,updated_at) VALUES(?,?,?,?)",
                (session_id, "", 0, now),
            )
        return self.get_session(session_id)

    def ensure_session(self, session_id: str | None) -> dict[str, Any]:
        if session_id:
            try:
                return self.get_session(session_id)
            except KeyError:
                pass
        return self.create_session()

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            return dict(row)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,title,created_at,updated_at,message_count FROM sessions "
                "ORDER BY updated_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title=?,updated_at=? WHERE id=?",
                (title.strip()[:40] or "新的志愿咨询", time.time(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def add_message(
        self, session_id: str, role: str, content: str, payload: dict[str, Any] | None = None
    ) -> int:
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id,role,content,payload_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, role, content, json.dumps(payload or {}, ensure_ascii=False), now),
            )
            connection.execute(
                "UPDATE sessions SET updated_at=?,message_count=message_count+1 WHERE id=?",
                (now, session_id),
            )
            message_id = int(cursor.lastrowid)
        self._refresh_summary(session_id)
        return message_id

    def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id,role,content,payload_json,created_at FROM messages WHERE session_id=?"
        params: list[Any] = [session_id]
        if limit:
            sql = (
                "SELECT * FROM (" + sql + " ORDER BY id DESC LIMIT ?) recent ORDER BY id"
            )
            params.append(limit)
        else:
            sql += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except json.JSONDecodeError:
                item["payload"] = {}
            result.append(item)
        return result

    def update_profile(self, session_id: str, intent: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM profiles WHERE session_id=?", (session_id,)
            ).fetchone()
            profile = json.loads(row[0]) if row else {}
            source = dict((intent or {}).get("profile") or {})
            direct = {
                "subject": (intent or {}).get("subject"),
                "score": (intent or {}).get("score"),
                "rank": (intent or {}).get("rank"),
                "majors": (intent or {}).get("target_major"),
                "priorities": (intent or {}).get("preferences"),
            }
            source.update({key: value for key, value in direct.items() if value not in (None, "", [])})
            for key in PROFILE_FIELDS:
                value = source.get(key)
                if value not in (None, "", []):
                    profile[key] = value
            connection.execute(
                "INSERT INTO profiles(session_id,profile_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET profile_json=excluded.profile_json,"
                "updated_at=excluded.updated_at",
                (session_id, json.dumps(profile, ensure_ascii=False), time.time()),
            )
        return profile

    def context(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            session = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            profile = connection.execute(
                "SELECT profile_json FROM profiles WHERE session_id=?", (session_id,)
            ).fetchone()
            summary = connection.execute(
                "SELECT summary FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        if session is None:
            raise KeyError(session_id)
        window = self.messages(session_id, self.window_messages)
        return {
            "metadata": dict(session),
            "profile": json.loads(profile[0]) if profile else {},
            "summary": summary[0] if summary else "",
            "window": [{"role": row["role"], "content": row["content"]} for row in window],
        }

    def _refresh_summary(self, session_id: str) -> None:
        """把滑动窗口之外的旧消息压入滚动摘要，避免上下文无限增长。"""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id,role,content FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
            cutoff = max(0, len(rows) - self.window_messages)
            old = rows[:cutoff]
            if not old:
                return
            previous = connection.execute(
                "SELECT summarized_until FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
            summarized_until = int(previous[0]) if previous else 0
            additions = [row for row in old if int(row["id"]) > summarized_until]
            if not additions:
                return
            fragments = []
            for row in additions:
                label = "用户" if row["role"] == "user" else "助手"
                compact = " ".join(str(row["content"]).split())[:500]
                fragments.append(f"{label}：{compact}")
            old_summary = connection.execute(
                "SELECT summary FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
            merged = ((old_summary[0] if old_summary else "") + "\n" + "\n".join(fragments)).strip()
            merged = merged[-self.summary_chars :]
            connection.execute(
                "UPDATE summaries SET summary=?,summarized_until=?,updated_at=? WHERE session_id=?",
                (merged, int(additions[-1]["id"]), time.time(), session_id),
            )
