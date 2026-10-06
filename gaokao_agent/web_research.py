from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html import unescape


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
)
OFFICIAL_HINTS = (
    "gov.cn", "edu.cn", "chsi.com.cn", "neea.edu.cn", "jxeea.cn",
    "bjeea.cn", "zjzs.net", "shmeea.edu.cn", "zsks.cn",
)


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    snippet: str
    official: bool

    def payload(self) -> dict:
        return asdict(self)


def should_web_research(question: str, intent: dict | None = None) -> bool:
    """只有时效性/政策性问题才联网；历史 Excel 事实优先走本地 RAG。"""

    text = question.lower()
    triggers = (
        "最新", "今年", "2026", "政策", "招生章程", "招生计划", "计划变化",
        "选科要求", "体检要求", "调剂规则", "退档", "滑档", "官方", "考试院",
        "新增专业", "停招", "学费变化",
        "就业", "好就业", "保证就业", "就业率", "行业需求", "就业方向",
    )
    if any(word in text for word in triggers):
        return True
    goal = str((intent or {}).get("goal") or "")
    return goal in {
        "policy_question", "school_recommendation", "school_comparison", "major_selection"
    }


def _clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", value))).strip()


def _safe_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class WebResearcher:
    def __init__(self, tavily_api_key: str = "", max_results: int = 5, timeout: int = 15):
        self.tavily_api_key = tavily_api_key.strip()
        self.max_results = max(1, min(int(max_results), 8))
        self.timeout = max(3, int(timeout))

    def _request(self, request: urllib.request.Request) -> str:
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")

    def _tavily(self, query: str) -> list[WebSource]:
        body = json.dumps({
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
            "include_answer": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=body,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        )
        payload = json.loads(self._request(request))
        return self._normalize(payload.get("results", []))

    def _duckduckgo(self, query: str) -> list[WebSource]:
        body = urllib.parse.urlencode({"q": query, "kl": "cn-zh"}).encode("utf-8")
        request = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=body,
            headers={"User-Agent": USER_AGENT},
        )
        html = self._request(request)
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
        )
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        rows = []
        for index, (href, title) in enumerate(links[: self.max_results]):
            parsed = urllib.parse.urlparse(href)
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
            rows.append({
                "title": _clean_html(title),
                "url": target,
                "content": _clean_html(snippets[index]) if index < len(snippets) else "",
            })
        return self._normalize(rows)

    def _normalize(self, rows: list[dict]) -> list[WebSource]:
        results: list[WebSource] = []
        seen: set[str] = set()
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not _safe_http_url(url) or url in seen:
                continue
            seen.add(url)
            host = urllib.parse.urlparse(url).netloc.lower()
            results.append(WebSource(
                title=str(row.get("title") or host).strip()[:180],
                url=url,
                snippet=str(row.get("content") or row.get("snippet") or "").strip()[:800],
                official=any(hint in host for hint in OFFICIAL_HINTS),
            ))
        return sorted(results, key=lambda item: item.official, reverse=True)[: self.max_results]

    def search(self, query: str) -> list[WebSource]:
        if self.tavily_api_key:
            try:
                results = self._tavily(query)
                if results:
                    return results
            except Exception:  # noqa: BLE001 - 自动回退免费搜索
                pass
        return self._duckduckgo(query)


def format_web_context(sources: list[WebSource]) -> str:
    if not sources:
        return "（未获得可核验的联网来源）"
    return "\n\n".join(
        f"[网页来源{index}] {source.title}\nURL: {source.url}\n"
        f"来源级别: {'官方/高校域名' if source.official else '第三方待核验'}\n"
        f"摘要: {source.snippet or '无摘要'}"
        for index, source in enumerate(sources, start=1)
    )
