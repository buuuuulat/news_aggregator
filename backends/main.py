# main.py
import os
import sys
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, validator
from urllib.parse import urlparse

# ---------------------------
# Paths & imports
# ---------------------------

APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "templates/index.html"

# Import RSS parser (твой файл rss_parser.py, который использует feedparser)
try:
    from rss_parser import parse_websites
except Exception:
    parse_websites = None  # type: ignore

# Import summarizer from ../summarizer/model.py
SUMM_DIR = (APP_DIR / ".." / "summarizer").resolve()
if str(SUMM_DIR) not in sys.path:
    sys.path.append(str(SUMM_DIR))
try:
    # summarize: для одного текста (совместимо с твоим API)
    # summarize_many: батч на несколько статей сразу (ускоряет)
    from model import summarize as ru_summarize, summarize_many as ru_summarize_many  # type: ignore
except Exception:
    ru_summarize = None   # type: ignore
    ru_summarize_many = None  # type: ignore


# ---------------------------
# FastAPI setup
# ---------------------------

app = FastAPI(title="News Aggregator (RSS → HTML summaries + SSE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Pydantic models
# ---------------------------

class RSSLinksRequest(BaseModel):
    """
    Принимает массив ссылок или одну строку c ссылками
    (разделители — новая строка или запятая).
    """
    urls: Union[List[str], str]
    latest_n: int = 100          # по умолчанию берём 100
    keywords: List[str] = []

    @validator("urls", pre=True)
    def _coerce_urls(cls, v):
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace(",", "\n").splitlines()]
            return [p for p in parts if p]
        return v

    @validator("latest_n", pre=True, always=True)
    def _cap_latest_n(cls, v):
        try:
            n = int(v)
        except Exception:
            n = 100
        # мягкий верхний лимит
        return max(1, min(n, 300))


# ---------------------------
# Helpers: URLs
# ---------------------------

def _ensure_scheme(u: str) -> str:
    if not u.lower().startswith(("http://", "https://")):
        return "https://" + u
    return u

def _normalize_urls(raw: List[str]) -> List[str]:
    urls: List[str] = []
    for u in raw:
        u = _ensure_scheme(u.strip())
        parsed = urlparse(u)
        if parsed.netloc:
            urls.append(u)
    # dedup с сохранением порядка
    return list(dict.fromkeys(urls))


# ---------------------------
# Helpers: Feed entries
# ---------------------------

def _get_entry_link(e: Any) -> str:
    get = (e.get if isinstance(e, dict) else lambda k, default=None: getattr(e, k, default))
    return get("link", "") or get("url", "") or ""

def _get_entry_title(e: Any) -> str:
    get = (e.get if isinstance(e, dict) else lambda k, default=None: getattr(e, k, default))
    return get("title", "") or "(без заголовка)"

def _get_entry_published(e: Any) -> str:
    get = (e.get if isinstance(e, dict) else lambda k, default=None: getattr(e, k, default))
    return get("published", "") or get("updated", "") or ""

def _get_entry_inline_content(e: Any) -> str:
    """
    Берём то, что уже пришло в RSS: summary/content, если есть.
    """
    if isinstance(e, dict):
        if "summary" in e and e["summary"]:
            return e["summary"]
        if "content" in e and e["content"]:
            try:
                return " ".join([c.get("value", "") for c in e["content"] if isinstance(c, dict)])
            except Exception:
                pass
    return ""


# ---------------------------
# Helpers: Article extraction & summarization
# ---------------------------

def _extract_article_text(url: str, fallback: str = "") -> str:
    """
    Пытаемся скачать и вытащить основной текст новости.
    1) trafilatura (если установлена)
    2) requests + BeautifulSoup
    3) fallback (summary/контент из RSS)
    """
    if not url:
        return fallback

    # 1) trafilatura
    try:
        import trafilatura  # type: ignore
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_images=False)
            if extracted:
                return extracted
    except Exception:
        pass

    # 2) requests + bs4
    try:
        from bs4 import BeautifulSoup  # type: ignore
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.ok:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            container = soup.find("article") or soup
            paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
            text = "\n".join([p for p in paragraphs if p])
            if text:
                return text
    except Exception:
        pass

    return fallback

def _summarize_text(text: str) -> str:
    """
    Обёртка над summarize из ../summarizer/model.py.
    Если модели нет или произошла ошибка — вернём 2–3 предложения исходного текста.
    """
    if not text:
        return ""
    try:
        if ru_summarize is not None:
            parts = ru_summarize(text, max_new_tokens=120)  # твоя функция возвращает list[str]
            return "\n".join(p for p in parts if p).strip()
    except Exception:
        pass

    # Fallback — первые 2–3 предложения
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sents[:3]).strip()


# ---------------------------
# Legacy JSON/text formatting (оставим для совместимости)
# ---------------------------

def _fetch_news_dict(
    urls: List[str],
    latest_n: int,
    keywords: List[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    Возвращает:
      ok: {url: [ {title,url,published,summary}, ... ]}
      errors: {url: "msg"}
    """
    ok: Dict[str, List[Dict[str, Any]]] = {}
    errors: Dict[str, str] = {}

    if not urls:
        return ok, {"input": "не переданы валидные ссылки"}

    if parse_websites is None:
        for u in urls:
            errors[u] = "rss_parser.parse_websites не импортирован"
        return ok, errors

    try:
        raw = parse_websites(urls, keywords, latest_n=latest_n) or {}
        for u, entries in raw.items():
            news_list: List[Dict[str, Any]] = []
            for e in entries or []:
                get = (e.get if isinstance(e, dict) else lambda k, default=None: getattr(e, k, default))
                news_list.append({
                    "title": get("title", "") or "",
                    "url": get("link", "") or get("url", "") or "",
                    "published": get("published", "") or get("updated", "") or "",
                    "summary": get("summary", "") or "",
                })
            ok[u] = news_list
    except Exception as ex:
        for u in urls:
            errors[u] = f"parse error: {ex}"

    for u in urls:
        if u not in ok:
            errors.setdefault(u, "no entries or unsupported feed")

    return ok, errors


# ---------------------------
# HTML non-stream renderer (опционально)
# ---------------------------

def _format_html_cards(raw_by_src: Dict[str, List[Any]]) -> str:
    """Возвращает HTML-фрагмент с карточками новостей по источникам."""
    lines: List[str] = []
    lines.append("""
<style>
.news-group{margin-top:18px}
.news-source{font-weight:700;margin:0 0 8px 0}
.card{border:1px solid #6663;border-radius:12px;padding:12px;margin:10px 0;box-shadow:0 4px 14px #0001}
.card .head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.card a.title{font-weight:700;text-decoration:none}
.card .date{opacity:.7;font-size:.9em}
.card .sum{margin-top:8px;line-height:1.5;white-space:pre-wrap}
</style>
    """)
    for src, items in raw_by_src.items():
        # 1) параллельно вытащим тексты
        texts = _extract_texts_batch(items, max_workers=12)

        # 2) батч-суммаризация
        if ru_summarize_many is not None:
            summaries = ru_summarize_many(texts, max_new_tokens=120, batch_size=8, num_beams=1)
        else:
            summaries = [_summarize_text(t) for t in texts]

        lines.append('<div class="news-group">')
        lines.append(f'<div class="news-source">{html.escape(src)}</div>')
        for e, summary in zip(items, summaries):
            title = _get_entry_title(e)
            link = _get_entry_link(e)
            pub  = _get_entry_published(e)

            lines.append('<div class="card">')
            lines.append('<div class="head">')
            if link:
                lines.append(
                    f'<a class="title" href="{html.escape(link)}" target="_blank" rel="noopener noreferrer">'
                    f'{html.escape(title)}</a>'
                )
            else:
                lines.append(f'<span class="title">{html.escape(title)}</span>')
            if pub:
                lines.append(f'<span class="date">{html.escape(pub)}</span>')
            lines.append('</div>')  # head
            lines.append(f'<div class="sum">{html.escape(summary or "(нет текста для суммаризации)")}</div>')
            lines.append('</div>')  # card
        lines.append('</div>')      # news-group
    return "\n".join(lines)


# ---------------------------
# SSE utilities (streaming)
# ---------------------------

def _sse_pack(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")

def _coerce_urls_str(s: str) -> List[str]:
    parts = [p.strip() for p in s.replace(",", "\n").splitlines()]
    return [p for p in parts if p]

def _extract_texts_batch(entries: List[Any], max_workers: int = 12) -> List[str]:
    """I/O-параллелизм: качаем тексты статей потоками. Порядок сохраняем."""
    texts = ["" for _ in range(len(entries))]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for i, e in enumerate(entries):
            link = _get_entry_link(e)
            inline = _get_entry_inline_content(e)
            futures[ex.submit(_extract_article_text, link, inline)] = i
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                texts[idx] = fut.result()
            except Exception:
                texts[idx] = ""
    return texts


# ---------------------------
# Routes
# ---------------------------

@app.get("/", response_class=FileResponse)
def serve_index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    html_fallback = """
    <!doctype html><meta charset="utf-8">
    <title>News Aggregator</title>
    <h3>templates/index.html не найден рядом с main.py</h3>
    <p>Разместите файл по пути ./templates/index.html</p>
    """
    return HTMLResponse(html_fallback, status_code=200)

@app.get("/health")
def health():
    return {"ok": True}

# JSON и TEXT оставлены для совместимости (front их не использует)
@app.post("/api/rss")
def api_rss(payload: RSSLinksRequest) -> Dict[str, Any]:
    urls = _normalize_urls(list(payload.urls or []))
    news, errors = _fetch_news_dict(urls, payload.latest_n, payload.keywords)
    return {"selected": urls, "news": news, "errors": errors}

@app.post("/api/rss_text", response_class=PlainTextResponse)
def api_rss_text(payload: RSSLinksRequest) -> PlainTextResponse:
    urls = _normalize_urls(list(payload.urls or []))
    news, errors = _fetch_news_dict(urls, payload.latest_n, payload.keywords)
    text_lines: List[str] = []
    if news:
        for url, items in news.items():
            text_lines.append(f"{url}:")
            for it in items:
                title = it.get("title") or "(без заголовка)"
                link = it.get("url") or ""
                pub  = it.get("published") or ""
                text_lines.append(f"- {title} [{pub}] {link}")
            text_lines.append("")
    if errors:
        text_lines.append("Ошибки:")
        for u, msg in errors.items():
            text_lines.append(f"- {u}: {msg}")
    text = "\n".join(text_lines) or "(пусто)"
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

# Нестримоовый HTML (на всякий случай)
@app.post("/api/rss_html", response_class=HTMLResponse)
def api_rss_html(payload: RSSLinksRequest) -> HTMLResponse:
    urls = _normalize_urls(list(payload.urls or []))
    if not urls:
        return HTMLResponse("<p class='muted'>(не переданы валидные ссылки)</p>", status_code=200)
    if parse_websites is None:
        return HTMLResponse("<p class='muted'>rss_parser.parse_websites не импортирован</p>", status_code=200)
    try:
        raw = parse_websites(urls, payload.keywords, latest_n=payload.latest_n) or {}
    except Exception as ex:
        return HTMLResponse(f"<p class='muted'>parse error: {html.escape(str(ex))}</p>", status_code=200)
    raw = {k: v for k, v in raw.items() if v}
    if not raw:
        return HTMLResponse("<p class='muted'>(нет записей)</p>", status_code=200)
    html_block = _format_html_cards(raw)
    return HTMLResponse(html_block, media_type="text/html; charset=utf-8")

# STREAM: события по мере готовности
@app.get("/api/rss_stream")
def api_rss_stream(urls: str, latest_n: int = 100, keywords: str = "") -> StreamingResponse:
    """
    Стримим карточки по мере готовности.
    GET-параметры:
      urls     — строка со ссылками (через запятую или с новой строки)
      latest_n — лимит на каждую ленту (1..300)
      keywords — строка ключевых через запятую (опционально)
    """
    try:
        latest_n = max(1, min(int(latest_n), 300))
    except Exception:
        latest_n = 100

    url_list = _normalize_urls(_coerce_urls_str(urls or ""))
    kw_list = [k.strip() for k in (keywords or "").split(",") if k.strip()]

    def event_stream():
        if not url_list:
            yield _sse_pack("done", {"ok": False, "error": "no valid urls"})
            return

        if parse_websites is None:
            yield _sse_pack("done", {"ok": False, "error": "rss_parser.parse_websites not imported"})
            return

        try:
            raw = parse_websites(url_list, kw_list, latest_n=latest_n) or {}
        except Exception as ex:
            yield _sse_pack("done", {"ok": False, "error": f"parse error: {ex}"})
            return

        total = sum(len(v or []) for v in raw.values())
        yielded = 0
        yield _sse_pack("start", {"total": total})

        # Обрабатываем по источникам, батчами (быстро и стабильно)
        batch_size = 8
        for src, items in raw.items():
            if not items:
                continue

            texts = _extract_texts_batch(items, max_workers=12)

            for i in range(0, len(items), batch_size):
                sub_items = items[i:i+batch_size]
                sub_texts = texts[i:i+batch_size]

                if ru_summarize_many is not None:
                    try:
                        sub_summaries = ru_summarize_many(
                            sub_texts, max_new_tokens=120, batch_size=batch_size, num_beams=1
                        )
                    except Exception:
                        sub_summaries = [_summarize_text(t) for t in sub_texts]
                else:
                    sub_summaries = [_summarize_text(t) for t in sub_texts]

                for entry, summary in zip(sub_items, sub_summaries):
                    payload = {
                        "src": src,
                        "title": _get_entry_title(entry),
                        "link": _get_entry_link(entry),
                        "published": _get_entry_published(entry),
                        "summary": summary or "(нет текста для суммаризации)",
                        "progress": {"done": yielded + 1, "total": total} if total else None
                    }
                    yielded += 1
                    yield _sse_pack("card", payload)

        yield _sse_pack("done", {"ok": True, "total": total})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
