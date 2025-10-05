import os
import sys
import html
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, validator
from urllib.parse import urlparse

# ---------------------------
# Paths & imports
# ---------------------------

APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "templates/index.html"

# Import RSS parser
try:
    from rss_parser import parse_websites
except Exception:
    parse_websites = None  # type: ignore

# Import summarizer from ../summarizer/model.py
SUMM_DIR = (APP_DIR / ".." / "summarizer").resolve()
if str(SUMM_DIR) not in sys.path:
    sys.path.append(str(SUMM_DIR))
try:
    from model import summarize as ru_summarize  # type: ignore
except Exception:
    ru_summarize = None  # gracefully degrade


# ---------------------------
# FastAPI setup
# ---------------------------

app = FastAPI(title="News Aggregator (RSS → dict & HTML summaries)")

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
# Helpers: Legacy JSON/text formatting
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

def _format_text_dict(news: Dict[str, List[Dict[str, Any]]], errors: Dict[str, str]) -> str:
    """
    Делает человекочитаемый «текстовый словарь».
    """
    lines: List[str] = []
    if news:
        for url, items in news.items():
            lines.append(f"{url}: [")
            for it in items:
                title = it.get("title") or "(без заголовка)"
                link = it.get("url") or ""
                pub  = it.get("published") or ""
                # одна запись — одной строкой
                lines.append('  {{ "title": "{}", "url": "{}", "published": "{}" }},'.format(
                    str(title).replace('"', '\\"'),
                    str(link).replace('"', '\\"'),
                    str(pub).replace('"', '\\"')
                ))
            lines.append("]\n")
    else:
        lines.append("(пусто)\n")

    if errors:
        lines.append("Ошибки:")
        for u, msg in errors.items():
            lines.append(f"- {u}: {msg}")

    return "\n".join(lines)


# ---------------------------
# HTML cards renderer
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
        lines.append('<div class="news-group">')
        lines.append(f'<div class="news-source">{html.escape(src)}</div>')
        for e in items or []:
            title = _get_entry_title(e)
            link = _get_entry_link(e)
            pub  = _get_entry_published(e)
            inline = _get_entry_inline_content(e)
            fulltext = _extract_article_text(link, fallback=inline)
            summary  = _summarize_text(fulltext) or "(нет текста для суммаризации)"

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
            lines.append(f'<div class="sum">{html.escape(summary)}</div>')
            lines.append('</div>')  # card
        lines.append('</div>')      # news-group
    return "\n".join(lines)


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
    <h3>index.html не найден рядом с main.py</h3>
    <p>Сгенерируйте интерфейс или разместите файл.</p>
    """
    return HTMLResponse(html_fallback, status_code=200)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/rss")
def api_rss(payload: RSSLinksRequest) -> Dict[str, Any]:
    """
    JSON-вариант (для совместимости).
    """
    urls = _normalize_urls(list(payload.urls or []))
    news, errors = _fetch_news_dict(urls, payload.latest_n, payload.keywords)
    return {"selected": urls, "news": news, "errors": errors}

@app.post("/api/rss_text", response_class=PlainTextResponse)
def api_rss_text(payload: RSSLinksRequest) -> PlainTextResponse:
    """
    ТЕКСТОВЫЙ вариант: text/plain со «словарём» новостей.
    """
    urls = _normalize_urls(list(payload.urls or []))
    news, errors = _fetch_news_dict(urls, payload.latest_n, payload.keywords)
    text = _format_text_dict(news, errors)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

@app.post("/api/rss_html", response_class=HTMLResponse)
def api_rss_html(payload: RSSLinksRequest) -> HTMLResponse:
    """
    Красивый HTML: заголовок (ссылка), дата, и сжатие через модель.
    """
    urls = _normalize_urls(list(payload.urls or []))

    if not urls:
        return HTMLResponse("<p class='muted'>(не переданы валидные ссылки)</p>", status_code=200)

    if parse_websites is None:
        return HTMLResponse("<p class='muted'>rss_parser.parse_websites не импортирован</p>", status_code=200)

    try:
        raw = parse_websites(urls, payload.keywords, latest_n=payload.latest_n) or {}
    except Exception as ex:
        return HTMLResponse(f"<p class='muted'>parse error: {html.escape(str(ex))}</p>", status_code=200)

    # Пустые источники не рендерим
    raw = {k: v for k, v in raw.items() if v}
    if not raw:
        return HTMLResponse("<p class='muted'>(нет записей)</p>", status_code=200)

    html_block = _format_html_cards(raw)
    return HTMLResponse(html_block, media_type="text/html; charset=utf-8")


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
