import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, validator
from urllib.parse import urlparse

# Используем твою функцию парсинга из rss_parser.py
try:
    from rss_parser import parse_websites
except Exception:
    parse_websites = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "templates/index.html"

app = FastAPI(title="News Aggregator (RSS → dict)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RSSLinksRequest(BaseModel):
    """
    Принимает массив ссылок или одну строку c ссылками
    (разделители — новая строка или запятая).
    """
    urls: Union[List[str], str]
    latest_n: int = 10
    keywords: List[str] = []

    @validator("urls", pre=True)
    def _coerce_urls(cls, v):
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace(",", "\n").splitlines()]
            return [p for p in parts if p]
        return v

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
            errors[u] = "parse error: {}".format(ex)

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
            lines.append("{}: [".format(url))
            for it in items:
                title = it.get("title") or "(без заголовка)"
                link = it.get("url") or ""
                pub  = it.get("published") or ""
                # одна запись — одной строкой
                lines.append('  {{ "title": "{}", "url": "{}", "published": "{}" }},'.format(
                    title.replace('"', '\\"'),
                    link.replace('"', '\\"'),
                    pub.replace('"', '\\"')
                ))
            lines.append("]\n")
    else:
        lines.append("(пусто)\n")

    if errors:
        lines.append("Ошибки:")
        for u, msg in errors.items():
            lines.append("- {}: {}".format(u, msg))

    return "\n".join(lines)

@app.get("/", response_class=FileResponse)
def serve_index():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    html = """
    <!doctype html><meta charset="utf-8">
    <title>News Aggregator</title>
    <h3>index.html не найден рядом с main.py</h3>
    <p>Сгенерируйте интерфейс или разместите файл.</p>
    """
    return HTMLResponse(html, status_code=200)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/rss")
def api_rss(payload: RSSLinksRequest) -> Dict[str, Any]:
    """
    JSON-вариант.
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
