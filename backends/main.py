# -*- coding: utf-8 -*-
"""
FastAPI backend for the RSS news aggregator.
Добавлена минимальная авторизация пользователей через cookie-сессию.
Важно: логика приложения (парсинг RSS, SSE-стрим и суммаризация) не менялась.
Мы лишь:
  • ввели /auth/register, /auth/login, /auth/me, /auth/logout
  • добавили middleware, которое требует авторизацию для /api/rss_stream
  • НЕ трогали summatizer/model.py и backends/rss_parser.py

Авторизация "минимальная":
  - Пользователи хранятся в памяти процесса (без БД), формат {username: {salt, pwd_hash}}
  - Пароли хэшируются (sha256 + индивидуальная соль).
  - Сессии — простые случайные токены в памяти процесса (cookie 'session').
  - Для продакшена потребуется постоянное хранилище и защищённые cookie (Secure/HTTPS).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # html cleanup для summary, если есть bs4
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - необязательная зависимость
    BeautifulSoup = None  # type: ignore[assignment]

# Попытка подключить summarize-модель из ../summarizer/model.py
SUMM_DIR = Path(__file__).resolve().parents[1] / "summarizer"
if str(SUMM_DIR) not in sys.path:
    sys.path.append(str(SUMM_DIR))
try:  # pragma: no cover - зависимости могут отсутствовать в окружении
    from model import summarize as ru_summarize, summarize_many as ru_summarize_many  # type: ignore
except Exception:  # noqa: PIE786 - fallback
    ru_summarize = None  # type: ignore[assignment]
    ru_summarize_many = None  # type: ignore[assignment]

# ВАЖНО: мы не меняем ваш парсер.
# Он должен экспортировать функцию-обработчик SSE под тем же URL (/api/rss_stream),
# либо предоставлять функцию/генератор, который мы вызываем ниже.
# Ниже мы просто импортируем и проксируем запрос в исходную реализацию.
try:
    # Вариант А: у вас есть готовый ASGI-роутер/эндпоинт внутри файла.
    # Тогда этот импорт обеспечить не нужно — мы оставляем ваш маршрут как есть.
    from . import rss_parser  # type: ignore
except Exception:  # pragma: no cover
    rss_parser = None  # на случай, если среда импорта временно недоступна

app = FastAPI(title="News Aggregator (with minimal auth)")

# Если у вас был CORS — сохраняем «как было» или включаем «минимально безопасно».
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при необходимости сузьте домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Память процесса: пользователи и сессии
# --------------------------
_USERS: dict[str, dict[str, str]] = {}
_SESSIONS: dict[str, dict[str, str | int]] = {}  # token -> {"username":..., "created_at":...}

SESSION_COOKIE = "session"
SESSION_TTL_SECONDS: Optional[int] = None  # можно выставить, например, 7*24*3600

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + ":" + password).encode("utf-8")).hexdigest()

def _create_user(username: str, password: str) -> None:
    if username in _USERS:
        raise ValueError("Пользователь уже существует")
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    _USERS[username] = {"salt": salt, "pwd_hash": pwd_hash}

def _verify_user(username: str, password: str) -> bool:
    u = _USERS.get(username)
    if not u:
        return False
    return _hash_password(password, u["salt"]) == u["pwd_hash"]

def _new_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = {"username": username, "created_at": int(time.time())}
    return token

def _get_username_by_session(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    sess = _SESSIONS.get(token)
    if not sess:
        return None
    if SESSION_TTL_SECONDS:
        if int(time.time()) - int(sess.get("created_at", 0)) > SESSION_TTL_SECONDS:
            # срок истёк
            _SESSIONS.pop(token, None)
            return None
    return str(sess.get("username"))

def _drop_session(token: Optional[str]) -> None:
    if token:
        _SESSIONS.pop(token, None)


def _split_urls(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace(";", "\n").replace(",", "\n").splitlines()]
    return [p for p in parts if p]


def _ensure_scheme(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        return "https://" + url
    return url


def _normalize_urls(urls: List[str]) -> List[str]:
    normalized: List[str] = []
    for raw in urls:
        candidate = _ensure_scheme(raw)
        parsed = urlparse(candidate)
        if parsed.netloc:
            normalized.append(candidate)
    # deduplicate c сохранением порядка
    seen: Dict[str, None] = {}
    for url in normalized:
        if url not in seen:
            seen[url] = None
    return list(seen.keys())


def _normalize_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_entry_attr(entry: Any, key: str, default: str = "") -> str:
    if isinstance(entry, dict):
        return str(entry.get(key, default) or default)
    return str(getattr(entry, key, default) or default)


def _get_entry_link(entry: Any) -> str:
    link = _get_entry_attr(entry, "link") or _get_entry_attr(entry, "url")
    return link.strip()


def _get_entry_title(entry: Any) -> str:
    title = _get_entry_attr(entry, "title")
    return title or "(без заголовка)"


def _get_entry_published(entry: Any) -> str:
    return _get_entry_attr(entry, "published") or _get_entry_attr(entry, "updated")


def _get_entry_inline_content(entry: Any) -> str:
    if isinstance(entry, dict):
        if entry.get("summary"):
            return str(entry["summary"])
        if entry.get("content"):
            blocks = entry["content"]
            if isinstance(blocks, list):
                try:
                    return " \n".join(str(block.get("value", "")) for block in blocks if isinstance(block, dict))
                except Exception:  # pragma: no cover - предохранитель
                    pass
    return ""


def _clean_summary(value: str | None) -> str:
    if not value:
        return ""
    if BeautifulSoup:
        return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return value


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _error_stream(message: str):
    yield _sse_event("start", {"total": 0})
    yield _sse_event("done", {"ok": False, "error": message})


def _extract_article_text(url: str, fallback: str = "") -> str:
    if not url:
        return fallback

    # 1) Попытка через trafilatura
    try:
        import trafilatura  # type: ignore

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_images=False,
            )
            if extracted:
                return extracted
    except Exception:  # pragma: no cover - опциональная зависимость
        pass

    # 2) requests + BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=10, headers=headers)
        if resp.ok and resp.text:
            if BeautifulSoup:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.extract()
                container = soup.find("article") or soup
                paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
                text = "\n".join(p for p in paragraphs if p)
                if text:
                    return text
    except Exception:  # pragma: no cover
        pass

    return fallback


def _summarize_text(text: str) -> str:
    if not text:
        return ""
    try:
        if ru_summarize is not None:
            parts = ru_summarize(text, max_new_tokens=120)
            if isinstance(parts, list):
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
    except Exception:  # pragma: no cover
        pass

    # Fallback: первые 2–3 предложения
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3]).strip()


def _extract_texts_batch(entries: List[Any], max_workers: int = 12) -> List[str]:
    texts = ["" for _ in range(len(entries))]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, entry in enumerate(entries):
            link = _get_entry_link(entry)
            inline = _clean_summary(_get_entry_inline_content(entry))
            futures[executor.submit(_extract_article_text, link, inline)] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                texts[idx] = fut.result() or ""
            except Exception:  # pragma: no cover
                texts[idx] = ""
    return texts

# --------------------------
# Схемы запросов/ответов
# --------------------------
class AuthPayload(BaseModel):
    username: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=1, max_length=500)

class WhoAmI(BaseModel):
    username: str

# --------------------------
# Middleware: защита /api/rss_stream
# --------------------------
@app.middleware("http")
async def auth_gate_for_rss_stream(request: Request, call_next):
    # Не меняем логику ваших маршрутов; просто требуем авторизацию для конкретного эндпоинта.
    if request.url.path.startswith("/api/rss_stream"):
        token = request.cookies.get(SESSION_COOKIE)
        user = _get_username_by_session(token)
        if not user:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        # можно пробросить имя пользователя вниз по цепочке, если нужно:
        request.state.username = user  # type: ignore[attr-defined]
    response = await call_next(request)
    return response

# --------------------------
# Auth endpoints
# --------------------------
@app.post("/auth/register", response_model=WhoAmI)
def auth_register(payload: AuthPayload, response: Response):
    username = payload.username.strip()
    if not username:
        raise HTTPException(400, "Укажите имя пользователя")
    try:
        _create_user(username, payload.password)
    except ValueError as e:
        raise HTTPException(409, str(e))
    token = _new_session(username)
    # Простейшая cookie-сессия
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # для HTTPS выставьте True
        max_age=SESSION_TTL_SECONDS,
    )
    return {"username": username}

@app.post("/auth/login", response_model=WhoAmI)
def auth_login(payload: AuthPayload, response: Response):
    username = payload.username.strip()
    if not _verify_user(username, payload.password):
        raise HTTPException(401, "Неверные логин или пароль")
    token = _new_session(username)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=SESSION_TTL_SECONDS,
    )
    return {"username": username}

@app.get("/auth/me", response_model=WhoAmI)
def auth_me(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    username = _get_username_by_session(token)
    if not username:
        raise HTTPException(401, "Не авторизован")
    return {"username": username}

@app.post("/auth/logout")
def auth_logout(response: Response, request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    _drop_session(token)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}

# --------------------------
# Проксирование существующего SSE-эндпоинта (если он у вас объявлен здесь же)
# --------------------------
# Если ваш /api/rss_stream объявлен в backends/rss_parser.py, ничего дополнительно делать не нужно.
# Middleware выше уже будет его защищать.
# Если же он был объявлен прямо в этом файле — оставьте ваш исходный код ниже без изменений.
# --------------------------


@app.get("/api/rss_stream")
def api_rss_stream(
    request: Request,
    urls: str = Query("", description="Список RSS-адресов через запятую или перевод строки"),
    latest_n: int = Query(100, ge=1, le=300, description="Количество статей на источник"),
    keywords: str = Query("", description="Ключевые слова через запятую"),
):
    if rss_parser is None or not hasattr(rss_parser, "parse_websites"):
        return StreamingResponse(_error_stream("RSS парсер не сконфигурирован"), media_type="text/event-stream")

    try:
        latest_n = max(1, min(int(latest_n), 300))
    except Exception:
        latest_n = 100

    url_list = _normalize_urls(_split_urls(urls))
    keyword_list = _normalize_keywords(keywords or "")

    def event_stream():
        if not url_list:
            yield _sse_event("start", {"total": 0})
            yield _sse_event("done", {"ok": False, "error": "no valid urls"})
            return

        try:
            parsed = rss_parser.parse_websites(url_list, keyword_list, latest_n=latest_n) or {}
        except Exception as exc:  # pragma: no cover - сеть/IO
            yield _sse_event("start", {"total": 0})
            yield _sse_event("done", {"ok": False, "error": f"parse error: {exc}"})
            return

        total = sum(len(v or []) for v in parsed.values()) if isinstance(parsed, dict) else 0
        yield _sse_event("start", {"total": total})

        if total == 0:
            yield _sse_event("done", {"ok": True, "message": "Нет записей по выбранным источникам"})
            return

        batch_size = 8
        issued = 0
        for src in url_list:
            entries = parsed.get(src, []) if isinstance(parsed, dict) else []
            if not entries:
                continue

            texts = _extract_texts_batch(entries)

            for i in range(0, len(entries), batch_size):
                chunk_entries = entries[i:i + batch_size]
                chunk_texts = texts[i:i + batch_size]

                if ru_summarize_many is not None:
                    try:
                        chunk_summaries = ru_summarize_many(
                            chunk_texts,
                            max_new_tokens=120,
                            batch_size=len(chunk_texts),
                            num_beams=1,
                        )
                    except Exception:  # pragma: no cover - падение модели
                        chunk_summaries = [_summarize_text(text) for text in chunk_texts]
                else:
                    chunk_summaries = [_summarize_text(text) for text in chunk_texts]

                for entry, summary in zip(chunk_entries, chunk_summaries):
                    issued += 1
                    clean_summary = _clean_summary(summary) or _clean_summary(_get_entry_inline_content(entry))
                    payload = {
                        "src": src,
                        "title": _get_entry_title(entry),
                        "link": _get_entry_link(entry),
                        "published": _get_entry_published(entry),
                        "summary": clean_summary or "(нет текста для суммаризации)",
                        "progress": {"done": issued, "total": total},
                    }
                    yield _sse_event("card", payload)

        yield _sse_event("done", {"ok": True, "total": total})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

# Подсказка: если у вас не было корневого роута, можно отдавать шаблон:
# (Если у вас уже есть свой обработчик '/', просто удалите этот маршрут.)
try:
    from fastapi.templating import Jinja2Templates  # используйте ваш шаблонизатор, если он есть
    from pathlib import Path

    TPL_DIR = Path(__file__).resolve().parents[0] / "templates"
    templates = Jinja2Templates(directory=str(TPL_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        # Рендер вашего index.html из /backends/templates/
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/auth", response_class=HTMLResponse)
    def auth_page(request: Request):
        return templates.TemplateResponse("auth.html", {"request": request})
except Exception:
    # Если у вас другой способ отдачи шаблонов — оставьте как было.
    pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backends.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
