import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# pip install fastapi uvicorn jinja2

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
NEWS_FILE = BASE_DIR / "data" / "news.jsonl"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _parse_dt(s: str) -> datetime:
    """ISO 8601 -> datetime (безопасный парсинг)."""
    try:
        # fromisoformat понимает 'YYYY-MM-DDTHH:MM:SS[+TZ]'
        return datetime.fromisoformat(s)
    except Exception:
        # fallback: без таймзоны
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


class NewsStore:
    """Ленивая загрузка JSONL с кешированием по mtime."""
    def __init__(self, path: Path):
        self.path = path
        self._mtime: Optional[float] = None
        self._items: List[Dict[str, Any]] = []

    def _load_if_changed(self) -> None:
        if not self.path.exists():
            self._items = []
            self._mtime = None
            return
        mtime = self.path.stat().st_mtime
        if self._mtime is None or mtime != self._mtime:
            items: List[Dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # нормализуем поля
                        obj["source"] = str(obj.get("source", "")).strip()
                        obj["title"] = str(obj.get("title", "")).strip()
                        obj["url"] = str(obj.get("url", "")).strip()
                        dt_raw = str(obj.get("published_at", "")).strip()
                        obj["_dt"] = _parse_dt(dt_raw) if dt_raw else datetime.min
                        items.append(obj)
                    except Exception:
                        # можно залогировать плохие строки
                        continue
            # сортировка по дате убыв.
            items.sort(key=lambda x: x.get("_dt", datetime.min), reverse=True)
            self._items = items
            self._mtime = mtime

    def available_sources(self) -> List[str]:
        self._load_if_changed()
        return sorted({it.get("source", "") for it in self._items if it.get("source")})

    def query(self, sources: Optional[List[str]] = None, limit: int = 50) -> List[Dict[str, Any]]:
        self._load_if_changed()
        data = self._items
        if sources:
            src_set = set(sources)
            data = [it for it in data if it.get("source") in src_set]
        return data[:limit]


store = NewsStore(NEWS_FILE)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Пустая страница с формой
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sources": store.available_sources(),  # список доступных источников из файла
            "selected": [],
            "items": []
        }
    )


@app.post("/", response_class=HTMLResponse)
async def filter_news(request: Request, websites: List[str] = Form(default=[])):
    # Возвращаем тот же шаблон с выбранными источниками и отфильтрованными новостями
    items = store.query(sources=websites, limit=100)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sources": store.available_sources(),
            "selected": websites,
            "items": items
        }
    )
