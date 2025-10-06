from __future__ import annotations
import asyncio
import time
from typing import Iterable, List, Dict, Any, Callable, Optional
import feedparser
import re
import os
from pathlib import Path
import importlib.util
import inspect

# =========================
#  Надёжная загрузка summarizer/model.py + поиск callable
# =========================

_SUMM_MOD = None               # модуль model.py
_SUMM_CALLABLE: Optional[Callable[..., Any]] = None
_SUMM_ERR = ""
_SUMM_INIT_DONE = False

PREF_FUNC_NAMES = (
    "summarize",
    "summarize_text",
    "compress",
    "shorten",
    "abstract",
    "inference",
    "run",
    "__call__",
)

PREF_CLASS_NAMES = (
    "Summarizer",
    "TextSummarizer",
    "SummarizerRU",
    "AbstractiveSummarizer",
    "Compressor",
)

def _import_by_module() -> object | None:
    try:
        from news_aggregator.summarizer import model as m  # type: ignore
        return m
    except Exception:
        return None

def _load_module_from_path(path: Path) -> object | None:
    try:
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("summarizer_model_dynamic", str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    except Exception:
        return None
    return None

def _find_model_candidates() -> List[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[3] if len(here.parents) >= 4 else here.parents[-1]
    pkg_root = here.parents[1]  # .../news_aggregator
    cwd = Path.cwd()
    env_path = os.getenv("SUMMARIZER_MODEL_PATH", "")
    return [
        Path(env_path) if env_path else Path("/dev/null"),
        pkg_root / "summarizer" / "model.py",
        repo_root / "news_aggregator" / "summarizer" / "model.py",
        repo_root / "summarizer" / "model.py",
        cwd / "news_aggregator" / "summarizer" / "model.py",
        cwd / "summarizer" / "model.py",
    ]

def _load_module() -> None:
    global _SUMM_MOD, _SUMM_ERR
    if _SUMM_MOD is not None:
        return
    mod = _import_by_module()
    if mod is not None:
        _SUMM_MOD = mod
        return
    tried: List[str] = []
    for p in _find_model_candidates():
        try:
            if not p or str(p) == "/dev/null":
                continue
            mod = _load_module_from_path(p)
            tried.append(str(p))
            if mod is not None:
                _SUMM_MOD = mod
                return
        except Exception:
            tried.append(str(p))
            continue
    _SUMM_MOD = None
    _SUMM_ERR = (
        "Не удалось импортировать summarizer/model.py. "
        "Укажите путь в SUMMARIZER_MODEL_PATH=/полный/путь/к/model.py. "
        f"Проверены пути: {', '.join([t for t in tried if t]) or '—'}"
    )

def _maybe_init(mod: object):
    """Один раз вызвать init/load/setup, если есть (не критично)."""
    global _SUMM_INIT_DONE
    if _SUMM_INIT_DONE:
        return
    for name in ("init", "initialize", "load", "setup"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                sig = None
                try:
                    sig = inspect.signature(fn)
                except Exception:
                    pass
                if not sig or len(sig.parameters) == 0:
                    fn()
                else:
                    fn()  # допустим **kwargs
            except Exception:
                pass
    _SUMM_INIT_DONE = True

def _get_attr_chain(obj: object, path: str) -> Optional[Callable[..., Any]]:
    parts = [p for p in path.strip().split(".") if p]
    cur = obj
    for i, part in enumerate(parts):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
        if inspect.isclass(cur) and i < len(parts) - 1:
            try:
                cur = cur()  # type: ignore
            except Exception:
                return None
    return cur if callable(cur) else None

def _find_callable(mod: object) -> Optional[Callable[..., Any]]:
    env_call = os.getenv("SUMMARIZER_CALLABLE", "").strip()
    if env_call:
        c = _get_attr_chain(mod, env_call)
        if callable(c):
            return c
    for name in PREF_FUNC_NAMES:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    for cname in PREF_CLASS_NAMES:
        cls = getattr(mod, cname, None)
        if inspect.isclass(cls):
            try:
                inst = cls()  # type: ignore
                for mname in ("summarize", "__call__", "run"):
                    meth = getattr(inst, mname, None)
                    if callable(meth):
                        return meth
            except Exception:
                continue
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj):
            try:
                sig = inspect.signature(obj)
                if any(p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                       for p in sig.parameters.values()):
                    return obj
            except Exception:
                return obj
    return None

def _get_summarizer_callable() -> Callable[..., Any]:
    global _SUMM_CALLABLE
    _load_module()
    if _SUMM_MOD is None:
        raise RuntimeError(_SUMM_ERR or "Не найден summarizer/model.py.")
    _maybe_init(_SUMM_MOD)
    if _SUMM_CALLABLE is None:
        _SUMM_CALLABLE = _find_callable(_SUMM_MOD)
    if _SUMM_CALLABLE is None:
        raise RuntimeError("В summarizer/model.py не найден подходящий вызываемый объект (summarize).")
    return _SUMM_CALLABLE

# =========================
#  Парсинг и форматирование
# =========================

_HTML_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")

def _clean_html(text: str) -> str:
    text = _HTML_TAGS.sub(" ", text or "")
    text = _WS.sub(" ", text).strip()
    return text

def _first_nonempty(*vals) -> str:
    for v in vals:
        if v:
            return str(v)
    return ""

def _to_rfc2822(dt_like) -> str:
    """Mon, 06 Oct 2025 19:33:00 +0300"""
    try:
        if isinstance(dt_like, time.struct_time):
            return time.strftime("%a, %d %b %Y %H:%M:%S %z", dt_like)
        if hasattr(dt_like, "timetuple"):
            return time.strftime("%a, %d %b %Y %H:%M:%S %z", dt_like.timetuple())  # type: ignore
    except Exception:
        pass
    if isinstance(dt_like, str):
        return dt_like
    return ""

def _extract_body(entry: Dict[str, Any]) -> str:
    # стараемся вытащить максимально полный текст
    content_list = entry.get("content")
    if isinstance(content_list, list) and content_list:
        # ищем первый непустой value
        for c in content_list:
            v = (c or {}).get("value") or ""
            if v:
                return _clean_html(v)
    for key in ("summary", "summary_detail", "description"):
        v = entry.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v:
            return _clean_html(v)
    return _clean_html(entry.get("title") or "")

def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"

def _heuristic_summary_ru(text: str, max_words: int) -> str:
    """Эвристика: первые 2–3 предложения, затем ограничение по словам."""
    t = _clean_html(text or "")
    if not t:
        return ""
    sents = _SENT_SPLIT.split(t)
    # берём 2–3 предложения, чтобы выйти на 30–40 слов
    piece = " ".join(sents[:3])
    return _limit_words(piece, max_words)

def _call_with_compat(fn: Callable[..., Any], text: str, source_words: int, target_words: int) -> str | None:
    """Универсальный вызов summarize с авто-подбором параметров."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    def accepts(name: str) -> bool:
        if sig is None:
            return True
        if name in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def filt(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if accepts(k)}

    # 1) по словам
    for k in ("max_words", "words", "n_words"):
        kwargs = {k: target_words}
        if accepts("lang"): kwargs["lang"] = "ru"
        elif accepts("language"): kwargs["language"] = "ru"
        try:
            return fn(text, **filt(kwargs))
        except TypeError:
            pass

    # 2) по символам (≈7 на слово)
    approx_chars = int(target_words * 7.0)
    for k in ("max_chars", "max_length", "length"):
        kwargs = {k: approx_chars}
        if accepts("lang"): kwargs["lang"] = "ru"
        elif accepts("language"): kwargs["language"] = "ru"
        try:
            return fn(text, **filt(kwargs))
        except TypeError:
            pass

    # 3) по предложениям
    approx_sent = max(1, round(target_words / 20))
    for k in ("sentences", "n_sentences", "max_sentences"):
        kwargs = {k: approx_sent}
        if accepts("lang"): kwargs["lang"] = "ru"
        elif accepts("language"): kwargs["language"] = "ru"
        try:
            return fn(text, **filt(kwargs))
        except TypeError:
            pass

    # 4) ratio
    ratio = min(1.0, max(0.05, target_words / max(1, source_words)))
    if accepts("ratio"):
        kwargs = {"ratio": ratio}
        if accepts("lang"): kwargs["lang"] = "ru"
        elif accepts("language"): kwargs["language"] = "ru"
        try:
            return fn(text, **filt(kwargs))
        except TypeError:
            pass

    # 5) без параметров
    try:
        return fn(text)
    except Exception:
        return None

def _summarize_ru_words(text: str, max_words: int) -> str:
    """
    1) вызываем ваш summarizer (любой интерфейс),
    2) если он вернул пусто/None — делаем аккуратную эвристику, чтобы текст не пропадал.
    """
    src = _clean_html(text or "")
    if not src:
        return ""
    fn = _get_summarizer_callable()
    source_words = len(src.split())

    out = _call_with_compat(fn, src, source_words, max_words)
    if isinstance(out, str) and out.strip():
        return _limit_words(out.strip(), max_words)

    # Резерв: не оставляем пустым — формируем сводку эвристикой
    return _heuristic_summary_ru(src, max_words)

def _match_keywords(text: str, keywords_csv: str) -> bool:
    if not keywords_csv:
        return True
    text = (text or "").lower()
    keys = [k.strip().lower() for k in keywords_csv.split(",") if k.strip()]
    return any(k in text for k in keys)

def _to_display_time(entry: Dict[str, Any]) -> str:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    s = _to_rfc2822(st or entry.get("published") or entry.get("updated"))
    return s

def parse_single_feed(
    url: str,
    latest_n: int = 3,
    keywords: str = "",
    summary_words: int = 36,
) -> List[Dict[str, Any]]:
    d = feedparser.parse(url)
    out: List[Dict[str, Any]] = []
    for e in d.get("entries", [])[: max(1, latest_n)]:
        title = _first_nonempty(e.get("title"), "(без заголовка)")
        link = _first_nonempty(e.get("link"))
        body = _extract_body(e)
        blob = " ".join([title or "", body or ""]).lower()
        if keywords:
            keys = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            if keys and not any(k in blob for k in keys):
                continue
        summary_ru = _summarize_ru_words(body, max_words=summary_words)
        out.append(
            {
                "title": title,
                "time_display": _to_display_time(e),
                "summary_ru": summary_ru,
                "link": link,
            }
        )
    return out

def parse_many(
    urls: Iterable[str],
    latest_n: int = 3,
    keywords: str = "",
    summary_words: int = 36,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        items.extend(parse_single_feed(u, latest_n=latest_n, keywords=keywords, summary_words=summary_words))
    return items

async def parse_many_async(
    urls: Iterable[str],
    latest_n: int = 3,
    keywords: str = "",
    summary_words: int = 36,
) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: parse_many(urls, latest_n=latest_n, keywords=keywords, summary_words=summary_words)
    )

def format_three_lines_plain_ru(item: Dict[str, Any]) -> str:
    """
    Ровно 3 строки (без HTML):
    <Заголовок>
    <Время>
    <Сводка>
    """
    title = (item.get("title") or "").strip()
    t = (item.get("time_display") or "").strip()
    summary = (item.get("summary_ru") or "").strip()
    return "\n".join([title, t, summary]).strip()
