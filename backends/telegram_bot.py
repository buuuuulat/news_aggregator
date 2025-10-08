from __future__ import annotations
import os
import textwrap
import html
import re
from typing import Dict, Any, List, Tuple, Optional
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

# опционально: не падаем, если bs4 не установлен
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

# подключим папку summarizer как в веб-бэке
SUMM_DIR = Path(__file__).resolve().parents[1] / "summarizer"
if str(SUMM_DIR) not in sys.path:
    sys.path.append(str(SUMM_DIR))
try:
    from model import summarize as ru_summarize, summarize_many as ru_summarize_many  # type: ignore
except Exception:
    ru_summarize = None  # type: ignore[assignment]
    ru_summarize_many = None  # type: ignore[assignment]

from backends.rss_utils import (
    parse_many_async,           # берём структуру items (title/link/published/summary/...)
    format_three_lines_plain_ru # fallback
)

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
USER_PREFS: Dict[int, Dict[str, Any]] = {}

class RSSForm(StatesGroup):
    waiting_for_urls = State()

class SettingsForm(StatesGroup):
    waiting_for_custom_n = State()
    waiting_for_keywords = State()
    waiting_for_custom_sumw = State()

# ---------------- общие утилиты ----------------

def _parse_urls_block(s: str) -> List[str]:
    raw = (s or "").replace("\r", "\n").split("\n")
    urls: List[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        urls.extend([u.strip() for u in line.split(",") if u.strip()])
    return urls

def _get_user_prefs(uid: int) -> Dict[str, Any]:
    return USER_PREFS.setdefault(
        uid,
        {
            "latest_n": 3,
            "keywords": "",
            "summary_words": 36,
            "preview": True,
            "last_urls": [],
        },
    )

def _fmt_source(url: str) -> str:
    try:
        from urllib.parse import urlparse
        host = urlparse(url or "").netloc or ""
        return host.replace("www.", "") if host else ""
    except Exception:
        return ""

def _truncate_words(s: str, max_words: int) -> str:
    parts = s.split()
    if len(parts) <= max_words:
        return s
    return " ".join(parts[:max_words]).rstrip(",.;:—- ") + "…"

def _fmt_date_human(s: str) -> str:
    if not s:
        return ""
    try:
        dt = parsedate_to_datetime(s)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s

def _clean_html(text: str) -> str:
    if not text:
        return ""
    if BeautifulSoup:
        try:
            return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        except Exception:
            pass
    # легкий фолбэк
    return re.sub(r"<[^>]+>", " ", text)

# ----- извлечение полного текста статьи (как на сайте) -----

def _extract_article_text(url: str, fallback: str = "") -> str:
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
    except Exception:
        pass

    return fallback

def _summarize_many_texts(texts: List[str], target_words: int) -> List[str]:
    # «приближаем» слова к токенам: ~2 * words (жесткий верх)
    max_new_tokens = max(30, min(200, target_words * 2))

    if ru_summarize_many is not None:
        try:
            outs = ru_summarize_many(
                texts,
                max_new_tokens=max_new_tokens,
                batch_size=len(texts) if texts else 1,
                num_beams=1,
            )
            if isinstance(outs, list) and outs:
                return [(_clean_html(o) if o else "") for o in outs]
        except Exception:
            pass

    results: List[str] = []
    if ru_summarize is not None:
        for t in texts:
            try:
                parts = ru_summarize(t, max_new_tokens=max_new_tokens)
                if isinstance(parts, list):
                    joined = "\n".join(p for p in parts if p).strip()
                else:
                    joined = str(parts or "").strip()
                results.append(_clean_html(joined))
            except Exception:
                results.append("")
    else:
        # глухой фолбэк: первые 2–3 предложения
        for t in texts:
            s = re.split(r"(?<=[.!?])\s+", (t or "").strip())
            results.append(" ".join(s[:3]).strip())

    return results

def _ensure_neural_summaries(items: List[Dict[str, Any]], target_words: int) -> List[Dict[str, Any]]:
    """
    Для каждого айтема:
      — если summary пустой/мусорный, тянем полный текст по ссылке и прогоняем через summarize,
      — сохраняем итог в item["summary"].
    Делаем батчами для скорости.
    """
    if not items:
        return items

    # какие нужно суммаризовать
    need_idx: List[int] = []
    fallbacks: List[str] = []
    links: List[str] = []

    for i, it in enumerate(items):
        cur = (it.get("summary") or "").strip()
        if cur and len(cur.split()) >= max(10, target_words // 3):
            # уже что-то вменяемое
            continue
        need_idx.append(i)
        inline = (
            it.get("summary") or it.get("description") or it.get("content")
            or it.get("text") or it.get("inline") or ""
        )
        fallbacks.append(_clean_html(str(inline)))
        links.append(str(it.get("link") or ""))

    if not need_idx:
        return items

    # вытаскиваем тексты параллельно
    texts: List[str] = [""] * len(need_idx)
    with ThreadPoolExecutor(max_workers=min(12, len(need_idx) or 1)) as ex:
        fut2idx = {}
        for j, idx in enumerate(need_idx):
            fut = ex.submit(_extract_article_text, links[j], fallbacks[j])
            fut2idx[fut] = j
        for fut in as_completed(fut2idx):
            j = fut2idx[fut]
            try:
                texts[j] = fut.result() or fallbacks[j] or ""
            except Exception:
                texts[j] = fallbacks[j] or ""

    # суммаризация батчем
    summaries = _summarize_many_texts(texts, target_words)

    # записываем в исходные items + подрезаем по словам
    for j, idx in enumerate(need_idx):
        s = _truncate_words((summaries[j] or "").replace("\n", " "), max_words=target_words).strip()
        items[idx]["summary"] = s if s else (fallbacks[j] or "—")

    return items

# --------- формат вывода (как ты попросил) ---------

def _format_card_html(item: Dict[str, Any], summary_words: int = 36) -> str:
    # Порядок: Заголовок / Сводка / Дата, <a href="...">Источник</a>
    title = str(item.get("title") or "(без заголовка)")
    link = str(item.get("link") or "")
    published = str(item.get("published") or "")
    src = str(item.get("src") or "") or _fmt_source(link)

    title_h = html.escape(title)
    summary_raw = (item.get("summary") or "").strip()
    summary_clean = html.escape(_truncate_words(summary_raw.replace("\n", " "), summary_words)) if summary_raw else "—"

    date_str = _fmt_date_human(published)
    src_label = html.escape(src or "Источник")
    link_h = html.escape(link)

    footer = ""
    if date_str and link_h:
        footer = f"{html.escape(date_str)}, <a href=\"{link_h}\">{src_label}</a>"
    elif date_str:
        footer = html.escape(date_str)
    elif link_h:
        footer = f"<a href=\"{link_h}\">{src_label}</a>"

    lines = [title_h, summary_clean]
    if footer:
        lines.append(footer)
    return "\n".join(lines).strip()

# ---------- Меню ----------

def _kb_main(uid: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="▶️ Собрать сейчас", callback_data="run_last")
    b.button(text="📰 RSS источники", callback_data="rss_prompt")
    b.button(text="⚙️ Настройки", callback_data="settings")
    b.button(text="❓ Справка", callback_data="help")
    b.adjust(1, 2, 1)
    return b.as_markup()

def _kb_settings(uid: int) -> InlineKeyboardMarkup:
    prefs = _get_user_prefs(uid)
    n = int(prefs.get("latest_n", 3))
    w = int(prefs.get("summary_words", 36))
    kw = str(prefs.get("keywords", ""))
    preview = bool(prefs.get("preview", True))

    b = InlineKeyboardBuilder()
    b.button(text=f"Записей: {n} ➖", callback_data="setn:dec")
    b.button(text="➕", callback_data="setn:inc")
    b.button(text="✏️ ввести", callback_data="setn:set")

    b.button(text=f"Сводка, слов: {w} ➖", callback_data="setsumw:dec")
    b.button(text="➕", callback_data="setsumw:inc")
    b.button(text="✏️ ввести", callback_data="setsumw:set")

    kw_label = kw if kw else "—"
    b.button(text=f"Ключевые: {kw_label}", callback_data="noop")
    b.button(text="✏️ изменить", callback_data="setk:edit")
    if kw:
        b.button(text="🧹 очистить", callback_data="setk:clear")

    b.button(text=f"Превью ссылок: {'ON' if preview else 'OFF'}", callback_data="preview:toggle")

    b.button(text="⬅️ Назад в меню", callback_data="menu")
    b.adjust(3, 3, 2, 1, 1)
    return b.as_markup()

async def _send_main_menu(message: Message):
    await message.answer(
        "Главное меню:\n— Запускай сбор из последних источников, меняй настройки или добавь новые RSS.",
        reply_markup=_kb_main(message.from_user.id),
    )

async def _edit_to_main_menu(cb: CallbackQuery):
    await cb.message.edit_text(
        "Главное меню:\n— Запускай сбор из последних источников, меняй настройки или добавь новые RSS.",
        reply_markup=_kb_main(cb.from_user.id),
    )

async def _edit_to_settings(cb: CallbackQuery):
    await cb.message.edit_text(
        "⚙️ Настройки:\n— Тыкни ➖/➕ или «✏️ ввести» для точного значения.\n— Превью: включает/выключает карточки-превью ссылок.",
        reply_markup=_kb_settings(cb.from_user.id),
    )

# ---------- Команды ----------

async def on_start(message: Message):
    await _setup_bot_commands(message.bot)
    await message.answer(
        textwrap.dedent(
            """\
            Привет! Я телеграм-интерфейс агрегатора.
            Формат: Заголовок / Сводка / Дата, Источник(ссылка).
            """
        ),
        parse_mode="HTML",
    )
    await _send_main_menu(message)

async def cmd_menu(message: Message):
    await _send_main_menu(message)

async def cmd_settings(message: Message):
    await message.answer("⚙️ Настройки:", reply_markup=_kb_settings(message.from_user.id))

async def cmd_help(message: Message):
    await message.answer(
        textwrap.dedent(
            """\
            Справка:
            • /menu — открыть главное меню
            • /rss — ввести список RSS-ссылок
            • /settings — открыть настройки
            • /setn N — установить число записей (1..300)
            • /setsumw W — целевой размер сводки (15..80 слов)
            • /setk слова — ключевые слова через запятую
            • /cancel — отменить ввод
            """
        )
    )

async def _setup_bot_commands(bot: Bot):
    await bot.set_my_commands(
        [
            BotCommand(command="menu", description="Главное меню"),
            BotCommand(command="rss", description="Ввести RSS источники"),
            BotCommand(command="settings", description="Настройки"),
            BotCommand(command="help", description="Справка"),
            BotCommand(command="cancel", description="Отменить ввод"),
        ]
    )

# ---------- Совместимость со «старыми» командами ----------

async def cmd_setn(message: Message):
    try:
        n = int(message.text.split(maxsplit=1)[1])
        if n < 1 or n > 300:
            raise ValueError
    except Exception:
        await message.answer("Укажите целое 1..300, например: /setn 3")
        return
    _get_user_prefs(message.from_user.id)["latest_n"] = n
    await message.answer(f"✅ Буду брать по {n} записей с каждого RSS.")

async def cmd_setk(message: Message):
    try:
        kw = message.text.split(maxsplit=1)[1].strip()
    except Exception:
        kw = ""
    _get_user_prefs(message.from_user.id)["keywords"] = kw
    await message.answer(f"✅ Фильтр по ключевым словам: {kw or 'отключён'}.")

async def cmd_setsumw(message: Message):
    try:
        w = int(message.text.split(maxsplit=1)[1])
        if w < 15 or w > 80:
            raise ValueError
    except Exception:
        await message.answer("Укажите 15..80 слов, например: /setsumw 36")
        return
    _get_user_prefs(message.from_user.id)["summary_words"] = w
    await message.answer(f"✅ Размер сводки установлен: {w} слов.")

async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("❎ Отменено. Можете начать заново (/rss или /menu).")

# ---------- RSS ввод/запуск ----------

async def cmd_rss(message: Message, state: FSMContext):
    await state.set_state(RSSForm.waiting_for_urls)
    await message.answer(
        "Отправьте список RSS-ссылок (каждую на новой строке или через запятую). Команда /cancel — отменить."
    )

async def handle_urls(message: Message, state: FSMContext):
    urls = _parse_urls_block(message.text or "")
    if not urls:
        await message.answer("Ссылок не найдено. Пришлите корректный список или /cancel.")
        return

    uid = message.from_user.id
    prefs = _get_user_prefs(uid)
    prefs["last_urls"] = urls

    latest_n = int(prefs.get("latest_n", 3))
    keywords = str(prefs.get("keywords", ""))
    summary_words = int(prefs.get("summary_words", 36))
    preview = bool(prefs.get("preview", True))

    await message.answer(
        f"🔎 Обрабатываю {len(urls)} источников… (по {latest_n} записей, сводка ≈ {summary_words} слов, фильтр: {keywords or '—'})"
    )

    try:
        items = await parse_many_async(
            urls, latest_n=latest_n, keywords=keywords, summary_words=summary_words
        )
    except RuntimeError as e:
        await state.clear()
        await message.answer(f"⚠️ {e}")
        return
    finally:
        await state.clear()

    # критично: гарантируем прогон через модель по ПОЛНОМУ тексту
    items = _ensure_neural_summaries(items, target_words=summary_words)

    if not items:
        await message.answer("Ничего не найдено по заданным условиям.")
        return

    for it in items:
        text_html = _format_card_html(it, summary_words=summary_words)
        if not text_html:
            fallback = format_three_lines_plain_ru(it)
            if fallback:
                await message.answer(fallback, disable_web_page_preview=(not preview))
            continue
        await message.answer(
            text_html,
            parse_mode="HTML",
            disable_web_page_preview=(not preview),
        )

# ---------- Callback-и меню ----------

async def cb_menu(cb: CallbackQuery):
    await cb.answer()
    await _edit_to_main_menu(cb)

async def cb_help(cb: CallbackQuery):
    await cb.answer()
    await cb.message.edit_text(
        textwrap.dedent(
            """\
            ❓ Справка:
            • «Собрать сейчас» — запускает сбор по последним источникам.
            • «RSS источники» — ввести новый список ссылок.
            • «Настройки» — число записей, длина сводки, ключевые слова, превью.
            Также доступны /menu, /rss, /settings, /help.
            """
        ),
        reply_markup=_kb_main(cb.from_user.id),
    )

async def cb_rss_prompt(cb: CallbackQuery, state: FSMContext):
    await cb.answer()
    await cb.message.edit_text("Отправьте список RSS-ссылок (каждую на новой строке или через запятую).")
    await state.set_state(RSSForm.waiting_for_urls)

async def cb_run_last(cb: CallbackQuery):
    await cb.answer()
    uid = cb.from_user.id
    prefs = _get_user_prefs(uid)
    last_urls = prefs.get("last_urls") or []
    if not last_urls:
        await cb.message.edit_text(
            "Сначала добавьте источники через «📰 RSS источники».",
            reply_markup=_kb_main(uid),
        )
        return

    latest_n = int(prefs.get("latest_n", 3))
    keywords = str(prefs.get("keywords", ""))
    summary_words = int(prefs.get("summary_words", 36))
    preview = bool(prefs.get("preview", True))

    await cb.message.edit_text(
        f"🔎 Обрабатываю {len(last_urls)} источников… (по {latest_n} записей, сводка ≈ {summary_words} слов, фильтр: {keywords or '—'})"
    )

    try:
        items = await parse_many_async(
            last_urls, latest_n=latest_n, keywords=keywords, summary_words=summary_words
        )
    except RuntimeError as e:
        await cb.message.answer(f"⚠️ {e}")
        await _edit_to_main_menu(cb)
        return

    items = _ensure_neural_summaries(items, target_words=summary_words)

    if not items:
        await cb.message.answer("Ничего не найдено по заданным условиям.")
        await _edit_to_main_menu(cb)
        return

    for it in items:
        text_html = _format_card_html(it, summary_words=summary_words)
        if not text_html:
            fallback = format_three_lines_plain_ru(it)
            if fallback:
                await cb.message.answer(fallback, disable_web_page_preview=(not preview))
            continue
        await cb.message.answer(
            text_html,
            parse_mode="HTML",
            disable_web_page_preview=(not preview),
        )
    await _edit_to_main_menu(cb)

async def cb_settings(cb: CallbackQuery):
    await cb.answer()
    await _edit_to_settings(cb)

async def cb_noop(cb: CallbackQuery):
    await cb.answer()

async def cb_toggle_preview(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["preview"] = not bool(prefs.get("preview", True))
    await _edit_to_settings(cb)

# setn
async def cb_setn_inc(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["latest_n"] = min(300, int(prefs.get("latest_n", 3)) + 1)
    await _edit_to_settings(cb)

async def cb_setn_dec(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["latest_n"] = max(1, int(prefs.get("latest_n", 3)) - 1)
    await _edit_to_settings(cb)

async def cb_setn_set(cb: CallbackQuery, state: FSMContext):
    await cb.answer()
    await cb.message.edit_text("Введи целое число записей (1..300):")
    await state.set_state(SettingsForm.waiting_for_custom_n)

# setsumw
async def cb_setsumw_inc(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["summary_words"] = min(80, int(prefs.get("summary_words", 36)) + 1)
    await _edit_to_settings(cb)

async def cb_setsumw_dec(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["summary_words"] = max(15, int(prefs.get("summary_words", 36)) - 1)
    await _edit_to_settings(cb)

async def cb_setsumw_set(cb: CallbackQuery, state: FSMContext):
    await cb.answer()
    await cb.message.edit_text("Введи целевую длину сводки в словах (15..80):")
    await state.set_state(SettingsForm.waiting_for_custom_sumw)

# setk
async def cb_setk_edit(cb: CallbackQuery, state: FSMContext):
    await cb.answer()
    await cb.message.edit_text("Введи ключевые слова через запятую (пусто — отключить):")
    await state.set_state(SettingsForm.waiting_for_keywords)

async def cb_setk_clear(cb: CallbackQuery):
    await cb.answer()
    prefs = _get_user_prefs(cb.from_user.id)
    prefs["keywords"] = ""
    await _edit_to_settings(cb)

# ---------- Обработчики ввода для состояний настроек ----------

async def handle_custom_n(message: Message, state: FSMContext):
    try:
        n = int(message.text.strip())
        if not (1 <= n <= 300):
            raise ValueError
    except Exception:
        await message.answer("Некорректно. Укажи целое 1..300.")
        return
    _get_user_prefs(message.from_user.id)["latest_n"] = n
    await state.clear()
    await message.answer("✅ Обновлено.", reply_markup=_kb_settings(message.from_user.id))

async def handle_custom_sumw(message: Message, state: FSMContext):
    try:
        w = int(message.text.strip())
        if not (15 <= w <= 80):
            raise ValueError
    except Exception:
        await message.answer("Некорректно. Введи 15..80.")
        return
    _get_user_prefs(message.from_user.id)["summary_words"] = w
    await state.clear()
    await message.answer("✅ Обновлено.", reply_markup=_kb_settings(message.from_user.id))

async def handle_keywords(message: Message, state: FSMContext):
    _get_user_prefs(message.from_user.id)["keywords"] = message.text.strip()
    await state.clear()
    await message.answer("✅ Ключевые слова сохранены.", reply_markup=_kb_settings(message.from_user.id))

# ---------- Сборка бота ----------

def build_bot_and_dispatcher() -> Tuple[Bot, Dispatcher]:
    token = os.getenv(BOT_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"Env var {BOT_TOKEN_ENV} is empty. "
            f"Set TELEGRAM_BOT_TOKEN='<your_bot_token>' в .env перед запуском."
        )

    storage = MemoryStorage()
    bot = Bot(token=token)
    dp = Dispatcher(storage=storage)

    dp.message.register(on_start, CommandStart())
    dp.message.register(cmd_menu, Command("menu"))
    dp.message.register(cmd_settings, Command("settings"))
    dp.message.register(cmd_help, Command("help"))

    dp.message.register(cmd_setn, Command("setn"))
    dp.message.register(cmd_setk, Command("setk"))
    dp.message.register(cmd_setsumw, Command("setsumw"))
    dp.message.register(cmd_cancel, Command("cancel"))

    dp.message.register(cmd_rss, Command("rss"))
    dp.message.register(handle_urls, StateFilter(RSSForm.waiting_for_urls), F.content_type == "text")

    dp.message.register(handle_custom_n, StateFilter(SettingsForm.waiting_for_custom_n), F.content_type == "text")
    dp.message.register(
        handle_custom_sumw, StateFilter(SettingsForm.waiting_for_custom_sumw), F.content_type == "text"
    )
    dp.message.register(handle_keywords, StateFilter(SettingsForm.waiting_for_keywords), F.content_type == "text")

    dp.callback_query.register(cb_menu, F.data == "menu")
    dp.callback_query.register(cb_help, F.data == "help")
    dp.callback_query.register(cb_rss_prompt, F.data == "rss_prompt")
    dp.callback_query.register(cb_run_last, F.data == "run_last")
    dp.callback_query.register(cb_settings, F.data == "settings")
    dp.callback_query.register(cb_noop, F.data == "noop")
    dp.callback_query.register(cb_toggle_preview, F.data == "preview:toggle")

    dp.callback_query.register(cb_setn_inc, F.data == "setn:inc")
    dp.callback_query.register(cb_setn_dec, F.data == "setn:dec")
    dp.callback_query.register(cb_setn_set, F.data == "setn:set")

    dp.callback_query.register(cb_setsumw_inc, F.data == "setsumw:inc")
    dp.callback_query.register(cb_setsumw_dec, F.data == "setsumw:dec")
    dp.callback_query.register(cb_setsumw_set, F.data == "setsumw:set")

    dp.callback_query.register(cb_setk_edit, F.data == "setk:edit")
    dp.callback_query.register(cb_setk_clear, F.data == "setk:clear")

    return bot, dp

async def run_bot_forever():
    bot, dp = build_bot_and_dispatcher()
    await _setup_bot_commands(bot)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
