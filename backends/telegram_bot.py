# /news_aggregator/backends/telegram_bot.py
from __future__ import annotations
import os
import textwrap
from typing import Dict, Any, List

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage

from backends.rss_utils import (
    parse_many_async,
    format_three_lines_plain_ru,
)

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"

# Простая in-memory "память" (без БД)
USER_PREFS: Dict[int, Dict[str, Any]] = {}

class RSSForm(StatesGroup):
    waiting_for_urls = State()

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
    # По умолчанию: 3 статьи и сводка около 36 слов
    return USER_PREFS.setdefault(uid, {"latest_n": 3, "keywords": "", "summary_words": 36})

async def on_start(message: Message):
    await message.answer(
        textwrap.dedent(
            """\
            Привет! Я телеграм-интерфейс агрегатора. Присылаю СЖАТЫЕ сводки по новостям (на русском).
            Формат каждого сообщения — 3 строки: заголовок / время / сводка.

            Команды:
            • /setn <число> — сколько статей брать с источника (по умолчанию 3)
            • /setk <слова через запятую> — фильтр по ключевым словам
            • /setsumw <число> — целевой размер сводки в словах (по умолчанию 36)
            • /rss — затем пришлите список ссылок (по одной на строку или через запятую)
            • /cancel — отменить текущий ввод
            """
        )
    )

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
    await message.answer("❎ Отменено. Можете начать заново (/rss).")

async def cmd_rss(message: Message, state: FSMContext):
    await state.set_state(RSSForm.waiting_for_urls)
    await message.answer(
        "Отправьте список RSS-ссылок (каждую на новой строке или через запятую). "
        "Команда /cancel — отменить."
    )

async def handle_urls(message: Message, state: FSMContext):
    urls = _parse_urls_block(message.text or "")
    if not urls:
        await message.answer("Ссылок не найдено. Пришлите корректный список или /cancel.")
        return

    user_id = message.from_user.id
    prefs = _get_user_prefs(user_id)
    latest_n = int(prefs.get("latest_n", 3))
    keywords = str(prefs.get("keywords", ""))
    summary_words = int(prefs.get("summary_words", 36))

    await message.answer(
        f"🔎 Обрабатываю {len(urls)} источников… (по {latest_n} записей, сводка ≈ {summary_words} слов, фильтр: {keywords or '—'})"
    )

    try:
        items = await parse_many_async(
            urls, latest_n=latest_n, keywords=keywords, summary_words=summary_words
        )
    except RuntimeError as e:
        # Обязательно использовать summarizer — честно сообщаем, если его нет/сломался
        await state.clear()
        await message.answer(f"⚠️ {e}")
        return
    finally:
        await state.clear()

    if not items:
        await message.answer("Ничего не найдено по заданным условиям.")
        return

    # Каждую новость — ОТДЕЛЬНЫМ сообщением, три строки, без HTML
    for it in items:
        text = format_three_lines_plain_ru(it)
        if not text:
            continue
        await message.answer(text, disable_web_page_preview=True)

def build_bot_and_dispatcher() -> (Bot, Dispatcher):
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
    dp.message.register(cmd_setn, Command("setn"))
    dp.message.register(cmd_setk, Command("setk"))
    dp.message.register(cmd_setsumw, Command("setsumw"))
    dp.message.register(cmd_cancel, Command("cancel"))
    dp.message.register(cmd_rss, Command("rss"))

    dp.message.register(
        handle_urls,
        StateFilter(RSSForm.waiting_for_urls),
        F.content_type == "text",
    )
    return bot, dp

async def run_bot_forever():
    bot, dp = build_bot_and_dispatcher()
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
