from __future__ import annotations
import asyncio
import os
from pathlib import Path

import uvicorn

try:
    from dotenv import load_dotenv
    candidates = [
        Path(__file__).resolve().parents[2] / ".env",  # repo root (…/ .env)
        Path(__file__).resolve().parents[1] / ".env",  # package root (…/news_aggregator/.env)
        Path.cwd() / ".env",                           # CWD
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break
    else:
        load_dotenv()
except Exception:
    pass

from backends.main import app

from backends.telegram_bot import run_bot_forever

HOST = os.getenv("NA_HOST", "0.0.0.0")
PORT = int(os.getenv("NA_PORT", "8000"))

async def run_uvicorn():
    config = uvicorn.Config(app=app, host=HOST, port=PORT, log_level=os.getenv("NA_LOG_LEVEL", "info"))
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    web_task = asyncio.create_task(run_uvicorn(), name="uvicorn")
    bot_task = asyncio.create_task(run_bot_forever(), name="telegram-bot")

    done, pending = await asyncio.wait({web_task, bot_task}, return_when=asyncio.FIRST_EXCEPTION)

    for t in pending:
        t.cancel()
    for t in done:
        exc = t.exception()
        if exc:
            raise exc

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
