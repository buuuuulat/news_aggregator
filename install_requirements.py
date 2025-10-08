import subprocess
import sys

REQ = [
    # базовое веб-приложение
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "jinja2>=3.1",
    "itsdangerous>=2.1",
    "python-multipart>=0.0.9",
    "httpx>=0.27",
    # RSS и обработка
    "feedparser>=6.0.11",
    "beautifulsoup4>=4.12",
    # Telegram-бот
    "aiogram>=3.6.0",
    "python-dotenv>=1.0.1",
]

def pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    for pkg in REQ:
        print(f"Installing {pkg} ...")
        pip_install(pkg)
    print("All requirements installed.")
