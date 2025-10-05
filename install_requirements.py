import subprocess
import sys

REQUIREMENTS = [
    "fastapi[standard]",
    "requests",
    "feedparser",
    "beautifulsoup4",
    "lxml",
    "trafilatura",
    "torch",
    "transformers",
    "sentencepiece",
    "safetensors",
    "accelerate",
]

def ensure_packages():
    for pkg in REQUIREMENTS:
        name = pkg.split("[")[0]
        try:
            __import__(name)
        except ImportError:
            print(f"⚙️ Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

if __name__ == "__main__":
    ensure_packages()
    print("✅ All required packages are installed or already up-to-date.")
