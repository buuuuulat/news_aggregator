import feedparser
from typing import Dict, List, Any


def _norm(s: str) -> str:
    # небольшая нормализация для русского текста
    return s.lower().replace("ё", "е")

def entry_matches_keywords(entry: dict, keywords: List[str]) -> bool:
    """Проверяем, встречается ли хотя бы одно ключевое слово
    в title, title_detail.value или в любом из tags[*].term.
    """
    kw = [_norm(k) for k in keywords if k]  # нормализуем ключевые слова

    # собираем поля для поиска
    fields: List[str] = []
    fields.append(entry.get("title", "") or "")
    fields.append((entry.get("title_detail") or {}).get("value", "") or "")

    # tags — это список словарей вида {"term": "..."}
    tags = entry.get("tags") or []
    for t in tags:
        if isinstance(t, dict):
            fields.append(t.get("term", "") or "")
        else:
            # на всякий случай, если придёт строка
            fields.append(str(t))

    haystack = _norm(" ".join(map(str, fields)))
    return any(k in haystack for k in kw)

def filter_parsed_by_keywords(parsed: Dict[str, List[dict]], keywords: List[str]) -> Dict[str, List[dict]]:
    """На вход твоя структура {rss_url: [entries...]}, на выход — только совпавшие записи по каждому источнику."""
    out: Dict[str, List[dict]] = {}
    for src, items in parsed.items():
        matched = [e for e in items if entry_matches_keywords(e, keywords)]
        if matched:
            out[src] = matched
    return out

def parse_websites(websites: List[str], keywords: List[str], latest_n: int = 100) -> Dict[str, List[dict]]:
    parsed = {}
    for website in websites:
        entries = feedparser.parse(website).get("entries", [])
        if latest_n != -1:
            entries = entries[:latest_n]
        matched = [e for e in entries if entry_matches_keywords(e, keywords)]
        parsed[website] = matched
    return parsed

parsed1 = parse_websites(["http://lenta.ru/rss/news", "https://ria.ru/export/rss2/archive/index.xml"],
                     ["россия", "наука", "экономика"], 10)

print(len(parsed1["http://lenta.ru/rss/news"]))

