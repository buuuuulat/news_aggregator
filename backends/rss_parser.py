import feedparser#
from typing import Dict, List, Any

def _norm(s: str) -> str:
    return s.lower().replace("ё", "е")

def entry_matches_keywords(entry: dict, keywords: List[str]) -> bool:
    kw = [_norm(k) for k in keywords if k]
    if not kw:
        return True

    fields: List[str] = []
    fields.append(entry.get("title", "") or "")
    fields.append((entry.get("title_detail") or {}).get("value", "") or "")

    tags = entry.get("tags") or []
    for t in tags:
        if isinstance(t, dict):
            fields.append(t.get("term", "") or "")
        else:
            fields.append(str(t))

    haystack = _norm(" ".join(map(str, fields)))
    return any(k in haystack for k in kw)

def parse_websites(websites: List[str], keywords: List[str], latest_n: int = 100) -> Dict[str, List[dict]]:
    parsed = {}
    for website in websites:
        entries = feedparser.parse(website).get("entries", [])
        if latest_n != -1:
            entries = entries[:latest_n]
        # если keywords пуст — вернём всё
        matched = [e for e in entries if entry_matches_keywords(e, keywords)]
        parsed[website] = matched
    return parsed

"""
parsed1 = parse_websites(["http://lenta.ru/rss/news", "https://ria.ru/export/rss2/archive/index.xml"],
                     ["россия", "наука", "экономика"], 10)

print(parsed1["http://lenta.ru/rss/news"])
"""
