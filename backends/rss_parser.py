import feedparser


def parse_websites(websites: list[str], keywords: list[str], latest_n: int=10):
    """
    :param websites: List of RSS links to the news websites
    :param keywords: Keywords filters (soon)
    :param latest_n: Choose last n articles from each source (-1 for all)
    :return: Dict: {rss_source_link: [dicts of RSS parsed params]}
    """
    parsed = {}
    for website in websites:
        parsed[website] = feedparser.parse(website)["entries"][:latest_n]
    return parsed

def parse_tg(tg_links: list[str], keywords: list[str], latest_n: int=10):
    pass

print(parse_websites(["http://lenta.ru/rss/news", "https://ria.ru/export/rss2/archive/index.xml"],
                     ["nigga"], 1))

