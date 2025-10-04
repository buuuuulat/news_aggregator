import feedparser

feed = feedparser.parse("https://lenta.ru/rss/news")
print(feed)
