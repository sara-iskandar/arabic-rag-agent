from app.core.loader import WebLoader

loader = WebLoader()

docs = loader.load_urls([
    "https://www.who.int/ar/news-room/fact-sheets/detail/diabetes"
])

print(f"\nTotal chunks: {len(docs)}")
if docs:
    print(f"\nFirst chunk source: {docs[0]['source']}")
    print(f"First chunk URL: {docs[0]['url']}")
    print(f"First chunk text preview:\n{docs[0]['text'][:200]}")