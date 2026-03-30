import requests
from bs4 import BeautifulSoup
from pathlib import Path
from app.core.preprocessor import ArabicPreprocessor
from app.core.config import settings

WHO_URLS = [
    "https://www.who.int/ar/news-room/fact-sheets/detail/diabetes",
    "https://www.who.int/ar/news-room/fact-sheets/detail/hypertension",
    "https://www.who.int/ar/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.who.int/ar/news-room/fact-sheets/detail/obesity-and-overweight",
    "https://www.who.int/ar/news-room/fact-sheets/detail/cancer",
]


class WebLoader:
    """
    Scrapes Arabic content from URLs, cleans it,
    and returns chunked documents ready for embedding.
    """

    def __init__(self):
        self.preprocessor = ArabicPreprocessor()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ar,en;q=0.9",
        }

    def scrape_url(self, url: str) -> dict:
        """
        Fetch a URL and extract main Arabic text content.
        Returns dict with url, title, and raw text.
        """
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'
        except requests.RequestException as e:
            print(f"  Failed: {e}")
            return {}

        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'footer',
                         'header', 'aside', 'form', 'button']):
            tag.decompose()

        title = ''
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.get_text(strip=True)

        content = ''
        for selector in ['main', 'article', '.sf-detail-body-wrapper',
                         '.content', '#content', 'body']:
            container = soup.find(selector) if '.' not in selector and '#' not in selector \
                else soup.select_one(selector)
            if container:
                content = container.get_text(separator=' ', strip=True)
                break

        return {
            'url': url,
            'title': title,
            'raw_text': content
        }

    def load_urls(self, urls: list[str] = None) -> list[dict]:
        """
        Scrape a list of URLs and return chunked documents.
        Defaults to WHO_URLS if no list provided.
        """
        if urls is None:
            urls = WHO_URLS

        documents = []

        for url in urls:
            page = self.scrape_url(url)

            if not page or not page.get('raw_text'):
                print(f"  Skipping — no content extracted")
                continue

            raw_text = page['raw_text']
            print(f"  Extracted {len(raw_text)} characters — '{page['title']}'")

            chunks = self.preprocessor.chunk_text(
                raw_text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap
            )

            for chunk in chunks:
                chunk['source'] = page['title'] or url
                chunk['url'] = url
                documents.append(chunk)

            print(f"  → {len(chunks)} chunks created")

        print(f"\nTotal: {len(documents)} chunks from {len(urls)} URLs")
        return documents

    def save_to_processed(self, documents: list[dict]) -> None:
        """
        Optionally cache processed chunks to data/processed/
        so you don't re-scrape on every run.
        """
        import json
        output_path = Path("data/processed/chunks.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(documents)} chunks to {output_path}")

    def load_from_processed(self) -> list[dict] | None:
        """
        Load cached chunks if they exist — avoids re-scraping.
        Returns None if no cache found.
        """
        import json
        cache_path = Path("data/processed/chunks.json")

        if not cache_path.exists():
            return None

        with open(cache_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        print(f"Loaded {len(documents)} chunks from cache")
        return documents