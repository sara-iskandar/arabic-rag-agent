import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from app.core.config import settings
import json
from pathlib import Path


class ArabicEmbedder:
    """
    Embeds Arabic text chunks using a multilingual model
    and stores them in ChromaDB. Also builds a BM25 index
    for hybrid retrieval.
    """

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.bm25_corpus = []

    def embed_documents(self, documents: list[dict]) -> None:
        """
        Embed all chunks and store in ChromaDB.
        Also builds BM25 index for hybrid retrieval.
        """
        if not documents:
            print("No documents to embed")
            return

        print(f"Embedding {len(documents)} chunks...")

        texts = [doc['text'] for doc in documents]
        ids = [f"chunk_{i}" for i in range(len(documents))]
        metadatas = [
            {
                'source': doc.get('source', ''),
                'url': doc.get('url', ''),
                'chunk_index': doc.get('chunk_index', i),
                'char_count': doc.get('char_count', 0)
            }
            for i, doc in enumerate(documents)
        ]

        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.extend(embeddings.tolist())
            print(f"  Embedded batch {i // batch_size + 1}")

        self.collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Stored {len(documents)} chunks in ChromaDB")

        self._build_bm25(texts)

        self._save_bm25_corpus(texts, metadatas)

    def _build_bm25(self, texts: list[str]) -> None:
        """Build BM25 index from tokenized Arabic texts"""
        tokenized = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_corpus = texts
        print("BM25 index built")

    def _save_bm25_corpus(
        self,
        texts: list[str],
        metadatas: list[dict]
    ) -> None:
        """Save BM25 corpus to disk for persistence"""
        corpus_path = Path("data/processed/bm25_corpus.json")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'texts': texts, 'metadatas': metadatas},
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"BM25 corpus saved to {corpus_path}")

    def load_bm25_corpus(self) -> bool:
        """Load BM25 corpus from disk if it exists"""
        corpus_path = Path("data/processed/bm25_corpus.json")
        if not corpus_path.exists():
            return False
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._build_bm25(data['texts'])
        print(f"Loaded BM25 corpus: {len(data['texts'])} texts")
        return True

    def get_collection_count(self) -> int:
        """Return number of chunks in ChromaDB"""
        return self.collection.count()