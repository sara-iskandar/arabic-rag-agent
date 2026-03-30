from rank_bm25 import BM25Okapi
from app.core.embedder import ArabicEmbedder
from app.core.config import settings
import json
from pathlib import Path


class HybridRetriever:
    """
    Combines dense vector search (ChromaDB) with sparse
    keyword search (BM25) for better Arabic retrieval.
    Dense handles semantic similarity, BM25 handles exact
    Arabic root/word matches that embeddings sometimes miss.
    """

    def __init__(self, embedder: ArabicEmbedder):
        self.embedder = embedder
        self.top_k = settings.top_k

    def dense_search(self, query: str, top_k: int = None) -> list[dict]:
        """Vector similarity search in ChromaDB"""
        k = top_k or self.top_k
        query_embedding = self.embedder.model.encode([query]).tolist()

        results = self.embedder.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        hits = []
        for i in range(len(results['documents'][0])):
            hits.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],
                'method': 'dense'
            })
        return hits

    def bm25_search(self, query: str, top_k: int = None) -> list[dict]:
        """BM25 keyword search"""
        k = top_k or self.top_k

        if self.embedder.bm25 is None:
            loaded = self.embedder.load_bm25_corpus()
            if not loaded:
                print("No BM25 index found, skipping BM25 search")
                return []

        tokenized_query = query.split()
        scores = self.embedder.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        corpus_path = Path("data/processed/bm25_corpus.json")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                hits.append({
                    'text': corpus['texts'][idx],
                    'metadata': corpus['metadatas'][idx],
                    'score': float(scores[idx]),
                    'method': 'bm25'
                })
        return hits

    def hybrid_search(self, query: str) -> list[dict]:
        """
        Merge dense and BM25 results using Reciprocal Rank Fusion.
        RRF rewards chunks that appear in both result lists —
        the best signal that a chunk is truly relevant.
        """
        dense_hits = self.dense_search(query, top_k=self.top_k)
        bm25_hits = self.bm25_search(query, top_k=self.top_k)

        k = 60  # RRF constant
        scores = {}
        texts = {}
        metadatas = {}

        for rank, hit in enumerate(dense_hits):
            key = hit['text'][:100]  # use text prefix as key
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            texts[key] = hit['text']
            metadatas[key] = hit['metadata']

        for rank, hit in enumerate(bm25_hits):
            key = hit['text'][:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            texts[key] = hit['text']
            metadatas[key] = hit['metadata']

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        results = []
        for key in sorted_keys[:self.top_k]:
            results.append({
                'text': texts[key],
                'metadata': metadatas[key],
                'rrf_score': round(scores[key], 4)
            })

        return results