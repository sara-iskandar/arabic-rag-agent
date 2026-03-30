# 🤖 Arabic RAG Agent — Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for Arabic language, built with FastAPI, ChromaDB, and Google Gemini. Answers Arabic medical questions grounded in WHO Arabic sources with full citation support.

## ✨ What makes this unique

- **Arabic-aware preprocessing** — diacritic removal, alef/hamza normalization, tatweel stripping via PyArabic
- **Hybrid retrieval** — combines dense vector search (multilingual embeddings) with BM25 keyword search using Reciprocal Rank Fusion
- **Citation grounding** — every answer references its WHO source, preventing hallucination
- **Production-ready** — FastAPI REST API with Swagger UI, Dockerized, persistent ChromaDB

## 🏗️ Architecture
```
Arabic Query
     │
     ▼
Arabic Preprocessor (normalization, diacritics, tatweel)
     │
     ▼
Hybrid Retriever
├── Dense Search (paraphrase-multilingual-mpnet-base-v2 → ChromaDB)
└── BM25 Search (rank-bm25)
     │
     ▼
Reciprocal Rank Fusion (RRF merge)
     │
     ▼
Arabic Prompt Builder (RTL-aware, citation template)
     │
     ▼
Gemini 2.5 Flash
     │
     ▼
Cited Arabic Answer + Sources
```

## 🚀 Quick Start

### With Docker (recommended)
```bash
git clone https://github.com/yourusername/arabic-rag-agent.git
cd arabic-rag-agent
cp .env.example .env  # Add your GEMINI_API_KEY
docker-compose up --build
```

### Without Docker
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Check API status and indexed chunk count |
| POST | `/api/v1/ingest` | Scrape Arabic URLs and index into ChromaDB |
| POST | `/api/v1/query` | Ask an Arabic question, get a cited answer |

### Example query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أعراض داء السكري؟"}'
```
```json
{
  "answer": "تشمل أعراض داء السكري: كثرة التبول، العطش الشديد، [1] ...",
  "sources": ["https://www.who.int/ar/news-room/fact-sheets/detail/diabetes"],
  "chunks_used": 5
}
```

## 🗂️ Knowledge Base

Default corpus scraped from WHO Arabic fact sheets:
- داء السكري (Diabetes)
- ارتفاع ضغط الدم (Hypertension)
- أمراض القلب والأوعية الدموية (Cardiovascular diseases)
- السمنة وزيادة الوزن (Obesity)
- السرطان (Cancer)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Embeddings | paraphrase-multilingual-mpnet-base-v2 |
| Vector Store | ChromaDB (persistent) |
| Keyword Search | BM25 (rank-bm25) |
| Arabic NLP | PyArabic |
| LLM | Google Gemini 2.5 Flash |
| Deployment | Docker + docker-compose |

## 📁 Project Structure
```
arabic-rag-agent/
├── app/
│   ├── api/
│   │   └── routes.py        # FastAPI endpoints
│   ├── core/
│   │   ├── config.py        # Settings
│   │   ├── preprocessor.py  # Arabic NLP pipeline
│   │   ├── loader.py        # WHO Arabic web scraper
│   │   ├── embedder.py      # Embeddings + ChromaDB
│   │   ├── retriever.py     # Hybrid BM25 + dense search
│   │   └── rag.py           # Full RAG pipeline
│   ├── models/
│   │   └── schemas.py       # Pydantic schemas
│   └── main.py              # FastAPI app
├── tests/                   # Unit tests
├── data/
│   └── processed/           # Cached chunks + BM25 index
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:
```env
GEMINI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
COLLECTION_NAME=arabic_rag
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Author

**Sara** — Software Engineer transitioning to AI/ML

[LinkedIn](https://linkedin.com/in/sara-iskandar) • [GitHub](https://github.com/sara-iskandar)

---

⭐ If you found this useful, give it a star!
