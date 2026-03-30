from google import genai
from google.genai import types
from app.core.retriever import HybridRetriever
from app.core.embedder import ArabicEmbedder
from app.core.preprocessor import ArabicPreprocessor
from app.core.config import settings


class ArabicRAGPipeline:
    """
    Full RAG pipeline:
    1. Preprocess Arabic query
    2. Hybrid retrieval (dense + BM25)
    3. Build Arabic prompt with context
    4. Generate answer with Gemini
    5. Return cited, grounded response
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = "gemini-2.0-flash-lite"
        self.embedder = ArabicEmbedder()
        self.retriever = HybridRetriever(self.embedder)
        self.preprocessor = ArabicPreprocessor()
        self.embedder.load_bm25_corpus()

    def build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """Build Arabic RAG prompt with retrieved context"""
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            source = chunk['metadata'].get('source', 'مصدر غير معروف')
            context_text += f"[{i+1}] المصدر: {source}\n{chunk['text']}\n\n"

        prompt = f"""أنت مساعد طبي متخصص يجيب على الأسئلة بالعربية بناءً على المصادر المقدمة فقط.

المصادر:
{context_text}

التعليمات:
- أجب على السؤال بالعربية الفصحى
- استند فقط إلى المعلومات الواردة في المصادر أعلاه
- اذكر رقم المصدر [1] أو [2] عند الاستشهاد بمعلومة
- إذا لم تجد الإجابة في المصادر، قل ذلك صراحةً
- كن دقيقاً وموجزاً

السؤال: {query}

الإجابة:"""
        return prompt

    def query(self, user_query: str) -> dict:
        """
        Main entry point. Takes Arabic query,
        returns answer with sources.
        """
        clean_query = self.preprocessor.clean(user_query)
        print(f"Query: {clean_query}")

        chunks = self.retriever.hybrid_search(clean_query)
        print(f"Retrieved {len(chunks)} chunks")

        if not chunks:
            return {
                "answer": "لم أجد معلومات كافية للإجابة على هذا السؤال.",
                "sources": [],
                "chunks_used": 0
            }

        prompt = self.build_prompt(user_query, chunks)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            )
        )
        answer = response.text

        sources = list({
            chunk['metadata'].get('url', '')
            for chunk in chunks
            if chunk['metadata'].get('url')
        })

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks)
        }