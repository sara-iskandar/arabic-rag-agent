from app.core.rag import ArabicRAGPipeline

pipeline = ArabicRAGPipeline()

queries = [
    "ما هي أعراض داء السكري؟",
    "كيف يمكن الوقاية من ارتفاع ضغط الدم؟",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"السؤال: {query}")
    print('='*60)
    result = pipeline.query(query)
    print(f"الإجابة:\n{result['answer']}")
    print(f"\nالمصادر: {result['sources']}")
    print(f"عدد المقاطع المستخدمة: {result['chunks_used']}")