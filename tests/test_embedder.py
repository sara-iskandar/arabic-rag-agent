from app.core.loader import WebLoader
from app.core.embedder import ArabicEmbedder

loader = WebLoader()
embedder = ArabicEmbedder()

documents = loader.load_from_processed()
if not documents:
    documents = loader.load_urls()
    loader.save_to_processed(documents)

embedder.embed_documents(documents)

print(f"\nChromaDB collection count: {embedder.get_collection_count()}")
print("Embedder test complete!")