from app.core.preprocessor import ArabicPreprocessor

preprocessor = ArabicPreprocessor()

test_text = """
تَعَلَّمَ الطِّفْلُ اللُّغَةَ العَرَبِيَّةَ بِسُهُولَةٍ.
إنَّ الصِّحَّةَ العَامَّةَ مـــن أهم الأولويات.
أَمْرَاضُ القَلْبِ وَالسُّكَّرِي مِنَ الأَمْرَاضِ المَزْمِنَةِ.
يَجِبُ عَلَى المَرِيضِ أَنْ يَلْتَزِمَ بِالجُرْعَةِ المُحَدَّدَةِ.
"""

print("=== Original Text ===")
print(test_text)

print("=== After clean() ===")
cleaned = preprocessor.clean(test_text)
print(cleaned)

print("\n=== Sentences ===")
sentences = preprocessor.split_sentences(cleaned)
for i, s in enumerate(sentences):
    print(f"{i+1}: {s}")

print("\n=== Chunks ===")
chunks = preprocessor.chunk_text(test_text, chunk_size=100, overlap=20)
for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']} ({chunk['char_count']} chars): {chunk['text'][:80]}...")