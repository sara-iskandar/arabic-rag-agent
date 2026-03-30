import re
import pyarabic.araby as araby


class ArabicPreprocessor:
    """
    Handles all Arabic-specific text normalization before
    embedding or retrieval. Each method is independent so
    you can use them selectively.
    """

    def normalize_alef(self, text: str) -> str:
        """Normalize all alef variants to bare alef: أ إ آ ٱ → ا"""
        return araby.normalize_alef(text)

    def normalize_teh_marbuta(self, text: str) -> str:
        """Normalize teh marbuta: ة → ه"""
        return araby.normalize_teh(text)

    def remove_diacritics(self, text: str) -> str:
        """Remove tashkeel: تَعَلَّمَ → تعلم"""
        return araby.strip_tashkeel(text)

    def remove_tatweel(self, text: str) -> str:
        """Remove tatweel (kashida): مـــدرسة → مدرسة"""
        return araby.strip_tatweel(text)

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation but keep Arabic letters, numbers, spaces"""
        return re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces and strip"""
        return re.sub(r'\s+', ' ', text).strip()

    def clean(self, text: str) -> str:
        """
        Full normalization pipeline — runs all steps in order.
        This is what you call before embedding or indexing.
        """
        text = self.remove_diacritics(text)
        text = self.remove_tatweel(text)
        text = self.normalize_alef(text)
        text = self.normalize_teh_marbuta(text)
        text = self.normalize_whitespace(text)
        return text

    def split_sentences(self, text: str) -> list[str]:
        """
        Split Arabic text into sentences.
        Arabic uses ؟ ! . ، as delimiters.
        """
        pattern = r'(?<=[.!?؟])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list[dict]:
        """
        Split text into overlapping chunks that respect
        sentence boundaries. Returns list of dicts with
        text and metadata.
        """
        cleaned = self.clean(text)
        sentences = self.split_sentences(cleaned)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'char_count': len(chunk_text)
                })
                chunk_index += 1

                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_index': chunk_index,
                'char_count': len(chunk_text)
            })

        return chunks