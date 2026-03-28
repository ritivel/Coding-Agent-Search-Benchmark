"""Stage 4a: Semantic Chunking & Embedding — split text semantically and generate embeddings."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    embedding: Optional[np.ndarray] = None
    chunk_index: int = 0
    source_document: str = ""
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Splits text using semantic similarity between sentences rather than fixed windows."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 512,
        overlap: int = 64,
        similarity_threshold: float = 0.75,
    ):
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info(f"Loading embedding model: {self.model_name}")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded.")

    def chunk_and_embed(self, text: str, source_document: str = "") -> List[TextChunk]:
        self._load_model()

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        try:
            chunks = self._semantic_chunk(sentences)
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to fixed: {e}")
            chunks = self._fixed_chunk(text)

        result = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            embedding = self._model.encode(chunk_text, normalize_embeddings=True)
            result.append(
                TextChunk(
                    text=chunk_text,
                    embedding=embedding,
                    chunk_index=i,
                    source_document=source_document,
                    metadata={"chunk_method": "semantic", "word_count": len(chunk_text.split())},
                )
            )

        logger.info(f"Created {len(result)} chunks from {source_document or 'input'}")
        return result

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(query, normalize_embeddings=True)

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _semantic_chunk(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= 1:
            return [" ".join(sentences)]

        embeddings = self._model.encode(sentences, normalize_embeddings=True)

        chunks = []
        current_chunk = [sentences[0]]
        current_words = len(sentences[0].split())

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            sent_words = len(sentences[i].split())

            if sim >= self.similarity_threshold and current_words + sent_words <= self.max_chunk_size:
                current_chunk.append(sentences[i])
                current_words += sent_words
            else:
                chunks.append(" ".join(current_chunk))
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1] if len(current_chunk[-1].split()) <= self.overlap else ""
                    current_chunk = [overlap_text, sentences[i]] if overlap_text else [sentences[i]]
                    current_words = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentences[i]]
                    current_words = sent_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _fixed_chunk(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.max_chunk_size - self.overlap
        return chunks
