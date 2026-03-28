"""
RAG pipeline orchestrator.

Ties together retrieval (FAISS vector search) and generation (LLM) to answer
queries about ingested handwritten documents and to produce structured output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
from vector_store import VectorStoreManager, EmbeddingWrapper

logger = logging.getLogger(__name__)

# ── Prompt templates ───────────────────────────────────────────────

STRUCTURING_PROMPT = """\
You are a document digitisation assistant. You have been given text that was \
extracted via OCR from a handwritten document. Your job is to produce a clean, \
well-structured digital version of the document.

INSTRUCTIONS:
1. Correct obvious OCR errors (misspellings, garbled characters).
2. Preserve the original meaning — do NOT add or remove information.
3. Organise the content with appropriate headings, paragraphs, and lists.
4. If the document contains a letter, preserve the greeting and sign-off.
5. If the document contains a form or table, reproduce it in Markdown table format.
6. Output ONLY the structured document — no commentary.

--- EXTRACTED TEXT ---
{extracted_text}
--- END ---

Structured digital document:
"""

QA_PROMPT = """\
Use the following context extracted from handwritten documents to answer the \
question. If the answer cannot be found in the context, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""


# ── LLM loader ─────────────────────────────────────────────────────

def _get_llm():
    """Return a LangChain-compatible LLM based on config."""
    if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )

    # Fallback: local HuggingFace pipeline
    from transformers import pipeline as hf_pipeline
    from langchain_community.llms import HuggingFacePipeline

    logger.info("Loading local LLM: %s", config.LOCAL_LLM_MODEL)
    pipe = hf_pipeline(
        "text2text-generation",
        model=config.LOCAL_LLM_MODEL,
        max_new_tokens=1024,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ── Pipeline class ─────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline for handwritten document processing.

    Supports two modes:
      • **structuring** — convert raw OCR text into a clean digital document
      • **qa** — answer questions about ingested documents
    """

    def __init__(self, vector_manager: VectorStoreManager | None = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _get_llm()
        return self._llm

    # ── Structuring mode ───────────────────────────────────────────

    def structure_document(self, extracted_text: str) -> str:
        """
        Use the LLM to transform raw OCR text into a clean, structured
        digital document.
        """
        prompt = STRUCTURING_PROMPT.format(extracted_text=extracted_text)
        response = self.llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    # ── QA mode ────────────────────────────────────────────────────

    def query(self, question: str, k: int | None = None) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns dict with keys: answer, sources, chunks.
        """
        results = self.vector_manager.search(question, k=k)

        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(doc.page_content)
            sources.append({
                "text": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "score": float(score),
            })

        context = "\n\n---\n\n".join(context_parts)
        prompt = QA_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "question": question,
            "sources": sources,
            "num_chunks_retrieved": len(results),
        }

    # ── Batch structuring ──────────────────────────────────────────

    def structure_from_retrieval(self, topic: str = "") -> str:
        """
        Retrieve all relevant chunks for a topic (or everything if empty)
        and produce a structured document.
        """
        query = topic or "full document content"
        results = self.vector_manager.search(query, k=20)
        combined = "\n".join(doc.page_content for doc, _ in results)
        return self.structure_document(combined)
