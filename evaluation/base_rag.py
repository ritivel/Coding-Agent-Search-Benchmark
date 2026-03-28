"""
Abstract base class for RAG models.

Each of the five RAG methodologies will subclass BaseRAGModel and implement
the `generate` and `retrieve` methods.  The evaluation framework calls
`run` to produce responses that can then be scored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseRAGModel(ABC):
    """
    Contract that every RAG methodology must satisfy.

    Subclasses implement:
      - retrieve(question, corpus) -> list of document strings
      - generate(question, documents) -> response string
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def retrieve(self, question: str, corpus: Optional[list[str]] = None) -> list[str]:
        """
        Retrieve context documents relevant to the question.

        Parameters
        ----------
        question : the user query.
        corpus   : optional external corpus. If None, use the model's own
                   built-in retriever / index.

        Returns
        -------
        List of document strings.
        """
        ...

    @abstractmethod
    def generate(self, question: str, documents: list[str]) -> str:
        """
        Generate a response given the question and retrieved documents.

        Parameters
        ----------
        question  : the user query.
        documents : context documents from the retriever.

        Returns
        -------
        Generated response string.
        """
        ...

    def run(self, question: str, corpus: Optional[list[str]] = None) -> dict:
        """
        Full RAG pipeline: retrieve then generate.

        Returns a dict with keys: question, documents, response.
        """
        documents = self.retrieve(question, corpus)
        response = self.generate(question, documents)
        return {
            "question": question,
            "documents": documents,
            "response": response,
        }

    def run_batch(
        self,
        examples: list[dict],
        use_existing_documents: bool = True,
    ) -> list[str]:
        """
        Run the model over a batch of examples.

        Parameters
        ----------
        examples : normalised RAGBench examples.
        use_existing_documents : if True, skip retrieval and use the documents
            already present in each example (useful when benchmarking
            only the generation component).

        Returns
        -------
        List of generated response strings (one per example).
        """
        responses = []
        for ex in examples:
            if use_existing_documents:
                resp = self.generate(ex["question"], ex["documents"])
            else:
                result = self.run(ex["question"])
                resp = result["response"]
            responses.append(resp)
        return responses

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
