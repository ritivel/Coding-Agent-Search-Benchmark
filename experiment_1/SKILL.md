---
name: rag-pipeline
description: >
  Build production-grade Retrieval-Augmented Generation (RAG) pipelines using the best
  available components and techniques. Use this skill whenever the user wants to build,
  improve, or evaluate a RAG system — including document ingestion, chunking, embedding,
  vector storage, retrieval, re-ranking, generation, or evaluation. Also trigger for
  questions about RAG frameworks (LangChain, LlamaIndex, Haystack), vector databases,
  embedding models, chunking strategies, hybrid search, agentic RAG, Corrective-RAG,
  Self-RAG, re-ranking, RAG evaluation (RAGAS, DeepEval, RAGBench), or any query about
  connecting LLMs with external knowledge sources. Even if the user doesn't say "RAG"
  explicitly, trigger if they mention semantic search, knowledge base Q&A, document
  chat, or grounding LLMs with custom data.
---

# RAG Pipeline — Build the Best Retrieval-Augmented Generation Systems

This skill provides concrete, research-backed component recommendations and working code
for every stage of a RAG pipeline. No searching required — all recommendations are baked in.

**Reference files** (read the relevant one for implementation details):
- `references/ingestion-and-chunking.md` — Document parsing, cleaning, 6 chunking strategies with code
- `references/embeddings-and-retrieval.md` — Embedding models, vector stores, hybrid search, re-ranking
- `references/advanced-rag.md` — Agentic RAG, CRAG, Self-RAG, GraphRAG, query transforms, multi-agent
- `references/evaluation.md` — RAGAS, DeepEval, RAGBench, TRACe, CI/CD integration, golden datasets

---

## Pipeline Architecture

```
Ingestion:   Documents → Parse → Clean → Chunk → Embed → Index (Vector Store)
Query:       User Query → Transform → Embed → Retrieve → Re-rank → Generate → Evaluate
```

---

## Component Recommendations at a Glance

### Frameworks

| Framework | Best For | Notes |
|---|---|---|
| **LangChain** | Flexible prototyping, huge ecosystem | Most integrations, LangGraph for agents |
| **LlamaIndex** | Document-heavy RAG, structured indexing | Best ingestion/parsing, built-in eval |
| **Haystack** | Production modular pipelines | Component-swappable, good for teams |
| **RAGFlow** | Visual/low-code RAG | Drag-and-drop DAG editor |
| **None (custom)** | Full control, minimal deps | Best when you know exactly what you need |

### Document Parsing

| Tool | Best For | Install |
|---|---|---|
| **Docling** (IBM) | Universal doc parsing (PDF, DOCX, PPTX, HTML) | `pip install docling` |
| **Unstructured** | Broad format support, partitioning | `pip install unstructured` |
| **PyMuPDF / pdfplumber** | Fast PDF text extraction | `pip install pymupdf pdfplumber` |
| **Marker** | PDF → Markdown with layout | `pip install marker-pdf` |
| **PaddleOCR** | Scanned docs, images with text | `pip install paddleocr` |

### Chunking (→ see `references/ingestion-and-chunking.md`)

| Strategy | When to Use | Typical Size |
|---|---|---|
| **Recursive splitting** | Default for 80% of cases | 512 tokens, 64 overlap |
| **Semantic chunking** | Unstructured text, no clear sections | 256-512 tokens |
| **Document-based** | Markdown/HTML with headers | Varies by section |
| **Token-based** | Strict LLM context budget | 256 tokens |
| **Late chunking** | When you want full-doc context in embeddings | Full doc → split after |
| **Contextual (Anthropic)** | Prepend LLM-generated summary per chunk | 512 tokens + summary |

### Embedding Models (→ see `references/embeddings-and-retrieval.md`)

| Model | Dims | MTEB | Cost | Best For |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~56% | Free | Prototypes, CPU |
| `BAAI/bge-large-en-v1.5` | 1024 | ~64% | Free | Production English |
| `BAAI/bge-m3` | 1024 | ~63% | Free | Multilingual, MIT |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | ~67% | Free | Top open-source |
| OpenAI `text-embedding-3-large` | 3072 | ~64.6% | $0.13/1M tok | Best cloud |
| Cohere `embed-v4` | — | ~65.2% | Pay | Enterprise multilingual |

### Vector Stores (→ see `references/embeddings-and-retrieval.md`)

| Store | Best For | Hybrid Search | Managed |
|---|---|---|---|
| **ChromaDB** | Prototyping, <10M vectors | Metadata only | Chroma Cloud |
| **Qdrant** | Production, complex filters | BM25 + vector (v1.10+) | Qdrant Cloud |
| **Pinecone** | Managed production at scale | Metadata only | Yes (only) |
| **Weaviate** | Hybrid search, GraphQL | Native BM25 + vector | Self-host or Cloud |
| **Milvus** | Massive scale (billions) | Sparse + dense | Zilliz Cloud |
| **pgvector** | Already using PostgreSQL | With pg_search | Self-host |

### Re-Ranking (→ see `references/embeddings-and-retrieval.md`)

| Model | Type | Best For |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | Fast, good default |
| Cohere `rerank-v3.5` | API | Best commercial |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder | Multilingual, open-source |
| `Qwen/Qwen3-Reranker-0.6B` | LLM-based | Top open-source quality |
| Jina `jina-reranker-v2` | API/local | Good balance |

### Evaluation (→ see `references/evaluation.md`)

| Tool | Type | Key Metrics |
|---|---|---|
| **RAGAS** | OSS framework | Faithfulness, context precision/recall, answer relevancy |
| **DeepEval** | OSS pytest-native | 5-metric RAG triad, CI/CD integration |
| **RAGBench** | Benchmark dataset | TRACe metrics, 100K examples, cross-domain |
| **LangSmith** | Platform | Tracing, evals, LangChain-native |
| **Arize Phoenix** | OSS observability | Embedding viz, retrieval debugging |

---

## Decision Matrices

### By Budget

| Budget | Parsing | Embedding | Vector Store | Re-ranker | Eval |
|---|---|---|---|---|---|
| **$0** | PyMuPDF + Unstructured | all-MiniLM-L6-v2 | ChromaDB | cross-encoder (local) | RAGAS |
| **Low** | Docling | bge-large-en-v1.5 | Qdrant (self-host) | bge-reranker-v2-m3 | DeepEval |
| **Production** | Docling + Marker | OpenAI embed-3-large | Pinecone or Weaviate | Cohere rerank-v3.5 | LangSmith + RAGAS |

### By Use Case

| Use Case | Architecture | Key Components |
|---|---|---|
| **Simple doc Q&A** | Basic RAG | Recursive chunking → embed → retrieve → generate |
| **Multi-source Q&A** | Agentic RAG | Query router → specialized retrievers → synthesize |
| **Accuracy-critical** | CRAG + re-ranking | Retrieve → validate → fallback web search → re-rank → generate |
| **Complex reasoning** | Self-RAG or multi-agent | Self-reflective generation with retrieval tokens |
| **Large knowledge base** | Hybrid search + re-rank | BM25 + vector → cross-encoder re-rank → generate |
| **Structured data + text** | Text-to-SQL + RAG | Route SQL vs vector, merge results |

### Advanced Techniques Worth Implementing (→ see `references/advanced-rag.md`)

1. **Hybrid search** (BM25 + vector): Gold standard for retrieval. 5-15% recall improvement.
2. **Re-ranking**: Retrieve 20, re-rank to top 5. Consistently improves precision.
3. **Query transformation**: Rewrite, decompose, or HyDE. Biggest wins on vague queries.
4. **Contextual retrieval**: Prepend chunk summaries before embedding. +49% retrieval accuracy.
5. **Corrective RAG (CRAG)**: Validate retrieved docs, fallback to web search if irrelevant.
6. **Late chunking**: Embed full document, then split. Preserves cross-sentence context.

---

## Quick-Start: Minimal Working Pipeline

```python
# pip install langchain langchain-community chromadb sentence-transformers

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Ingest
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)

# Embed + Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("What is the refund policy?")
```

→ For production-grade implementations, read the reference files.
