# Embeddings, Vector Stores & Retrieval Reference

## Table of Contents
1. [Embedding Models](#embedding-models)
2. [Vector Stores](#vector-stores)
3. [Hybrid Search (BM25 + Vector)](#hybrid-search)
4. [Re-Ranking](#re-ranking)
5. [Full Retrieval Pipeline](#full-retrieval-pipeline)

---

## Embedding Models

### Open-Source (sentence-transformers)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# ---- Prototyping (CPU-friendly, fast) ----
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384d, 22M params

# ---- Production English ----
# model = SentenceTransformer("BAAI/bge-large-en-v1.5")  # 1024d, 335M params

# ---- Multilingual ----
# model = SentenceTransformer("BAAI/bge-m3")  # 1024d, 100+ languages, MIT

# ---- Top open-source quality (GPU needed) ----
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  # 1024d, MTEB ~67%

embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
```

**BGE models:** Prefix queries with `"Represent this sentence: "` for optimal results.

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

def embed_openai(texts, model="text-embedding-3-small", dimensions=None):
    """text-embedding-3-small: 1536d, $0.02/1M tok
       text-embedding-3-large: 3072d, $0.13/1M tok, MTEB 64.6%
       Both support dimension reduction via 'dimensions' param."""
    kwargs = {"model": model, "input": texts}
    if dimensions:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]
```

### Cohere

```python
import cohere
co = cohere.Client()

def embed_cohere(texts, input_type="search_document"):
    """embed-v4: MTEB 65.2%. input_type: 'search_document' or 'search_query'."""
    return co.embed(texts=texts, model="embed-v4", input_type=input_type).embeddings
```

### Fine-Tuning Embeddings

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
examples = [InputExample(texts=["query", "positive_doc", "negative_doc"])]
loader = DataLoader(examples, shuffle=True, batch_size=16)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=3, output_path="./finetuned")
```

Domain fine-tuning typically yields +10-30% gains on specialized data (legal, medical, code).

---

## Vector Stores

### ChromaDB (Prototyping, Local)

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

# Index
collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)

# Search
results = collection.query(query_embeddings=[query_emb], n_results=5,
                            where={"source": "manual.pdf"},
                            include=["documents", "metadatas", "distances"])

# Zero-config with auto-embedding:
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection("docs", embedding_function=ef)
collection.add(documents=texts, ids=ids)  # auto-embeds
```

### Qdrant (Production, Complex Filtering)

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(path="./qdrant_db")  # or url="http://localhost:6333"
client.create_collection("docs", vectors_config=VectorParams(size=384, distance=Distance.COSINE))

# Index
client.upsert("docs", points=[
    PointStruct(id=i, vector=emb, payload={"text": text, "source": src})
    for i, (emb, text, src) in enumerate(zip(embs, texts, sources))
])

# Search with filter
results = client.search("docs", query_vector=q_emb, limit=5,
    query_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="manual.pdf"))]))
```

### Pinecone (Managed, Scale)

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone()  # PINECONE_API_KEY env var
if "docs" not in pc.list_indexes().names():
    pc.create_index("docs", dimension=384, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index("docs")

# Upsert (batch 100)
vectors = [{"id": f"c_{i}", "values": emb, "metadata": {"text": text, **meta}}
           for i, (emb, text, meta) in enumerate(zip(embs, texts, metas))]
for i in range(0, len(vectors), 100):
    index.upsert(vectors=vectors[i:i+100])

# Query
results = index.query(vector=q_emb, top_k=5, include_metadata=True,
                      filter={"source": {"$eq": "manual.pdf"}})
```

### Weaviate (Hybrid Search, GraphQL)

```python
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import MetadataQuery

client = weaviate.connect_to_local()
coll = client.collections.create("Docs", properties=[
    Property(name="text", data_type=DataType.TEXT),
    Property(name="source", data_type=DataType.TEXT),
], vectorizer_config=Configure.Vectorizer.none())

# Hybrid search (BM25 + vector)
results = coll.query.hybrid(query="refund policy", vector=q_emb,
    alpha=0.7, limit=5, return_metadata=MetadataQuery(score=True))
```

### pgvector (PostgreSQL)

```sql
CREATE EXTENSION vector;
CREATE TABLE chunks (id serial PRIMARY KEY, text text, embedding vector(384),
                     source text, created_at timestamptz DEFAULT now());
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Search
SELECT text, source, 1 - (embedding <=> $1::vector) AS score
FROM chunks ORDER BY embedding <=> $1::vector LIMIT 5;
```

```python
# With SQLAlchemy + pgvector
from pgvector.sqlalchemy import Vector
# Or with LangChain:
from langchain_community.vectorstores import PGVector
```

---

## Hybrid Search

Combining BM25 (keyword/sparse) with vector (dense) search is the gold standard for retrieval. Weaviate has it built-in. For others, implement manually:

### With Qdrant (Sparse + Dense from v1.10+)

```python
from qdrant_client.models import SparseVector, NamedSparseVector

# Qdrant supports sparse vectors natively for hybrid search
# Use a sparse encoder like SPLADE or BM25
```

### Manual Hybrid with Reciprocal Rank Fusion (RRF)

Works with any vector store + any keyword search:

```python
from rank_bm25 import BM25Okapi

def hybrid_search(query: str, chunks: list[str], chunk_embeddings, embed_fn,
                  vectorstore, alpha=0.7, k=5):
    """Combine BM25 and vector search using Reciprocal Rank Fusion."""
    # Vector search
    q_emb = embed_fn([query])[0]
    vector_results = vectorstore.query(query_embeddings=[q_emb], n_results=20)
    vector_ids = vector_results["ids"][0]

    # BM25 search
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    bm25_ids = [f"chunk_{i}" for i in bm25_top]

    # Reciprocal Rank Fusion
    rrf_scores = {}
    rrf_k = 60  # standard RRF constant
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (rrf_k + rank + 1)

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return top_ids
```

### With LangChain EnsembleRetriever

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=20)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.3, 0.7])
results = ensemble.invoke("What is the refund policy?")
```

---

## Re-Ranking

Two-stage retrieval: retrieve broadly (top 20), re-rank to top 5. Consistently improves precision.

### Cross-Encoder (Local, Free)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Or for multilingual: CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, docs: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

### Cohere Reranker (API, Best Commercial)

```python
import cohere
co = cohere.Client()

def rerank_cohere(query, docs, top_k=5):
    response = co.rerank(query=query, documents=docs, model="rerank-v3.5", top_n=top_k)
    return [(docs[r.index], r.relevance_score) for r in response.results]
```

### With LangChain

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
results = retriever.invoke("What is the refund policy?")
```

---

## Full Retrieval Pipeline

Best-practice production pipeline: hybrid search → re-rank → generate.

```python
import anthropic

def production_rag_query(query, vectorstore, chunks, embed_fn, top_k=5):
    """Hybrid search → re-rank → generate."""
    # 1. Hybrid retrieval
    top_ids = hybrid_search(query, chunks, None, embed_fn, vectorstore, alpha=0.7, k=20)
    candidates = [chunks[int(id.split("_")[1])] for id in top_ids]

    # 2. Re-rank
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = rerank(query, candidates, top_k=top_k)
    context = "\n\n---\n\n".join([doc for doc, score in reranked])

    # 3. Generate
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2048,
        system="Answer based only on the provided context. If the context doesn't contain the answer, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return response.content[0].text
```
