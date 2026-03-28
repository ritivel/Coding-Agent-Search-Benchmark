# Advanced RAG Techniques Reference

## Table of Contents
1. [Query Transformation](#query-transformation)
2. [Corrective RAG (CRAG)](#corrective-rag-crag)
3. [Self-RAG](#self-rag)
4. [Agentic RAG](#agentic-rag)
5. [GraphRAG](#graphrag)
6. [Multi-Agent RAG](#multi-agent-rag)
7. [Adaptive Retrieval](#adaptive-retrieval)

---

## Query Transformation

Before retrieval, transform the user query for better results. Three main techniques:

### Query Rewriting

```python
def rewrite_query(query: str, client) -> str:
    """Rewrite query for better retrieval."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=256,
        messages=[{"role": "user", "content": f"""Rewrite this question to be more
specific and search-friendly. Return only the rewritten query.
Original: {query}"""}])
    return response.content[0].text.strip()
```

### Query Decomposition (Sub-Questions)

Break complex queries into simpler sub-queries, retrieve for each, merge.

```python
def decompose_query(query: str, client) -> list[str]:
    """Split complex query into sub-questions."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=512,
        messages=[{"role": "user", "content": f"""Break this complex question into
2-4 simpler sub-questions that, when answered together, fully address the original.
Return one sub-question per line, nothing else.
Question: {query}"""}])
    return [q.strip() for q in response.content[0].text.strip().split("\n") if q.strip()]
```

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, embed it, use that for retrieval. Works well for vague queries.

```python
def hyde_retrieval(query: str, client, embed_fn, vectorstore, k=5):
    """Generate hypothetical answer, embed it, retrieve similar real docs."""
    # Generate hypothetical answer
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=512,
        messages=[{"role": "user", "content": f"""Write a short paragraph that would
answer this question. It doesn't need to be factual — just plausible.
Question: {query}"""}])
    hypothetical = response.content[0].text

    # Embed hypothetical answer and retrieve
    hyp_emb = embed_fn([hypothetical])[0]
    results = vectorstore.query(query_embeddings=[hyp_emb], n_results=k,
                                 include=["documents", "metadatas"])
    return results
```

### Multi-Query Retrieval

Generate multiple query variants, retrieve for each, deduplicate.

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    llm=llm,
)
results = multi_retriever.invoke("What are the system requirements?")
```

---

## Corrective RAG (CRAG)

Add a validation step: before generating, check if retrieved documents are actually relevant. If not, trigger fallback (web search, different index, or "I don't know").

```python
def crag_pipeline(query, retriever, client, embed_fn, k=5):
    """Corrective RAG: validate retrieval, fallback if needed."""
    # 1. Retrieve
    docs = retriever.invoke(query)

    # 2. Grade relevance
    relevant_docs = []
    for doc in docs:
        grade = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=10,
            messages=[{"role": "user", "content": f"""Is this document relevant to the question?
Question: {query}
Document: {doc.page_content[:500]}
Answer YES or NO only."""}])
        if "YES" in grade.content[0].text.upper():
            relevant_docs.append(doc)

    # 3. Decide action
    if len(relevant_docs) == 0:
        # Fallback: web search or admit no answer
        return "I don't have enough information to answer this question."
    elif len(relevant_docs) < 2:
        # Partial info: supplement with web search
        # web_results = tavily_search(query)  # if you have a web search tool
        pass

    # 4. Generate from validated docs
    context = "\n\n".join([d.page_content for d in relevant_docs])
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2048,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])
    return response.content[0].text
```

### LangGraph Implementation (Recommended)

```python
from langgraph.graph import StateGraph, END

def build_crag_graph():
    """Build CRAG as a LangGraph state machine."""
    graph = StateGraph(dict)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("web_search", web_search_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", decide_to_generate,
        {"generate": "generate", "web_search": "web_search"})
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
```

---

## Self-RAG

The LLM itself decides: (a) whether retrieval is needed, (b) if retrieved docs are relevant, (c) if its answer is supported by the docs. Uses special reflection tokens during generation.

### Simplified Implementation

```python
def self_rag(query, retriever, client):
    """Self-RAG: LLM controls retrieval and self-critiques."""
    # Step 1: Does the query need retrieval?
    need = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=50,
        messages=[{"role": "user", "content": f"""Can this question be answered from
general knowledge, or does it need specific document retrieval?
Question: {query}
Answer RETRIEVE or GENERAL only."""}])

    if "GENERAL" in need.content[0].text.upper():
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            messages=[{"role": "user", "content": query}])
        return response.content[0].text

    # Step 2: Retrieve and generate
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs[:5]])
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}])
    answer = response.content[0].text

    # Step 3: Self-critique — is the answer supported?
    critique = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=100,
        messages=[{"role": "user", "content": f"""Is this answer fully supported by the context?
Context: {context[:1000]}
Answer: {answer}
Reply SUPPORTED, PARTIALLY_SUPPORTED, or NOT_SUPPORTED."""}])

    support_level = critique.content[0].text.strip().upper()
    if "NOT_SUPPORTED" in support_level:
        return f"[Low confidence] {answer}\n\nNote: This answer may not be fully supported by the available documents."
    return answer
```

---

## Agentic RAG

Instead of a fixed pipeline, give the LLM tools to search, evaluate, and iterate. The LLM drives retrieval as an agent.

### With LangChain Tools

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic

# Define retrieval as a tool
def search_knowledge_base(query: str) -> str:
    results = vectorstore.similarity_search(query, k=5)
    return "\n\n".join([r.page_content for r in results])

def search_specific_source(query_and_source: str) -> str:
    """Search a specific document. Format: 'query | source_name'"""
    parts = query_and_source.split("|")
    query, source = parts[0].strip(), parts[1].strip()
    results = vectorstore.similarity_search(query, k=5,
        filter={"source": source})
    return "\n\n".join([r.page_content for r in results])

tools = [
    Tool(name="search_kb", func=search_knowledge_base,
         description="Search the knowledge base for information. Input: search query."),
    Tool(name="search_source", func=search_specific_source,
         description="Search a specific document. Input: 'query | document_name'"),
]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
agent = create_tool_calling_agent(llm, tools, prompt_template)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
result = executor.invoke({"input": "Compare the refund policies across all our products"})
```

---

## GraphRAG

Use knowledge graphs for structured relationships between entities. Best for queries that require reasoning across connections (e.g., "What companies are related to X?").

### When to Use GraphRAG

ICLR 2026 paper (GraphRAG-Bench) found: GraphRAG frequently underperforms vanilla RAG on many real-world tasks. Use it when:
- Queries require multi-hop reasoning across entities
- Data has rich relational structure (org charts, supply chains, citation networks)
- Questions are about relationships ("Who reports to whom?", "What's connected to X?")

Don't use it for: simple factual Q&A, document summarization, or when data is mostly unstructured text.

### Microsoft GraphRAG

```bash
pip install graphrag
```

```python
# GraphRAG builds a knowledge graph from documents, then queries it
# See: https://github.com/microsoft/graphrag
# Two query modes:
# - Local search: entity-centric, uses graph neighborhoods
# - Global search: community summaries, for broad questions
```

### LightRAG (Simpler Alternative)

```bash
pip install lightrag-hku
```

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(working_dir="./lightrag_data")
rag.insert(documents)
result = rag.query("What are the key relationships?", param=QueryParam(mode="hybrid"))
```

---

## Multi-Agent RAG

Specialized agents handle different parts of the pipeline. Best for complex queries over diverse sources.

```python
# Conceptual architecture:
# Agent 1 (Router): Analyzes query, routes to appropriate specialist
# Agent 2 (Vector Search): Searches embeddings in vector store
# Agent 3 (SQL Agent): Queries structured databases
# Agent 4 (Web Search): Falls back to internet search
# Agent 5 (Synthesizer): Combines all results into coherent answer

# Implementation with LangGraph:
from langgraph.graph import StateGraph

def build_multi_agent_rag():
    graph = StateGraph(dict)
    graph.add_node("router", route_query)
    graph.add_node("vector_search", vector_search_agent)
    graph.add_node("sql_search", sql_search_agent)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("synthesizer", synthesize_results)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "vector": "vector_search",
        "sql": "sql_search",
        "web": "web_search",
    })
    # All paths lead to synthesizer
    for node in ["vector_search", "sql_search", "web_search"]:
        graph.add_edge(node, "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()
```

---

## Adaptive Retrieval

Auto-tune retrieval parameters (top-k, chunk size, alpha for hybrid) based on query characteristics.

```python
def adaptive_retrieve(query, vectorstore, embed_fn, client):
    """Classify query complexity, adjust retrieval strategy."""
    # Classify query
    classification = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=50,
        messages=[{"role": "user", "content": f"""Classify this query:
SIMPLE (single fact), MODERATE (needs context), COMPLEX (multi-step reasoning).
Query: {query}
Reply with one word only."""}])

    level = classification.content[0].text.strip().upper()

    if "SIMPLE" in level:
        k, use_rerank = 3, False
    elif "COMPLEX" in level:
        k, use_rerank = 20, True
    else:
        k, use_rerank = 10, True

    # Retrieve with adapted parameters
    q_emb = embed_fn([query])[0]
    results = vectorstore.query(query_embeddings=[q_emb], n_results=k,
                                 include=["documents", "distances"])

    if use_rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc) for doc in results["documents"][0]]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(results["documents"][0], scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:5]]

    return results["documents"][0]
```
