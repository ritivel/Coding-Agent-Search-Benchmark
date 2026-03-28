# Ingestion & Chunking Reference

## Table of Contents
1. [Document Parsing](#document-parsing)
2. [Text Cleaning for RAG](#text-cleaning-for-rag)
3. [Chunking Strategies](#chunking-strategies)
4. [Contextual Retrieval (Anthropic Technique)](#contextual-retrieval)
5. [Late Chunking](#late-chunking)
6. [Metadata Enrichment](#metadata-enrichment)

---

## Document Parsing

### Docling (IBM) — Best Universal Parser

Handles PDF, DOCX, PPTX, HTML, images. OCR built-in. MIT license.

```bash
pip install docling
```

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def parse_document(file_path: str) -> str:
    """Parse any document to markdown."""
    result = converter.convert(file_path)
    return result.document.export_to_markdown()
```

### Unstructured — Broadest Format Support

```bash
pip install "unstructured[all-docs]"
```

```python
from unstructured.partition.auto import partition

def parse_with_unstructured(file_path: str) -> list[dict]:
    elements = partition(file_path)
    return [{"type": type(el).__name__, "text": str(el), "metadata": el.metadata.to_dict()}
            for el in elements]
```

### PyMuPDF — Fast PDF Text

```python
import pymupdf

def parse_pdf(path: str) -> list[dict]:
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        pages.append({"page": i+1, "text": page.get_text()})
    return pages
```

### Marker — PDF to Markdown with Layout

```bash
pip install marker-pdf
```

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

models = create_model_dict()
converter = PdfConverter(artifact_dict=models)

def pdf_to_markdown(path: str) -> str:
    rendered = converter(path)
    return rendered.markdown
```

---

## Text Cleaning for RAG

```python
import re

def clean_for_rag(text: str) -> str:
    """Clean parsed text before chunking."""
    text = re.sub(r'\n{3,}', '\n\n', text)           # collapse blank lines
    text = re.sub(r'[ \t]+', ' ', text)               # normalize whitespace
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)       # remove excessive newlines
    text = re.sub(r'Page \d+ of \d+', '', text)        # remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # stray numbers
    text = text.strip()
    return text
```

---

## Chunking Strategies

### 1. Recursive Character Splitting (Default)

Best for 80% of cases. Tries paragraph breaks first, then sentences, then words.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_text(text)
```

**Recommended defaults:** 512 tokens, 10-15% overlap. Research shows 400-512 token chunks with `text-embedding-3-large` achieve 88-89% recall (Chroma 2025 study).

### 2. Semantic Chunking

Groups sentences by embedding similarity. Best when text lacks paragraph structure.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75,
)
docs = chunker.create_documents([text])
```

**Caveat:** NAACL 2025 paper found semantic chunking doesn't consistently beat fixed 200-word chunks on retrieval+generation. Benchmark on your data. Can produce very small fragments (~43 tokens avg in some tests).

### 3. Document-Based (Structure-Aware)

Splits by document structure: headers, sections, code blocks.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk has metadata: {"h1": "...", "h2": "..."}
```

### 4. Token-Based

Ensures chunks fit exactly within model token limits.

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
chunks = splitter.split_text(text)
```

### 5. Hierarchical Chunking

Multiple layers: summaries for broad queries, detail chunks for specific ones.

```python
# Create two levels
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

parent_chunks = parent_splitter.split_text(text)
for parent in parent_chunks:
    children = child_splitter.split_text(parent)
    # Index both: children for retrieval, parent for context
```

Use with LlamaIndex `AutoMergingRetriever` or LangChain `ParentDocumentRetriever`:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
# Retrieves child chunks, returns parent context
```

### 6. Agentic/Dynamic Chunking

LLM decides chunk boundaries. Expensive but highest quality for complex docs.

```python
def llm_chunk(text: str, client) -> list[str]:
    """Use LLM to identify natural breakpoints."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4096,
        messages=[{"role": "user", "content": f"""Split this text into semantically
coherent chunks. Each chunk should be a self-contained unit of information.
Return chunks separated by ===CHUNK_BREAK===. Do not modify the text.

Text:
{text}"""}])
    return [c.strip() for c in response.content[0].text.split("===CHUNK_BREAK===") if c.strip()]
```

---

## Contextual Retrieval

Anthropic technique: prepend a short LLM-generated context summary to each chunk before embedding. Reported +49% retrieval accuracy when combined with BM25 hybrid search.

```python
def add_chunk_context(chunk: str, full_document: str, client) -> str:
    """Prepend contextual summary to chunk before embedding."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=200,
        messages=[{"role": "user", "content": f"""<document>{full_document}</document>

Here is a chunk from the document:
<chunk>{chunk}</chunk>

Write a short (2-3 sentence) context that situates this chunk within the overall
document. The context should help a search engine understand what this chunk is about.
Return ONLY the context, nothing else."""}])
    context = response.content[0].text
    return f"{context}\n\n{chunk}"
```

**Cost optimization:** Cache contexts. Use a smaller model (Haiku) for context generation. Only apply to chunks where retrieval tests show low recall.

---

## Late Chunking

Embed the full document first using a long-context embedding model, then split the embeddings into chunks. Preserves cross-sentence context that normal chunking loses.

```python
# Requires a long-context embedding model like jina-embeddings-v2 (8K context)
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "jinaai/jina-embeddings-v2-base-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

def late_chunk(text: str, chunk_size: int = 512):
    """Embed full document, then split embeddings by token spans."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**tokens)
    # Full-doc token embeddings
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    # Split into chunks by token spans
    chunks, chunk_embeddings = [], []
    token_texts = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    for start in range(0, len(token_texts), chunk_size):
        end = min(start + chunk_size, len(token_texts))
        chunk_text = tokenizer.decode(tokens["input_ids"][0][start:end])
        chunk_emb = token_embeddings[start:end].mean(dim=0)  # pool chunk tokens
        chunks.append(chunk_text)
        chunk_embeddings.append(chunk_emb)
    return chunks, chunk_embeddings
```

---

## Metadata Enrichment

Always attach metadata to chunks for filtering and debugging:

```python
from langchain_core.documents import Document

def enrich_chunks(chunks: list[str], source: str, extra: dict = None) -> list[Document]:
    docs = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source": source,
            "chunk_index": i,
            "chunk_size": len(chunk),
            **(extra or {}),
        }
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs
```

For production: add `created_at`, `document_version`, `section_title`, `confidence_score`.
Filter on metadata during retrieval to scope results (e.g., only recent docs, specific sources).
