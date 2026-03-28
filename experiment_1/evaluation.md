# RAG Evaluation Reference

## Table of Contents
1. [Core RAG Metrics](#core-rag-metrics)
2. [RAGAS Framework](#ragas)
3. [DeepEval Framework](#deepeval)
4. [RAGBench & TRACe](#ragbench--trace)
5. [Building Golden Datasets](#building-golden-datasets)
6. [CI/CD Integration](#cicd-integration)
7. [Production Monitoring](#production-monitoring)

---

## Core RAG Metrics

RAG evaluation has two distinct targets: **retrieval quality** and **generation quality**.

### Retrieval Metrics

| Metric | What It Measures | Good Score |
|---|---|---|
| **Context Precision** | Are retrieved docs relevant? (precision of top-k) | >0.85 |
| **Context Recall** | Did retrieval find all needed info? | >0.80 |
| **Context Relevancy** | Ratio of relevant sentences in retrieved context | >0.70 |
| **Hit Rate / Recall@k** | Is the correct doc in top-k results? | >0.90 |
| **MRR (Mean Reciprocal Rank)** | How high is the first relevant result ranked? | >0.75 |
| **nDCG@k** | Quality of ranking across top-k | >0.70 |

### Generation Metrics

| Metric | What It Measures | Good Score |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) | >0.90 |
| **Answer Relevancy** | Does the answer address the question? | >0.85 |
| **Answer Correctness** | Is the answer factually correct vs ground truth? | >0.80 |
| **Answer Completeness** | Does it cover all aspects of the question? | >0.75 |

### Score Targets by Domain

- General Q&A: faithfulness >0.80, relevancy >0.80
- Customer-facing documentation: faithfulness >0.85
- Regulated (finance, healthcare, legal): faithfulness >0.90

---

## RAGAS

Open-source framework for RAG evaluation. Reference-free (no ground truth needed for some metrics). Best for quick experimentation and metric exploration.

```bash
pip install ragas
```

### Basic Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import EvaluationDataset, SingleTurnSample

# Prepare samples
samples = []
for q, answer, contexts, ground_truth in zip(questions, answers, retrieved_contexts, ground_truths):
    samples.append(SingleTurnSample(
        user_input=q,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    ))

dataset = EvaluationDataset(samples=samples)
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
# Returns per-metric scores and overall dataset scores
```

### Synthetic Test Data Generation

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_anthropic import ChatAnthropic

llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-20250514"))
generator = TestsetGenerator(llm=llm)

# Generate from your documents
testset = generator.generate_with_langchain_docs(
    documents=docs,    # LangChain Document objects
    testset_size=50,   # number of test questions
)
testset_df = testset.to_pandas()
```

### Known Limitations

- NaN scores when LLM judge returns invalid JSON (no graceful fallback)
- Pin your ragas version and wrap eval calls in try/except
- Metrics can be inconsistent across runs — average over 3+ runs
- Some metrics require ground truth (context_recall), others don't (faithfulness)

---

## DeepEval

Test-driven RAG evaluation. Pytest-native. Maps each metric to a specific pipeline component so you know *what to fix*. Best for CI/CD integration.

```bash
pip install deepeval
```

### RAG Metrics (5-Metric Triad)

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

# Each metric maps to a pipeline component:
# AnswerRelevancy    → prompt template / generation
# Faithfulness       → generation model / grounding
# ContextualRelevancy → embedding model / retrieval
# ContextualRecall   → chunking strategy / coverage
# ContextualPrecision → re-ranker / ranking quality

test_case = LLMTestCase(
    input="What is the refund policy?",
    actual_output="You can get a full refund within 30 days.",
    expected_output="Customers are eligible for a 30-day full refund.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

metrics = [
    AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
    FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
    ContextualRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
    ContextualRecallMetric(threshold=0.7, model="gpt-4o-mini"),
    ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini"),
]

results = evaluate(test_cases=[test_case], metrics=metrics)
```

### Pytest Integration

```python
# test_rag.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

def test_refund_policy():
    test_case = LLMTestCase(
        input="What is the refund policy?",
        actual_output=rag_pipeline("What is the refund policy?"),
        retrieval_context=get_retrieved_contexts("What is the refund policy?"),
    )
    metric = FaithfulnessMetric(threshold=0.8, model="gpt-4o-mini")
    assert_test(test_case, [metric])

# Run: deepeval test run test_rag.py
```

### Synthetic Dataset Generation

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer(model="gpt-4o-mini")
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["./docs/manual.pdf", "./docs/faq.txt"],
    max_goldens_per_context=2,
)
# Returns test cases with questions, expected answers, and source contexts
```

### DAG Metric (Deterministic, No LLM Judge)

Avoids LLM-as-judge non-determinism:

```python
from deepeval.metrics import DAGMetric

# Define scoring as a decision tree — deterministic and reproducible
dag_metric = DAGMetric(
    name="answer_quality",
    criteria="Does the answer address the question using information from the context?",
    threshold=0.7,
)
```

---

## RAGBench & TRACe

RAGBench is a 100K-example benchmark dataset across multiple domains (user manuals, StackOverflow, etc.). TRACe is its evaluation framework with 4 metrics:

| TRACe Metric | What It Measures |
|---|---|
| **Context Relevance** | Are retrieved documents relevant to the query? |
| **Context Utilization** | Does the response actually use the retrieved context? |
| **Answer Faithfulness** | Is the response faithful to the context (no hallucination)? |
| **Answer Completeness** | Does the response fully address the query? |

```python
# Load RAGBench dataset
from datasets import load_dataset

ragbench = load_dataset("rungalileo/ragbench")
# Subsets: covidqa, cuad, delucionqa, emanual, expertqa, finqa, hagrid, msmarco, pubmedqa, tatqa, techqa
```

Key finding: A fine-tuned DeBERTa model on RAGBench achieves competitive performance with billion-parameter LLM judges, at a fraction of the cost.

---

## Building Golden Datasets

Golden datasets are critical. 50 well-curated questions give stable metrics. 200-500 for comprehensive regression testing.

### Strategy

1. **Start manual**: Write 20-50 question/answer pairs from your actual docs
2. **Augment with synthetic**: Use RAGAS or DeepEval generators for 100-200 more
3. **Validate synthetics**: Human review at least 20% of generated questions
4. **Version everything**: Freeze dataset versions for comparable results across experiments

### Manual Golden Dataset Template

```python
golden_dataset = [
    {
        "question": "What is the refund policy for annual subscriptions?",
        "ground_truth_answer": "Annual subscriptions are eligible for a prorated refund within the first 60 days.",
        "expected_source_docs": ["pricing_policy.md"],
        "difficulty": "simple",  # simple, moderate, complex
        "category": "policy",
    },
    {
        "question": "How do the enterprise and team plans differ in terms of SSO support?",
        "ground_truth_answer": "Enterprise plans include SAML SSO; team plans support Google/GitHub OAuth only.",
        "expected_source_docs": ["enterprise_features.md", "team_features.md"],
        "difficulty": "moderate",
        "category": "comparison",
    },
]
```

### Automated Evaluation Loop

```python
def evaluate_pipeline(pipeline_fn, golden_dataset, metrics):
    """Run full evaluation against golden dataset."""
    test_cases = []
    for item in golden_dataset:
        answer, contexts = pipeline_fn(item["question"])
        test_cases.append(LLMTestCase(
            input=item["question"],
            actual_output=answer,
            expected_output=item["ground_truth_answer"],
            retrieval_context=contexts,
        ))
    return evaluate(test_cases=test_cases, metrics=metrics)
```

---

## CI/CD Integration

### GitHub Actions with DeepEval

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install deepeval
      - run: deepeval test run tests/test_rag.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Pre-Commit Quality Gate

```python
# tests/test_rag_quality.py
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric

QUALITY_THRESHOLD = 0.8

@pytest.fixture
def rag_pipeline():
    """Initialize your RAG pipeline."""
    # ... setup code
    return pipeline

@pytest.mark.parametrize("question,expected", GOLDEN_DATASET)
def test_rag_faithfulness(rag_pipeline, question, expected):
    output, contexts = rag_pipeline(question)
    case = LLMTestCase(input=question, actual_output=output,
                       expected_output=expected, retrieval_context=contexts)
    assert_test(case, [FaithfulnessMetric(threshold=QUALITY_THRESHOLD)])
```

---

## Production Monitoring

### Key Signals to Track

1. **Retrieval latency** (p50, p95, p99)
2. **Empty retrieval rate** (queries returning 0 relevant docs)
3. **Average similarity score** of top-k results (dropping = index drift)
4. **Faithfulness score** on sampled production queries
5. **User feedback** (thumbs up/down, regeneration rate)

### With Arize Phoenix (Open-Source Observability)

```bash
pip install arize-phoenix
```

```python
import phoenix as px

# Launch dashboard
session = px.launch_app()

# Auto-instrument LangChain
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(project_name="rag-prod")
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Now all LangChain calls are traced — view at http://localhost:6006
```

### Lightweight Custom Logging

```python
import json, time
from datetime import datetime

def monitored_query(query, pipeline_fn, log_file="rag_log.jsonl"):
    """Wrap pipeline with logging."""
    start = time.time()
    answer, contexts, scores = pipeline_fn(query)
    latency = time.time() - start

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer_length": len(answer),
        "num_contexts": len(contexts),
        "avg_similarity": sum(scores) / len(scores) if scores else 0,
        "latency_ms": round(latency * 1000),
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return answer
```
