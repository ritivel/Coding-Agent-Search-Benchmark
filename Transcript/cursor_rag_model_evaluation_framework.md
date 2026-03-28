# RAG model evaluation framework
_Exported on 28/03/2026 at 21:35:27 GMT+5:30 from Cursor (2.6.20)_

---

**User**

I'm evaluating an idea, and for that I'll be conducting five experiments where I will be creating five rag models using five different approaches. Right now I want to set up an evaluation framework. From the hugging phase, I'll give you the information to connect the data, but at least connect the evaluation. Create the evaluation framework as a first step. Once that evaluation framework has been created and you tell me that the evaluation framework has been created, then we can go ahead with creating those five methodologies. I'll tell you more about how we'll create those five rag systems using the five methodologies.

Research Paper  - arxiv.org/abs/2407.11005
https://huggingface.co/docs/hub/datasets-usage 

---

**Cursor**

Let me start by understanding the research paper, the dataset, and your current workspace structure.

Let me get the full paper details for the TRACe evaluation framework and understand the dataset schema.

Now I have a thorough understanding of the RAGBench paper and its TRACe evaluation framework (uTilization, Relevance, Adherence, Completeness). Let me build the evaluation framework.

Now let me implement the main evaluator, report generator, base RAG model, and the CLI entrypoint.

Let me install the dependencies and run a smoke test.

Good, dependencies are installed. Now let me verify the framework imports correctly and run a quick structural test.

All imports and metrics compute correctly. Now let me run a real test against the actual RAGBench dataset from HuggingFace with a small sample.

The framework is fully working against the real RAGBench dataset from HuggingFace. Let me check for any linter issues across the files.

The evaluation framework is fully built and verified. Here's a summary of what was created:

---

## Evaluation Framework — Structure

```
Coding_Agent_Search_Benchmark/
├── config.py                         # Central config (dataset IDs, metrics, domains)
├── data_loader.py                    # Loads RAGBench from HuggingFace, normalises schema
├── requirements.txt                  # All dependencies
├── run_evaluation.py                 # CLI entry-point
├── evaluation/
│   ├── __init__.py
│   ├── trace_metrics.py              # TRACe framework (paper's core metrics)
│   ├── traditional_metrics.py        # ROUGE, BLEU, F1, EM, BERTScore
│   ├── evaluator.py                  # Orchestrates both metric suites
│   └── report.py                     # Comparison tables, rankings, CSV/JSON export
└── models/
    ├── __init__.py
    └── base_rag.py                   # Abstract base class for your 5 RAG models
```

## What the Framework Covers

**TRACe Metrics** (from the [RAGBench paper](https://arxiv.org/abs/2407.11005)):
- **Relevance** — fraction of retrieved context relevant to the query (evaluates the *retriever*)
- **Utilization** — fraction of context actually used by the generator (evaluates the *generator*)
- **Completeness** — fraction of relevant context that is utilized (Len(R∩U)/Len(R))
- **Adherence** — binary hallucination detection (is the response grounded in context?)

**Traditional Metrics**: ROUGE-1/2/L, BLEU, token-level F1, Exact Match, BERTScore

**Data**: All 12 RAGBench subsets across 5 domains (biomedical, general knowledge, legal, customer support, finance) — 95k+ examples total, loaded directly from `rungalileo/ragbench` on HuggingFace.

**Reporting**: Side-by-side comparison tables, per-subset breakdowns, composite ranking, CSV and JSON export — ready for comparing your 5 methodologies.

## How Your 5 RAG Models Will Plug In

Each methodology will subclass `BaseRAGModel` and implement `retrieve()` + `generate()`. The evaluator calls `model.run_batch(examples)` to get responses, then scores them against the framework. Example usage:

```python
from evaluation import RAGEvaluator, ComparisonReport

evaluator = RAGEvaluator()
report = ComparisonReport()

for model in [model_1, model_2, model_3, model_4, model_5]:
    responses = model.run_batch(examples)
    ev = evaluator.evaluate(examples, model_name=model.name, model_responses=responses)
    report.add(ev)

print(report.full_report())
report.to_json("results.json")
```

---

The evaluation framework is ready. Whenever you're ready, tell me about the five RAG methodologies and we'll build them out.

---

**User**

Just wanted to check: apart from the evaluation framework, there is no other methodology or any other part of the code which is written right. Give me a yes or no answer only.

---

**Cursor**

Yes — only the evaluation framework has been written. No RAG methodology implementations exist yet.

---

**User**

Move all the files related to evaluation in a single folder and make the relevant changes in code if required

---

**Cursor**

Let me first understand the current project structure.

Let me read the key files to understand the import dependencies before moving things around.

I have a clear picture of the current layout. The evaluation files are scattered across the root, `evaluation/`, and `models/` folders. I'll consolidate everything into a single `evaluation/` folder and fix all imports.

First, let me write the moved files with corrected imports into `evaluation/`.

Now let me write the updated `run_evaluation.py` inside `evaluation/`, update `__init__.py`, and move the `requirements.txt`.

Now let me delete the old files that have been moved.

Now let me verify the final structure and run a smoke test.

All evaluation files are now consolidated. Let me run the smoke test.

Now let me also verify the CLI entrypoint works from the project root.

Done. All evaluation files are now consolidated into a single `evaluation/` folder. Here's the final structure:

```
Coding_Agent_Search_Benchmark/
├── evaluation/
│   ├── __init__.py              # package exports
│   ├── config.py                # central configuration
│   ├── data_loader.py           # RAGBench HuggingFace loader
│   ├── trace_metrics.py         # TRACe metrics
│   ├── traditional_metrics.py   # ROUGE, BLEU, F1, EM, BERTScore
│   ├── evaluator.py             # main evaluation orchestrator
│   ├── report.py                # comparison report generator
│   ├── base_rag.py              # abstract base class for RAG models
│   ├── run_evaluation.py        # CLI entry-point
│   └── requirements.txt         # dependencies
└── experiment_0/                # (separate, untouched)
```

**What changed:**
- `config.py`, `data_loader.py`, `run_evaluation.py`, and `requirements.txt` moved from the project root into `evaluation/`
- `models/base_rag.py` moved into `evaluation/base_rag.py` and the now-empty `models/` directory was removed
- All imports updated to use relative imports (e.g. `from .config import ...` instead of `from config import ...`)
- `__init__.py` updated to export everything from the consolidated package
- CLI now runs via `python -m evaluation.run_evaluation` from the project root

