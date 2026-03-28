"""Central configuration for the RAG evaluation framework."""

RAGBENCH_HF_REPO = "rungalileo/ragbench"

RAGBENCH_SUBSETS = [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]

DOMAIN_MAP = {
    "covidqa": "biomedical",
    "pubmedqa": "biomedical",
    "hotpotqa": "general_knowledge",
    "msmarco": "general_knowledge",
    "hagrid": "general_knowledge",
    "expertqa": "general_knowledge",
    "cuad": "legal",
    "delucionqa": "customer_support",
    "emanual": "customer_support",
    "techqa": "customer_support",
    "finqa": "finance",
    "tatqa": "finance",
}

TRACE_METRICS = ["relevance", "utilization", "completeness", "adherence"]

TRADITIONAL_METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "f1", "exact_match", "bert_score"]

DEFAULT_SPLIT = "test"

DEFAULT_SAMPLE_SIZE = None  # None = use full split
