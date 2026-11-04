# Person 2 — Retrieval + Graph-Augmented RAG

This folder contains the Person 2 retrieval contribution for the AnnoCTR CTI project.

## Included components

- `StandardRAGRetriever`: dense sentence-embedding retrieval.
- `HybridRAGRetriever`: BM25 + dense retrieval combined with Reciprocal Rank Fusion.
- `GraphRAGRetriever`: entity matching and assertion retrieval from `assertions.jsonl`.
- `EvidenceFusion`: deduplicates document chunks and keeps graph assertions separate.
- `ContextBuilder`: builds evidence context with source, date, report/segment or assertion identifiers.
- `AnswerGenerator`: transparent structured evidence formatter. It does **not** call an LLM.
- `GraphRAGPipeline`: orchestrates retrieval, fusion, context construction and answer formatting.
- `evaluation.py`: Recall@K, MRR and retrieval latency evaluation for document retrievers.

## Required existing project data

Run the team's knowledge-graph pipeline first. The following files must exist:

- `data/processed_segments.jsonl`
- `data/cti_knowledge_graph_trusted.json`
- `data/assertions.jsonl`

The graph retriever uses the graph node `id`/`name` fields and assertion `subject_id`, `object_id`, `assertion_id`, `claim_text`, `evidence_sentence`, `source_vendor`, `publication_date`, `trust_level`, `confidence`, `verification_status`, and `human_verified` fields. Optional fields are handled where possible.

## Install in Google Colab

From the repository root:

```python
!pip install -q sentence-transformers rank-bm25 numpy
```

Copy the files in this package's `src/` folder into the matching repository `src/` folder. Do not replace teammates' preprocessing, graph-construction, or verification modules.

## Run

```python
from src.graph_rag_pipeline import GraphRAGPipeline

rag = GraphRAGPipeline()
result = rag.ask("How was TrickBot related to Emotet?", top_k=5)

print(result["answer"])
print(result["context"])
```

Dense embeddings are generated once when the pipeline is initialized. Reuse the same `rag` object for subsequent questions; do not reinitialize it for every query.

## Test graph pair retrieval independently

```python
from src.graph_retrieval import GraphRAGRetriever

graph = GraphRAGRetriever()
matches = graph.match_entities("How was TrickBot related to Emotet?")
print("Matched entities:", matches)

results = graph.retrieve("How was TrickBot related to Emotet?", top_k=10)
for item in results:
    print(item.get("claim_text"), item.get("source_vendor"), item.get("evidence_sentence"))
```

When two or more entities are matched, graph retrieval returns only assertions whose subject and object IDs are both among the matched entity IDs. If no direct assertion exists for that pair, the result is empty rather than filling the list with unrelated facts. This is deliberate and avoids false relevance.

## Evaluation

Create `data/retrieval_qrels.csv` with manually checked relevance labels:

```csv
query,relevant_ids
"How was TrickBot related to Emotet?","REPORT_ID::SEGMENT_ID;REPORT_ID_2::SEGMENT_ID_2"
```

Then run:

```python
from src.evaluation import load_qrels, evaluate_retrievers
qrels = load_qrels("data/retrieval_qrels.csv")

rows = evaluate_retrievers(
    {"dense": rag.retriever.dense, "hybrid": rag.retriever.hybrid},
    qrels,
    top_k=5,
    output_path="results/retrieval_metrics.csv",
)
print(rows)
```

**Important:** Do not invent relevance labels. The evaluation file must be annotated by a human. Graph assertion retrieval is not compared with segment-level qrels in this helper because it returns assertion IDs, not segment IDs. A separate assertion-level gold set is needed for fair Graph-RAG retrieval evaluation.

## Research integrity notes

- The current `AnswerGenerator` formats retrieved evidence; it is not an LLM answer generator.
- `trusted` is not the same as human verified.
- Automatically extracted assertions may be incorrect or only partly supported by their evidence. Always show evidence, source and date.
- Candidate assertions should be presented cautiously.
- Dense/hybrid and graph scores are not directly comparable.
- The simple evidence-fusion ordering is a prototype heuristic, not a calibrated relevance model.
- AnnoCTR is a static CTI corpus; results should not be described as live threat intelligence.

## Run basic unit test

```python
!python -m unittest tests/test_retrieval_helpers.py
```
