"""
Retrieval evaluation.

Expected qrels CSV columns:
query, relevant_ids

relevant_ids is a semicolon-separated list of relevant segment IDs, or
report_id::segment_id identifiers. Only manually checked relevance labels
should be used for scientific results.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path


def load_qrels(path):
    qrels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qrels[row["query"]] = {
                x.strip() for x in row["relevant_ids"].split(";") if x.strip()
            }
    return qrels


def _doc_key(item):
    rid, sid = item.get("report_id"), item.get("segment_id")
    return f"{rid}::{sid}" if rid is not None and sid is not None else str(sid)


def retrieval_metrics(results, relevant_ids, k_values=(1, 3, 5)):
    ranked = [_doc_key(x) for x in results]
    relevant = set(relevant_ids)
    out = {}
    for k in k_values:
        top = ranked[:k]
        out[f"recall@{k}"] = len(set(top) & relevant) / len(relevant) if relevant else 0.0
    first_rank = next((i + 1 for i, key in enumerate(ranked) if key in relevant), None)
    out["mrr"] = 1.0 / first_rank if first_rank else 0.0
    out["relevant_found"] = len(set(ranked) & relevant)
    out["relevant_total"] = len(relevant)
    return out


def evaluate_retrievers(retrievers, qrels, top_k=5, output_path=None):
    rows = []
    for query, relevant_ids in qrels.items():
        for name, retriever in retrievers.items():
            start = time.perf_counter()
            results = retriever.retrieve(query, top_k=top_k)
            elapsed = time.perf_counter() - start
            # Graph results are assertions, not segment-level qrels; skip them here.
            if name == "graph":
                continue
            metrics = retrieval_metrics(results, relevant_ids)
            rows.append({
                "query": query, "method": name, "latency_seconds": round(elapsed, 4),
                **metrics,
            })
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return rows


def save_results_json(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
