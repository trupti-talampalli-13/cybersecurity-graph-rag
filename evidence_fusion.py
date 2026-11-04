"""Deduplicates document evidence and preserves graph assertion provenance."""
class EvidenceFusion:
    def fuse(self, all_results, top_k=10):
        fused = {}

        for method in ("dense", "hybrid"):
            for item in all_results.get(method, []):
                key = ("text", item.get("report_id"), item.get("segment_id"))
                if key not in fused:
                    fused[key] = {
                        "type": "text",
                        "report_id": item.get("report_id"),
                        "segment_id": item.get("segment_id"),
                        "text": item.get("text", ""),
                        "source": item.get("source"),
                        "publication_date": item.get("publication_date"),
                        "retrieval_methods": set(),
                        "scores": {},
                    }
                fused[key]["retrieval_methods"].add(method)
                for score_name in ("score", "rrf_score", "dense_score", "bm25_score"):
                    if score_name in item:
                        fused[key]["scores"][score_name] = item[score_name]

        for item in all_results.get("graph", []):
            key = ("graph", item.get("assertion_id"))
            fused[key] = {
                "type": "graph",
                "assertion_id": item.get("assertion_id"),
                "claim": item.get("claim_text"),
                "evidence": item.get("evidence_sentence") or item.get("evidence_text"),
                "source": item.get("source_vendor") or item.get("source"),
                "publication_date": item.get("publication_date"),
                "trust_level": item.get("trust_level", "unknown"),
                "confidence": item.get("confidence", "unknown"),
                "verification_status": item.get("verification_status", "unknown"),
                "human_verified": bool(item.get("human_verified", False)),
                "subject_id": item.get("subject_id"),
                "object_id": item.get("object_id"),
                "predicate": item.get("predicate"),
                "retrieval_methods": {"graph"},
                "scores": {},
            }

        values = list(fused.values())
        # Method overlap is a useful tie-breaker, not a calibrated relevance score.
        values.sort(
            key=lambda x: (
                len(x["retrieval_methods"]),
                x.get("trust_level") == "trusted",
                x.get("confidence") == "high",
            ),
            reverse=True,
        )
        for item in values:
            item["retrieval_methods"] = sorted(item["retrieval_methods"])
        return values[:top_k]
