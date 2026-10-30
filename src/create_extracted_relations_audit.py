"""Create a blank review worksheet for every relation produced by extract_relations.py.

Same label scheme as the V2 `SUPPORTED` audit worksheet (TRUE / FALSE / UNCERTAIN) so the two
audits can be reported together. No labels are generated; reviewers fill in the last five columns.

Reads:  data/relations.jsonl, data/cti_knowledge_graph.json
Writes: data/extracted_relations_audit_template.csv

Run from the cti_graph_rag directory:  python src/create_extracted_relations_audit.py
"""
import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ORDER = {"EXPLOITS": 0, "USES_TECHNIQUE": 1, "USES": 2, "TARGETS": 3}
COLUMNS = ["audit_id", "document", "segment", "source", "publication_date", "relation", "head", "head_mention",
           "head_type", "tail", "tail_mention", "tail_type", "trigger", "direction", "confidence",
           "evidence_text", "human_label", "error_category", "rationale", "reviewer", "reviewed_at"]


def main():
    graph = json.load(open(DATA / "cti_knowledge_graph.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    relations = [json.loads(l) for l in open(DATA / "relations.jsonl", encoding="utf-8")]
    relations.sort(key=lambda r: (ORDER[r["relation"]], r["document"], r["segment"], r["head"], r["tail"]))
    with open(DATA / "extracted_relations_audit_template.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for i, r in enumerate(relations, 1):
            w.writerow({
                "audit_id": f"REL-{i:03d}", "document": r["document"], "segment": r["segment"],
                "source": r["source"], "publication_date": r["publication_date"], "relation": r["relation"],
                "head": nodes[r["head"]]["name"], "head_mention": r["head_mention"],
                "head_type": nodes[r["head"]]["entity_type"], "tail": nodes[r["tail"]]["name"],
                "tail_mention": r["tail_mention"], "tail_type": nodes[r["tail"]]["entity_type"],
                "trigger": r["trigger"], "direction": r["direction"], "confidence": r["confidence"],
                "evidence_text": r["evidence_text"],
            })
    print(f"Wrote {len(relations)} rows to data/extracted_relations_audit_template.csv")


if __name__ == "__main__":
    main()
