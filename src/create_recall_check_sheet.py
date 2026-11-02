"""Create the sheet for a recall check of the relation extractor.

Recall = of the relations a sentence really states, how many does the extractor find? To measure it,
a person lists the relations in a random sample of sentences WITHOUT seeing the extractor's output.

Sentences are drawn uniformly at random from the gold-tagged reports among those with at least one actor,
malware or tool entity and at least two distinct relevant entities. The draw does not look at what the
extractor found, so it is not biased toward sentences the extractor already handles.

Writes: data/recall_check_sheet.csv   (annotation columns blank)

Run from the cti_graph_rag directory:  python src/create_recall_check_sheet.py
"""
import csv
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SEED, N_SENTENCES = 21, 30
HEAD_TYPES = {"GROUP", "MALWARE", "TOOL"}
RELEVANT = {"GROUP", "MALWARE", "TOOL", "TECHNIQUE", "TACTIC", "CVE", "ORG", "SECTOR", "LOC"}


def relevant_entities(seg):
    seen, out = set(), []
    for e in sorted(seg["entities"], key=lambda e: e["char_start"] if e["char_start"] is not None else -1):
        if e["type"] in RELEVANT and e["text"].lower() not in seen:
            seen.add(e["text"].lower())
            out.append(e)
    return out


def main():
    segments = [json.loads(line) for line in open(DATA / "processed_segments.jsonl", encoding="utf-8")]
    pool = []
    for s in segments:
        ents = relevant_entities(s)
        if len(ents) >= 2 and any(e["type"] in HEAD_TYPES for e in ents):
            pool.append((s, ents))
    sample = random.Random(SEED).sample(pool, N_SENTENCES)
    cols = ["sentence_id", "report_id", "segment_id", "sentence", "entities", "gold_relations", "notes"]
    with open(DATA / "recall_check_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for i, (s, ents) in enumerate(sample, 1):
            w.writerow({"sentence_id": f"REC-{i:02d}", "report_id": s["report_id"], "segment_id": s["segment_id"],
                        "sentence": s["clean_text"],
                        "entities": " ; ".join(f"{e['text']} [{e['type']}]" for e in ents),
                        "gold_relations": "", "notes": ""})
    print(f"Pool of {len(pool)} candidate sentences; wrote {N_SENTENCES} to data/recall_check_sheet.csv")


if __name__ == "__main__":
    main()
