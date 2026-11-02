"""Score the extractor's recall against the relations a person wrote down.

Reads the filled sheet (default data/recall_check_human.csv). In `gold_relations`, relations are written as
`head | RELATION | tail`, separated by `;`, using entity names exactly as listed in the `entities` column;
write NONE when the sentence states no relation. The extractor is run on the same sentences (gold entity tags)
and its relations are matched on relation type, head text and tail text.

Writes: data/recall_check_summary.json and prints recall and precision on these sentences.

Run from the cti_graph_rag directory:  python src/score_recall_check.py [path to filled sheet]
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_relations import find_relations, norm  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
RELATIONS = {"USES", "USES_TECHNIQUE", "EXPLOITS", "TARGETS"}


def parse_gold(cell):
    out, problems = set(), []
    for part in cell.replace("\n", ";").split(";"):
        part = part.strip()
        if not part or part.upper() == "NONE":
            continue
        bits = [b.strip() for b in part.split("|")]
        if len(bits) != 3 or bits[1].upper() not in RELATIONS:
            problems.append(part)
            continue
        out.add((bits[1].upper(), norm(bits[0]), norm(bits[2])))
    return out, problems


def main():
    sheet = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "recall_check_human.csv"
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    segments = {}
    for line in open(DATA / "processed_segments.jsonl", encoding="utf-8"):
        s = json.loads(line)
        segments[(s["report_id"], s["segment_id"])] = s

    gold_total = found = extracted_total = extracted_ok = 0
    by_rel = defaultdict(lambda: [0, 0])
    missed, wrong, bad_cells, unfilled = [], [], [], []
    for r in rows:
        if not r["gold_relations"].strip():
            unfilled.append(r["sentence_id"])
            continue
        gold, problems = parse_gold(r["gold_relations"])
        bad_cells += [(r["sentence_id"], p) for p in problems]
        seg = segments[(r["report_id"], int(r["segment_id"]))]
        got = {(rel, norm(h["text"]), norm(t["text"])) for rel, h, t, *_ in find_relations(seg, {})}
        gold_total += len(gold)
        found += len(gold & got)
        extracted_total += len(got)
        extracted_ok += len(got & gold)
        for g in gold:
            by_rel[g[0]][0] += 1
            by_rel[g[0]][1] += g in got
        missed += [(r["sentence_id"], *g) for g in sorted(gold - got)]
        wrong += [(r["sentence_id"], *g) for g in sorted(got - gold)]

    summary = {
        "sentences_scored": len(rows) - len(unfilled), "sentences_unfilled": unfilled,
        "gold_relations": gold_total, "found_by_extractor": found,
        "recall": round(found / gold_total, 3) if gold_total else None,
        "extractor_relations_on_these_sentences": extracted_total, "of_which_in_gold": extracted_ok,
        "precision_vs_gold": round(extracted_ok / extracted_total, 3) if extracted_total else None,
        "recall_by_relation": {k: {"gold": v[0], "found": v[1]} for k, v in by_rel.items()},
        "missed": missed, "extracted_not_in_gold": wrong, "unparsed_cells": bad_cells,
        "note": "single reviewer; relation match on type, head text and tail text; gold-tagged entities",
    }
    (DATA / "recall_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Sentences scored: {summary['sentences_scored']} (unfilled: {len(unfilled)})")
    print(f"Gold relations: {gold_total}; found by extractor: {found}; recall {summary['recall']}")
    print(f"Extractor relations on these sentences: {extracted_total}; in gold: {extracted_ok}; "
          f"precision {summary['precision_vs_gold']}")
    for item in bad_cells:
        print("could not parse:", item)


if __name__ == "__main__":
    main()
