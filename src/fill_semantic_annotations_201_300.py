import json
from pathlib import Path

INPUT = Path("data/semantic_annotation_201_300.jsonl")
OUTPUT = Path("data/semantic_annotation_201_300_provisional.jsonl")


labels = {
    201: ("FALSE", "SEMANTIC_MISMATCH"),
    202: ("FALSE", "WRONG_SUBJECT"),
    203: ("UNCERTAIN", "OTHER"),
    204: ("FALSE", "COMPARISON"),
    205: ("FALSE", "SEMANTIC_MISMATCH"),
    206: ("TRUE", "VALID_RELATION"),
    207: ("FALSE", "SEMANTIC_MISMATCH"),
    208: ("FALSE", "SEMANTIC_MISMATCH"),
    209: ("FALSE", "TABLE"),
    210: ("FALSE", "TABLE"),
    211: ("FALSE", "SEMANTIC_MISMATCH"),
    212: ("FALSE", "COORDINATION_ERROR"),
    213: ("FALSE", "WRONG_SUBJECT"),
    214: ("FALSE", "WRONG_SUBJECT"),
    215: ("FALSE", "REPORTING_CONTEXT"),
    216: ("FALSE", "WRONG_SUBJECT"),
    217: ("FALSE", "REPORTING_CONTEXT"),
    218: ("FALSE", "TABLE"),
    219: ("FALSE", "WRONG_SUBJECT"),
    220: ("FALSE", "REPORTING_CONTEXT"),
    221: ("TRUE", "VALID_RELATION"),
    222: ("FALSE", "SEMANTIC_MISMATCH"),
    223: ("FALSE", "REPORTING_CONTEXT"),
    224: ("FALSE", "TABLE"),
    225: ("FALSE", "SEMANTIC_MISMATCH"),
    226: ("FALSE", "TABLE"),
    227: ("UNCERTAIN", "SEMANTIC_MISMATCH"),
    228: ("FALSE", "WRONG_SUBJECT"),
    229: ("FALSE", "WRONG_SUBJECT"),
    230: ("UNCERTAIN", "OTHER"),
    231: ("TRUE", "VALID_RELATION"),
    232: ("FALSE", "WRONG_SUBJECT"),
    233: ("FALSE", "REPORTING_CONTEXT"),
    234: ("FALSE", "WRONG_SUBJECT"),
    235: ("FALSE", "WRONG_SUBJECT"),
    236: ("FALSE", "WRONG_SUBJECT"),
    237: ("FALSE", "WRONG_DIRECTION"),
    238: ("FALSE", "SEMANTIC_MISMATCH"),
    239: ("UNCERTAIN", "REPORTING_CONTEXT"),
    240: ("FALSE", "WRONG_SUBJECT"),
    241: ("FALSE", "WRONG_SUBJECT"),
    242: ("UNCERTAIN", "OTHER"),
    243: ("FALSE", "TABLE"),
    244: ("FALSE", "WRONG_SUBJECT"),
    245: ("TRUE", "VALID_RELATION"),
    246: ("FALSE", "SEMANTIC_MISMATCH"),
    247: ("UNCERTAIN", "OTHER"),
    248: ("FALSE", "COMPARISON"),
    249: ("UNCERTAIN", "OTHER"),
    250: ("UNCERTAIN", "OTHER"),
    251: ("UNCERTAIN", "OTHER"),
    252: ("FALSE", "WRONG_SUBJECT"),
    253: ("FALSE", "WRONG_SUBJECT"),
    254: ("FALSE", "SEMANTIC_MISMATCH"),
    255: ("TRUE", "VALID_RELATION"),
    256: ("FALSE", "WRONG_SUBJECT"),
    257: ("FALSE", "WRONG_SUBJECT"),
    258: ("FALSE", "REPORTING_CONTEXT"),
    259: ("FALSE", "WRONG_SUBJECT"),
    260: ("FALSE", "WRONG_SUBJECT"),
    261: ("FALSE", "WRONG_DIRECTION"),
    262: ("FALSE", "TABLE"),
    263: ("FALSE", "WRONG_SUBJECT"),
    264: ("FALSE", "WRONG_SUBJECT"),
    265: ("FALSE", "SEMANTIC_MISMATCH"),
    266: ("TRUE", "VALID_RELATION"),
    267: ("FALSE", "WRONG_SUBJECT"),
    268: ("FALSE", "WRONG_SUBJECT"),
    269: ("FALSE", "SEMANTIC_MISMATCH"),
    270: ("FALSE", "SEMANTIC_MISMATCH"),
    271: ("FALSE", "TABLE"),
    272: ("FALSE", "COORDINATION_ERROR"),
    273: ("FALSE", "WRONG_SUBJECT"),
    274: ("FALSE", "WRONG_SUBJECT"),
    275: ("FALSE", "WRONG_DIRECTION"),
    276: ("FALSE", "WRONG_SUBJECT"),
    277: ("FALSE", "WRONG_SUBJECT"),
    278: ("FALSE", "WRONG_OBJECT"),
    279: ("UNCERTAIN", "OTHER"),
    280: ("FALSE", "COORDINATION_ERROR"),
    281: ("FALSE", "SEMANTIC_MISMATCH"),
    282: ("FALSE", "WRONG_DIRECTION"),
    283: ("FALSE", "TABLE"),
    284: ("FALSE", "WRONG_SUBJECT"),
    285: ("FALSE", "REPORTING_CONTEXT"),
    286: ("UNCERTAIN", "OTHER"),
    287: ("FALSE", "SEMANTIC_MISMATCH"),
    288: ("FALSE", "WRONG_DIRECTION"),
    289: ("UNCERTAIN", "OTHER"),
    290: ("FALSE", "WRONG_SUBJECT"),
    291: ("FALSE", "COORDINATION_ERROR"),
    292: ("FALSE", "WRONG_SUBJECT"),
    293: ("FALSE", "SEMANTIC_MISMATCH"),
    294: ("FALSE", "REPORTING_CONTEXT"),
    295: ("FALSE", "WRONG_SUBJECT"),
    296: ("TRUE", "VALID_RELATION"),
    297: ("FALSE", "TABLE"),
    298: ("FALSE", "WRONG_SUBJECT"),
    299: ("FALSE", "WRONG_SUBJECT"),
    300: ("TRUE", "VALID_RELATION"),
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    records = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 100:
        raise ValueError(
            f"Expected 100 records, but found {len(records)}"
        )

    seen_ids = set()

    for record in records:
        gold_id = int(record["gold_id"])

        if gold_id in seen_ids:
            raise ValueError(f"Duplicate gold_id: {gold_id}")

        seen_ids.add(gold_id)

        if gold_id not in labels:
            raise ValueError(f"No label defined for gold_id {gold_id}")

        human_label, error_category = labels[gold_id]

        record["human_label"] = human_label
        record["error_category"] = error_category
        record["annotator_notes"] = (
            "Provisional assistant-assisted annotation; "
            "requires independent human verification before use as final gold."
        )

    records.sort(key=lambda x: int(x["gold_id"]))

    with OUTPUT.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    from collections import Counter

    label_counts = Counter(r["human_label"] for r in records)
    category_counts = Counter(r["error_category"] for r in records)

    print("\nCompleted provisional annotation")
    print(f"Input  : {INPUT}")
    print(f"Output : {OUTPUT}")
    print(f"Records: {len(records)}")

    print("\nHuman labels:")
    for label, count in label_counts.items():
        print(f"  {label:10s}: {count}")

    print("\nError categories:")
    for category, count in category_counts.most_common():
        print(f"  {category:25s}: {count}")


if __name__ == "__main__":
    main()