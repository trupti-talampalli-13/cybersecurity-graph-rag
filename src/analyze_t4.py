import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "azerg"


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


train = load_jsonl(DATA_DIR / "t4_train_clean.jsonl")
test = load_jsonl(DATA_DIR / "t4_test_clean.jsonl")


for name, records in [("TRAIN", train), ("TEST", test)]:

    counts = Counter(r["relation"] for r in records)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Total examples:", len(records))
    print("Unique relations:", len(counts))

    print("\nRelation distribution:")

    for relation, count in counts.most_common():
        percentage = count / len(records) * 100
        print(f"{relation:25s} {count:4d} ({percentage:6.2f}%)")