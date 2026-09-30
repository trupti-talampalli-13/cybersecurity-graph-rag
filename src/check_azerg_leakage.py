import json
from pathlib import Path


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


train_texts = set(record["text"].strip() for record in train)
test_texts = set(record["text"].strip() for record in test)

exact_overlap = train_texts & test_texts


print("Train records:", len(train))
print("Test records:", len(test))
print("Unique train passages:", len(train_texts))
print("Unique test passages:", len(test_texts))
print("Exact overlapping passages:", len(exact_overlap))

if exact_overlap:
    print("\nExamples of overlapping passages:\n")

    for text in list(exact_overlap)[:5]:
        print("-" * 80)
        print(text)