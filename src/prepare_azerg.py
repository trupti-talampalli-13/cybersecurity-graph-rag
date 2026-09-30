import json
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "azerg"


def parse_t4_item(item):
    text = item["input"]

    source_match = re.search(
        r"### Source Entity:\s*(.*?)\s*### Target Entity:",
        text,
        re.DOTALL,
    )

    target_match = re.search(
        r"### Target Entity:\s*(.*?)\s*### Possible Relationship Labels:",
        text,
        re.DOTALL,
    )

    passage_match = re.search(
        r"### Text Passage:\s*(.*)",
        text,
        re.DOTALL,
    )

    relation_match = re.search(
        r"<label>(.*?)</label>",
        item["output"],
        re.DOTALL,
    )

    if not all([source_match, target_match, passage_match, relation_match]):
        return None

    return {
        "source": source_match.group(1).strip(),
        "target": target_match.group(1).strip(),
        "text": passage_match.group(1).strip(),
        "relation": relation_match.group(1).strip(),
    }


def process_file(filename):
    input_path = DATA_DIR / filename

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for item in data:
        parsed = parse_t4_item(item)

        if parsed:
            records.append(parsed)

    return records


train = process_file("azerg_T4_train.json")
test = process_file("azerg_T4_test.json")


train_output = DATA_DIR / "t4_train_clean.jsonl"
test_output = DATA_DIR / "t4_test_clean.jsonl"


with open(train_output, "w", encoding="utf-8") as f:
    for record in train:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


with open(test_output, "w", encoding="utf-8") as f:
    for record in test:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


print("AZERG T4 preprocessing complete.")
print()
print(f"Train records: {len(train)}")
print(f"Test records:  {len(test)}")
print()
print(f"Train output: {train_output}")
print(f"Test output:  {test_output}")

if train:
    print()
    print("Example cleaned record:")
    print(json.dumps(train[0], indent=2, ensure_ascii=False))