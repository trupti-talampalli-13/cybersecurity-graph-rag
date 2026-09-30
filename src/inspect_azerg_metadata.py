import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "azerg"

with open(DATA_DIR / "azerg_T4_train.json", "r", encoding="utf-8") as f:
    train = json.load(f)

with open(DATA_DIR / "azerg_T4_test.json", "r", encoding="utf-8") as f:
    test = json.load(f)

print("TRAIN FIRST RECORD KEYS:")
print(train[0].keys())

print("\nTEST FIRST RECORD KEYS:")
print(test[0].keys())

print("\nTRAIN RECORD:")
print(json.dumps(train[0], indent=2, ensure_ascii=False))

print("\n" + "=" * 80)

print("Fields available in every training record:")

all_keys = set(train[0].keys())

for item in train:
    all_keys &= set(item.keys())

print(all_keys)