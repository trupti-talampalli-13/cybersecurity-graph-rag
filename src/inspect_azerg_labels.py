import requests
import re
import collections

base = "https://huggingface.co/datasets/QCRI/AZERG-Dataset/resolve/main/"

url = base + "train/azerg_T4_train.json"

response = requests.get(url, timeout=30)
response.raise_for_status()

t4 = response.json()

labels = []
possible = []

for item in t4:
    match = re.search(r"<label>(.*?)</label>", item["output"])
    if match:
        labels.append(match.group(1))

    before_text = item["input"].split("### Text Passage:")[0]
    possible.extend(re.findall(r"'([^']+)'", before_text))


print("Unique OUTPUT labels:", len(set(labels)))

print("\nOUTPUT LABEL COUNTS:")
for label, count in collections.Counter(labels).most_common():
    print(f"{label}: {count}")

print("\n" + "=" * 80)

print("Unique POSSIBLE labels:", len(set(possible)))

print("\nPOSSIBLE LABELS:")
for label in sorted(set(possible)):
    print(label)

print("\n" + "=" * 80)

missing = sorted(set(labels) - set(possible))

print("Outputs NOT appearing in possible-label lists:")
print(missing)