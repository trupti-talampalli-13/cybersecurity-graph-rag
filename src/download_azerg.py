import requests
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/QCRI/AZERG-Dataset/resolve/main/"

FILES = [
    "train/azerg_T3_train.json",
    "train/azerg_T4_train.json",
    "test/azerg_T3_test.json",
    "test/azerg_T4_test.json",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "azerg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for file_path in FILES:
    url = BASE_URL + file_path
    output_path = OUTPUT_DIR / Path(file_path).name

    print(f"Downloading: {file_path}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved: {output_path}")
    print(f"Size: {len(response.content):,} bytes")
    print()


print("AZERG download complete.")