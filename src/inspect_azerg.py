from datasets import load_dataset

dataset = load_dataset("QCRI/AZERG-Dataset")

print(dataset)
print()

for split in dataset:
    print("SPLIT:", split)
    print("Number of records:", len(dataset[split]))
    print("Columns:", dataset[split].column_names)
    print()
    
    if len(dataset[split]) > 0:
        print("FIRST RECORD:")
        print(dataset[split][0])
        print("-" * 80)