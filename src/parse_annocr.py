from pathlib import Path


# Path to the AnnoCTR BIO file
BIO_FILE = Path(
    "../anno-ctr-lrec-coling-2024/AnnoCTR/ner_bio/train.bio"
)


def extract_entities(bio_file):
    entities = []
    current_tokens = []
    current_type = None

    with open(bio_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Empty line = sentence boundary
            if not line:
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_type
                    })
                    current_tokens = []
                    current_type = None
                continue

            # Document marker
            if line.startswith("<DOCSTART>"):
                continue

            parts = line.split()

            # Token + 6 annotation columns
            if len(parts) < 7:
                continue

            token = parts[0]
            label = parts[6]   # Column 6

            if label.startswith("B-"):
                # Save previous entity
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_type
                    })

                current_tokens = [token]
                current_type = label[2:]

            elif label.startswith("I-"):
                if current_tokens:
                    current_tokens.append(token)

            else:  # O
                if current_tokens:
                    entities.append({
                        "text": " ".join(current_tokens),
                        "type": current_type
                    })
                    current_tokens = []
                    current_type = None

        # Save final entity
        if current_tokens:
            entities.append({
                "text": " ".join(current_tokens),
                "type": current_type
            })

    return entities


if __name__ == "__main__":
    entities = extract_entities(BIO_FILE)

    print(f"Total entities extracted: {len(entities)}")
    print("\nFirst 30 entities:\n")

    for entity in entities[:30]:
        print(f"{entity['type']:15} -> {entity['text']}")