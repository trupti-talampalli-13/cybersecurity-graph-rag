import json
from pathlib import Path

NER_FILE = Path(
    "../anno-ctr-lrec-coling-2024/AnnoCTR/ner_json/train.json"
)

OUTPUT_FILE = Path(
    "data/train_segments.jsonl"
)


def extract_entities(tokens, tags):
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, tags):

        if tag.startswith("B-"):

            # Finish previous entity
            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_type
                })

            current_tokens = [token]
            current_type = tag[2:]

        elif tag.startswith("I-"):

            if current_tokens:
                current_tokens.append(token)

        else:

            if current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "type": current_type
                })

                current_tokens = []
                current_type = None

    # Finish final entity
    if current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "type": current_type
        })

    return entities


def main():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    entity_count = 0

    with open(NER_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:

            record = json.loads(line)

            doc_id = record["id"]

            # Example:
            # proofpoint_...__s0000
            if "__s" in doc_id:
                document, segment = doc_id.rsplit("__s", 1)
                segment = int(segment)
            else:
                document = doc_id
                segment = None

            entities = extract_entities(
                record["tokens"],
                record["all_tags"]
            )

            output = {
                "document": document,
                "segment": segment,
                "text": record["text"],
                "entities": entities
            }

            outfile.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )

            count += 1
            entity_count += len(entities)

    print(f"Segments processed: {count}")
    print(f"Entities extracted: {entity_count}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()