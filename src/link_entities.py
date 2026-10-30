import json
import re
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENTITIES_FILE = BASE_DIR / "data" / "entities.jsonl"

LINKING_DIR = (
    BASE_DIR.parent
    / "anno-ctr-lrec-coling-2024"
    / "AnnoCTR"
    / "linking"
)

LINKING_FILES = [LINKING_DIR / f"{split}.jsonl" for split in ("train", "dev", "test")]

OUTPUT_FILE = (
    BASE_DIR / "data" / "linked_entities_candidates.jsonl"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """Conservative formatting-only normalization."""

    text = text.lower().strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize spaces around &
    text = re.sub(r"\s*&\s*", " & ", text)

    # Normalize hyphens
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    return text


# ============================================================
# LOAD LINKING DATA
# ============================================================

print("=" * 60)
print("BUILDING CONTEXT-PRESERVING ENTITY LINKS")
print("=" * 60)

print("\nLoading linking annotations:")
print(*LINKING_FILES, sep=", ")

linking_candidates = defaultdict(list)

linking_records = 0


for LINKING_FILE in LINKING_FILES:

    with open(LINKING_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            linking_records += 1

            document = record.get("document")
            mention = record.get("mention")

            if not document or not mention:
                continue

            key = (
                document,
                normalize(mention)
            )

            candidate = {
                "id": record.get("label_id"),
                "name": record.get("label_title"),
                "link": record.get("label_link"),
                "entity_class": record.get("entity_class"),
                "entity_type": record.get("entity_type")
            }

            # ----------------------------------------------------
            # Deduplicate identical candidates
            # ----------------------------------------------------

            candidate_key = (
                candidate["id"],
                candidate["name"],
                candidate["link"],
                candidate["entity_class"],
                candidate["entity_type"]
            )

            exists = False

            for existing in linking_candidates[key]:

                existing_key = (
                    existing["id"],
                    existing["name"],
                    existing["link"],
                    existing["entity_class"],
                    existing["entity_type"]
                )

                if existing_key == candidate_key:
                    exists = True
                    break

            if not exists:
                linking_candidates[key].append(candidate)

print(f"Linking records loaded : {linking_records}")
print(
    f"Unique document+mention keys : "
    f"{len(linking_candidates)}"
)


# ============================================================
# PROCESS ENTITIES
# ============================================================

total_mentions = 0
linked_mentions = 0
unlinked_mentions = 0
ambiguous_mentions = 0

candidate_distribution = defaultdict(int)

output_records = []


with open(ENTITIES_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)

        # ----------------------------------------------------
        # YOUR ACTUAL ENTITIES.JSONL SCHEMA
        # ----------------------------------------------------

        document = record.get("document")
        segment = record.get("segment")
        mention = record.get("entity_text")
        ner_type = record.get("entity_type")

        if not mention:
            continue

        total_mentions += 1

        normalized_mention = normalize(mention)

        key = (
            document,
            normalized_mention
        )

        candidates = linking_candidates.get(
            key,
            []
        )

        # ----------------------------------------------------
        # Linking status
        # ----------------------------------------------------

        # CVE ids have no AnnoCTR gold link; the id itself is the canonical identifier
        if not candidates and ner_type == "CVE":
            cve_id = re.sub(r"\s+", "", mention).replace("‑", "-").replace("–", "-").upper()
            candidates = [{
                "id": cve_id,
                "name": cve_id,
                "link": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                "entity_class": "NE",
                "entity_type": "CVE",
            }]

        linked = len(candidates) > 0

        if linked:
            linked_mentions += 1
        else:
            unlinked_mentions += 1

        # ----------------------------------------------------
        # Conflict detection
        # ----------------------------------------------------

        distinct_ids = {
            candidate["id"]
            for candidate in candidates
        }

        conflict = len(distinct_ids) > 1

        if conflict:
            ambiguous_mentions += 1

        candidate_distribution[
            len(candidates)
        ] += 1

        # ----------------------------------------------------
        # Build output
        # ----------------------------------------------------

        output_record = {
            "document": document,
            "segment": segment,
            "source": record.get("source"),
            "publication_date": record.get("publication_date"),

            "mention": mention,
            "normalized_mention": normalized_mention,

            # Original AnnoCTR NER type
            "ner_type": ner_type,

            "linked": linked,
            "conflict": conflict,
            "candidate_count": len(candidates),

            # Preserve ALL canonical candidates
            "canonical_candidates": candidates
        }

        output_records.append(output_record)


# ============================================================
# WRITE OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in output_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LINKING RESULTS")
print("=" * 60)

print(
    f"Total entity mentions       : "
    f"{total_mentions}"
)

print(
    f"Linked mentions             : "
    f"{linked_mentions}"
)

print(
    f"Unlinked mentions           : "
    f"{unlinked_mentions}"
)

print(
    f"Ambiguous/conflicting       : "
    f"{ambiguous_mentions}"
)


if total_mentions > 0:

    linking_rate = (
        linked_mentions /
        total_mentions
    ) * 100

    ambiguity_rate = (
        ambiguous_mentions /
        total_mentions
    ) * 100

    print(
        f"Linking rate                : "
        f"{linking_rate:.2f}%"
    )

    print(
        f"Ambiguity rate              : "
        f"{ambiguity_rate:.2f}%"
    )


# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

print("\nCandidate-count distribution:")

for count in sorted(candidate_distribution):

    print(
        f"  {count} candidate(s): "
        f"{candidate_distribution[count]}"
    )


# ============================================================
# OUTPUT
# ============================================================

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")