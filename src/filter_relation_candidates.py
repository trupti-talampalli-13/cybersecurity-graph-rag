import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "annoctr_relation_candidates.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "annoctr_relation_candidates_filtered.jsonl"
)


# ---------------------------------------------------------
# Entity-type compatibility
#
# This is deliberately conservative.
# We are NOT assigning relations here.
# We are only removing pairs that are extremely unlikely
# to represent one of the AZERG relations.
# ---------------------------------------------------------

COMPATIBLE_TYPES = {
    "attributed-to": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "ORG",
            "TECHNIQUE", "CON"
        },
        "target": {
            "GROUP", "ORG", "CON"
        },
    },

    "authored-by": {
        "source": {
            "MALWARE", "TOOL", "CON"
        },
        "target": {
            "GROUP", "ORG"
        },
    },

    "communicates-with": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "ORG"
        },
        "target": {
            "GROUP", "MALWARE", "TOOL", "ORG"
        },
    },

    "delivers": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "MALWARE", "TOOL", "CON"
        },
    },

    "drops": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "MALWARE", "TOOL", "CON"
        },
    },

    "exfiltrates-to": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "ORG", "LOC", "SECTOR", "CON"
        },
    },

    "indicates": {
        "source": {
            "MALWARE", "TOOL", "TECHNIQUE",
            "TACTIC", "CON", "GROUP"
        },
        "target": {
            "MALWARE", "TOOL", "TECHNIQUE",
            "TACTIC", "CON", "GROUP", "ORG"
        },
    },

    "located-at": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "ORG",
            "CON", "SECTOR"
        },
        "target": {
            "LOC", "ORG", "SECTOR"
        },
    },

    "owns": {
        "source": {
            "GROUP", "ORG"
        },
        "target": {
            "MALWARE", "TOOL", "CON", "ORG"
        },
    },

    "targets": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "ORG", "SECTOR", "LOC", "CON",
            "GROUP"
        },
    },

    "uses": {
        "source": {
            "GROUP", "MALWARE", "TOOL", "ORG"
        },
        "target": {
            "MALWARE", "TOOL", "TECHNIQUE",
            "TACTIC", "CON"
        },
    },

    "variant-of": {
        "source": {
            "MALWARE", "TOOL", "GROUP", "CON"
        },
        "target": {
            "MALWARE", "TOOL", "GROUP", "CON"
        },
    },
}


# ---------------------------------------------------------
# Build union of all compatible source-target pairs
# ---------------------------------------------------------

ALLOWED_PAIRS = set()

for relation_info in COMPATIBLE_TYPES.values():

    for source_type in relation_info["source"]:

        for target_type in relation_info["target"]:

            ALLOWED_PAIRS.add(
                (source_type, target_type)
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AnnoCTR Relation Candidate Filtering")
    print("=" * 60)

    total = 0
    kept = 0
    removed = 0

    original_pairs = Counter()
    kept_pairs = Counter()
    removed_pairs = Counter()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as fin, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            record = json.loads(line)

            total += 1

            pair = (
                record["source_type"],
                record["target_type"]
            )

            original_pairs[pair] += 1

            if pair in ALLOWED_PAIRS:

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                kept += 1
                kept_pairs[pair] += 1

            else:

                removed += 1
                removed_pairs[pair] += 1

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print(f"\nTotal candidate pairs : {total}")
    print(f"Kept                   : {kept}")
    print(f"Removed                 : {removed}")

    if total > 0:

        print(
            f"Retention rate         : "
            f"{kept / total * 100:.2f}%"
        )

        print(
            f"Removal rate           : "
            f"{removed / total * 100:.2f}%"
        )

    print("\nTop retained type pairs:")
    print("-" * 60)

    for pair, count in kept_pairs.most_common(20):

        print(
            f"{pair[0]:25s} -> "
            f"{pair[1]:25s} : {count}"
        )

    print("\nTop removed type pairs:")
    print("-" * 60)

    for pair, count in removed_pairs.most_common(20):

        print(
            f"{pair[0]:25s} -> "
            f"{pair[1]:25s} : {count}"
        )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()