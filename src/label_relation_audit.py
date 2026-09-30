import json
from pathlib import Path


INPUT_PATH = Path("data/relation_audit_100.jsonl")
OUTPUT_PATH = Path("data/relation_audit_100_labeled.jsonl")


# ============================================================
# HUMAN GOLD LABELS
# ============================================================

LABELS = {
    1:  ("UNCERTAIN", "TABLE"),
    2:  ("FALSE", "COMPARISON"),
    3:  ("FALSE", "NESTED_ENTITY"),
    4:  ("FALSE", "WRONG_SUBJECT"),
    5:  ("FALSE", "SEMANTIC_MISMATCH"),
    6:  ("FALSE", "TABLE"),
    7:  ("TRUE", "VALID_RELATION"),
    8:  ("FALSE", "COORDINATION_ERROR"),
    9:  ("FALSE", "WRONG_SUBJECT"),
    10: ("UNCERTAIN", "OTHER"),

    11: ("FALSE", "SEMANTIC_MISMATCH"),
    12: ("FALSE", "WRONG_SUBJECT"),
    13: ("FALSE", "WRONG_SUBJECT"),
    14: ("FALSE", "COMPARISON"),
    15: ("UNCERTAIN", "OTHER"),
    16: ("FALSE", "COMPARISON"),
    17: ("FALSE", "REPORTING_CONTEXT"),
    18: ("FALSE", "NESTED_ENTITY"),
    19: ("FALSE", "WRONG_DIRECTION"),
    20: ("UNCERTAIN", "TABLE"),

    21: ("FALSE", "WRONG_OBJECT"),
    22: ("FALSE", "PASSIVE_REVERSAL"),
    23: ("FALSE", "COORDINATION_ERROR"),
    24: ("UNCERTAIN", "OTHER"),
    25: ("TRUE", "VALID_RELATION"),
    26: ("FALSE", "TABLE"),
    27: ("FALSE", "COORDINATION_ERROR"),
    28: ("FALSE", "REPORTING_CONTEXT"),
    29: ("FALSE", "SEMANTIC_MISMATCH"),
    30: ("FALSE", "WRONG_OBJECT"),

    31: ("FALSE", "SEMANTIC_MISMATCH"),
    32: ("FALSE", "SEMANTIC_MISMATCH"),
    33: ("UNCERTAIN", "OTHER"),
    34: ("FALSE", "WRONG_SUBJECT"),
    35: ("UNCERTAIN", "OTHER"),
    36: ("FALSE", "TABLE"),
    37: ("FALSE", "SEMANTIC_MISMATCH"),
    38: ("FALSE", "TABLE"),
    39: ("FALSE", "REPORTING_CONTEXT"),
    40: ("FALSE", "REPORTING_CONTEXT"),

    41: ("FALSE", "TABLE"),
    42: ("FALSE", "WRONG_SUBJECT"),
    43: ("FALSE", "SEMANTIC_MISMATCH"),
    44: ("FALSE", "SEMANTIC_MISMATCH"),
    45: ("FALSE", "WRONG_SUBJECT"),
    46: ("FALSE", "SEMANTIC_MISMATCH"),
    47: ("FALSE", "WRONG_DIRECTION"),
    48: ("UNCERTAIN", "OTHER"),
    49: ("TRUE", "VALID_RELATION"),
    50: ("UNCERTAIN", "OTHER"),

    51: ("FALSE", "SEMANTIC_MISMATCH"),
    52: ("UNCERTAIN", "TABLE"),
    53: ("TRUE", "VALID_RELATION"),
    54: ("FALSE", "TABLE"),
    55: ("FALSE", "WRONG_DIRECTION"),
    56: ("FALSE", "TABLE"),
    57: ("FALSE", "WRONG_DIRECTION"),
    58: ("FALSE", "COORDINATION_ERROR"),
    59: ("TRUE", "VALID_RELATION"),
    60: ("FALSE", "COORDINATION_ERROR"),

    61: ("FALSE", "WRONG_SUBJECT"),
    62: ("FALSE", "SEMANTIC_MISMATCH"),
    63: ("FALSE", "COORDINATION_ERROR"),
    64: ("FALSE", "WRONG_SUBJECT"),
    65: ("UNCERTAIN", "OTHER"),
    66: ("TRUE", "VALID_RELATION"),
    67: ("FALSE", "SEMANTIC_MISMATCH"),
    68: ("FALSE", "TABLE"),
    69: ("UNCERTAIN", "OTHER"),
    70: ("FALSE", "WRONG_DIRECTION"),

    71: ("TRUE", "VALID_RELATION"),
    72: ("FALSE", "SEMANTIC_MISMATCH"),
    73: ("UNCERTAIN", "TABLE"),
    74: ("UNCERTAIN", "OTHER"),
    75: ("UNCERTAIN", "TABLE"),
    76: ("FALSE", "SEMANTIC_MISMATCH"),
    77: ("FALSE", "COORDINATION_ERROR"),
    78: ("FALSE", "COORDINATION_ERROR"),
    79: ("UNCERTAIN", "OTHER"),
    80: ("FALSE", "WRONG_DIRECTION"),

    81: ("UNCERTAIN", "OTHER"),
    82: ("FALSE", "COORDINATION_ERROR"),
    83: ("FALSE", "WRONG_DIRECTION"),
    84: ("FALSE", "COORDINATION_ERROR"),
    85: ("FALSE", "WRONG_DIRECTION"),
    86: ("TRUE", "VALID_RELATION"),
    87: ("FALSE", "TABLE"),
    88: ("FALSE", "SEMANTIC_MISMATCH"),
    89: ("FALSE", "WRONG_SUBJECT"),
    90: ("FALSE", "REPORTING_CONTEXT"),

    91: ("FALSE", "SEMANTIC_MISMATCH"),
    92: ("FALSE", "REPORTING_CONTEXT"),
    93: ("FALSE", "WRONG_DIRECTION"),
    94: ("FALSE", "WRONG_SUBJECT"),
    95: ("FALSE", "SEMANTIC_MISMATCH"),
    96: ("FALSE", "WRONG_DIRECTION"),
    97: ("UNCERTAIN", "OTHER"),
    98: ("TRUE", "VALID_RELATION"),
    99: ("FALSE", "WRONG_SUBJECT"),
    100: ("FALSE", "WRONG_SUBJECT"),
}


def main():

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    if len(records) != 100:

        raise ValueError(
            f"Expected 100 records, found {len(records)}"
        )

    for record in records:

        audit_id = int(
            record["audit_id"]
        )

        if audit_id not in LABELS:

            raise ValueError(
                f"Missing label for ID {audit_id}"
            )

        label, category = LABELS[audit_id]

        record["human_label"] = label
        record["error_category"] = category

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("=" * 60)
    print("AUDIT LABELS INJECTED")
    print("=" * 60)
    print(f"Records: {len(records)}")
    print(f"Output:  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()