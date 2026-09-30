import json
import re
from collections import Counter

INPUT_FILE = "data/annoctr_relation_predictions_arguments_v2.jsonl"
OUTPUT_FILE = "data/annoctr_relation_predictions_validated.jsonl"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_SVM_MARGIN = 0.70

SUPPORTED_THRESHOLD = 0.70
WEAK_THRESHOLD = 0.45

# Evidence patterns that are generally useful for each relation.
# These are deliberately conservative.
RELATION_PATTERNS = {
    "uses": [
        r"\buses?\b",
        r"\busing\b",
        r"\bused\b",
        r"\butiliz(?:e|es|ed|ing)\b",
        r"\bemploy(?:s|ed|ing)?\b",
        r"\bleverag(?:e|es|ed|ing)\b",
        r"\bdeploy(?:s|ed|ing)?\b",
        r"\bexecutes?\b",
        r"\bruns?\b",
        r"\bloads?\b",
    ],

    "targets": [
        r"\btargets?\b",
        r"\btargeted\b",
        r"\battacks?\b",
        r"\battacked\b",
        r"\bfocus(?:es|ed)?\b",
        r"\baim(?:s|ed)?\b",
        r"\bvictim(?:ize|izes|ized)\b",
    ],

    "originates-from": [
        r"\boriginat(?:e|es|ed)\b",
        r"\bassociat(?:e|ed|es)\b",
        r"\blink(?:ed|s)?\b",
        r"\btraced\b",
        r"\bbased\b",
        r"\bfrom\b",
    ],

    "located-at": [
        r"\blocated\b",
        r"\bbased\b",
        r"\boperat(?:e|es|ed|ing)\b",
        r"\bresid(?:e|es|ed)\b",
    ],

    "downloads": [
        r"\bdownloads?\b",
        r"\bdownloaded\b",
        r"\bretriev(?:e|es|ed)\b",
        r"\bfetch(?:es|ed)?\b",
    ],

    "communicates-with": [
        r"\bcommunicat(?:e|es|ed|ing)\b",
        r"\bconnect(?:s|ed|ing)?\b",
        r"\bcontact(?:s|ed|ing)?\b",
        r"\btalk(?:s|ed|ing)?\b",
    ],

    "exfiltrates-to": [
        r"\bexfiltrat(?:e|es|ed|ing)\b",
        r"\bupload(?:s|ed|ing)?\b",
        r"\btransfer(?:s|red|ring)?\b",
        r"\bsend(?:s|ing)?\b",
        r"\bsent\b",
    ],

    "attributed-to": [
        r"\battribut(?:e|ed|es|ing)\b",
        r"\bbeliev(?:e|ed|es)\b",
        r"\bcredit(?:ed|s)?\b",
        r"\bassociat(?:e|ed|es)\b",
        r"\blink(?:ed|s)?\b",
    ],

    "delivers": [
        r"\bdeliver(?:s|ed|ing)?\b",
    ],

    "impersonates": [
        r"\bimpersonat(?:e|es|ed|ing)\b",
    ],

    "variant-of": [
        r"\bvariant\b",
        r"\bvariants\b",
    ],

    "exploits": [
        r"\bexploit(?:s|ed|ing)?\b",
    ],
}


# ------------------------------------------------------------
# Entity type compatibility
# ------------------------------------------------------------

# This is NOT a hard KG filter.
# It is only one signal used by the validator.

TYPE_RULES = {
    "uses": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL",
            "TECHNIQUE", "CON", "TACTIC"
        },
        "target": {
            "MALWARE", "TOOL", "TECHNIQUE", "CON",
            "TACTIC", "ORG"
        }
    },

    "targets": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL",
            "TECHNIQUE", "CON"
        },
        "target": {
            "ORG", "SECTOR", "LOC", "MALWARE", "CON"
        }
    },

    "originates-from": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "LOC", "ORG"
        }
    },

    "located-at": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "LOC", "ORG"
        }
    },

    "downloads": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL"
        },
        "target": {
            "MALWARE", "TOOL", "CON"
        }
    },

    "communicates-with": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL"
        },
        "target": {
            "GROUP", "ORG", "MALWARE", "TOOL"
        }
    },

    "exfiltrates-to": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "ORG", "LOC", "CON"
        }
    },

    "attributed-to": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL",
            "CON", "TECHNIQUE"
        },
        "target": {
            "GROUP", "ORG"
        }
    },

    "delivers": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL", "CON"
        },
        "target": {
            "MALWARE", "TOOL", "CON", "TECHNIQUE"
        }
    },

    "impersonates": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL"
        },
        "target": {
            "ORG", "GROUP"
        }
    },

    "variant-of": {
        "source": {
            "MALWARE", "TOOL", "TECHNIQUE", "CON"
        },
        "target": {
            "MALWARE", "TOOL", "TECHNIQUE", "CON"
        }
    },

    "exploits": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL"
        },
        "target": {
            "CON", "TECHNIQUE", "MALWARE"
        }
    },
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def pattern_present(patterns, text):
    text = normalize(text)

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def exact_trigger_supported(record):
    """
    Check whether the extracted trigger itself belongs to the
    predicted relation.
    """

    relation = record["predicted_relation"]
    args = record["relation_arguments_v2"]

    trigger = args.get("trigger")

    if not trigger:
        return False

    patterns = RELATION_PATTERNS.get(relation, [])

    return pattern_present(patterns, trigger)


def evidence_contains_entities(record):
    """
    Check whether both entity strings actually appear in the
    evidence sentence.
    """

    args = record["relation_arguments_v2"]

    sentence = args.get("sentence")

    if not sentence:
        return False, False

    sentence_norm = normalize(sentence)

    source = normalize(record["source_mention"])
    target = normalize(record["target_mention"])

    source_present = source in sentence_norm
    target_present = target in sentence_norm

    return source_present, target_present


def type_compatible(record):

    relation = record["predicted_relation"]

    if relation not in TYPE_RULES:
        return True

    source_type = record.get("source_type")
    target_type = record.get("target_type")

    rules = TYPE_RULES[relation]

    return (
        source_type in rules["source"]
        and target_type in rules["target"]
    )


def get_argument_quality(record):

    args = record["relation_arguments_v2"]

    direction = args.get("direction")
    structure = args.get("argument_structure")

    if direction == "forward":
        return 1.0

    if direction == "reverse":
        return 0.0

    return 0.5


def dependency_quality(record):

    args = record["relation_arguments_v2"]

    structure = args.get("argument_structure")

    if structure in {
        "SOURCE_SUBJECT_TARGET_OBJECT",
        "TARGET_SUBJECT_SOURCE_OBJECT",
        "SOURCE_PASSIVE_SUBJECT_TARGET_AGENT",
        "TARGET_PASSIVE_SUBJECT_SOURCE_AGENT",
    }:
        return 1.0

    return 0.0


def evidence_quality(record):

    args = record["relation_arguments_v2"]

    sentence = args.get("sentence")

    if not sentence:
        return 0.0

    words = sentence.split()

    score = 1.0

    if len(words) <= 5:
        score -= 0.4

    source_present, target_present = evidence_contains_entities(record)

    if not source_present:
        score -= 0.3

    if not target_present:
        score -= 0.3

    return max(0.0, score)


# ------------------------------------------------------------
# Relation-specific semantic checks
# ------------------------------------------------------------

def relation_specific_check(record):

    relation = record["predicted_relation"]
    args = record["relation_arguments_v2"]

    sentence = args.get("sentence")

    if not sentence:
        return 0.0, ["NO_EVIDENCE_SENTENCE"]

    patterns = RELATION_PATTERNS.get(relation, [])

    if not pattern_present(patterns, sentence):
        return 0.0, ["NO_RELATION_TRIGGER_IN_EVIDENCE"]

    reasons = []

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = args.get("direction")

    if direction == "forward":
        reasons.append("FORWARD_ARGUMENT_STRUCTURE")

    elif direction == "reverse":
        reasons.append("REVERSE_ARGUMENT_STRUCTURE")

    else:
        reasons.append("UNKNOWN_ARGUMENT_STRUCTURE")

    # --------------------------------------------------------
    # Trigger
    # --------------------------------------------------------

    trigger = args.get("trigger")

    if trigger:
        if exact_trigger_supported(record):
            reasons.append("RELATION_TRIGGER_MATCH")
        else:
            reasons.append("TRIGGER_RELATION_MISMATCH")

    # --------------------------------------------------------
    # Relation-specific checks
    # --------------------------------------------------------

    if relation == "uses":

        if direction == "forward":
            score = 0.9
            reasons.append("ACTIVE_USE_PATTERN")

        elif direction == "reverse":
            score = 0.1
            reasons.append("REVERSED_USE_PATTERN")

        else:
            score = 0.3

    elif relation == "targets":

        if direction == "forward":
            score = 0.9
            reasons.append("ACTIVE_TARGET_PATTERN")

        elif direction == "reverse":
            score = 0.1
            reasons.append("REVERSED_TARGET_PATTERN")

        else:
            score = 0.3

    elif relation == "downloads":

        if direction == "forward":
            score = 0.9
            reasons.append("ACTIVE_DOWNLOAD_PATTERN")

        elif direction == "reverse":
            score = 0.1
            reasons.append("REVERSED_DOWNLOAD_PATTERN")

        else:
            score = 0.3

    elif relation == "communicates-with":

        if direction in {"forward", "reverse"}:
            score = 0.8
            reasons.append("COMMUNICATION_ARGUMENT_PATTERN")
        else:
            score = 0.3

    elif relation == "exfiltrates-to":

        if direction == "forward":
            score = 0.9
            reasons.append("FORWARD_EXFILTRATION_PATTERN")
        else:
            score = 0.2

    elif relation == "attributed-to":

        if direction in {"forward", "reverse"}:
            score = 0.75
            reasons.append("ATTRIBUTION_ARGUMENT_PATTERN")
        else:
            score = 0.3

    elif relation in {
        "originates-from",
        "located-at"
    }:

        if direction == "forward":
            score = 0.8
            reasons.append("LOCATION_ARGUMENT_PATTERN")
        else:
            score = 0.2

    else:

        if direction == "forward":
            score = 0.75
        elif direction == "reverse":
            score = 0.2
        else:
            score = 0.3

    return score, reasons


# ------------------------------------------------------------
# Main validation
# ------------------------------------------------------------

def validate(record):

    margin = float(record.get("margin", 0.0))

    args = record["relation_arguments_v2"]

    reasons = []

    # --------------------------------------------------------
    # 1. SVM confidence
    # --------------------------------------------------------

    if margin >= MIN_SVM_MARGIN:
        reasons.append("HIGH_SVM_MARGIN")
    else:
        reasons.append("LOW_SVM_MARGIN")

    # --------------------------------------------------------
    # 2. Evidence
    # --------------------------------------------------------

    eq = evidence_quality(record)

    if eq >= 0.9:
        reasons.append("GOOD_EVIDENCE")
    elif eq > 0:
        reasons.append("PARTIAL_EVIDENCE")
    else:
        reasons.append("NO_EVIDENCE")

    # --------------------------------------------------------
    # 3. Dependency argument quality
    # --------------------------------------------------------

    dq = dependency_quality(record)

    if dq > 0:
        reasons.append("DEPENDENCY_ARGUMENTS_RESOLVED")
    else:
        reasons.append("DEPENDENCY_ARGUMENTS_UNRESOLVED")

    # --------------------------------------------------------
    # 4. Entity presence
    # --------------------------------------------------------

    source_present, target_present = evidence_contains_entities(record)

    if source_present:
        reasons.append("SOURCE_IN_EVIDENCE")
    else:
        reasons.append("SOURCE_NOT_IN_EVIDENCE")

    if target_present:
        reasons.append("TARGET_IN_EVIDENCE")
    else:
        reasons.append("TARGET_NOT_IN_EVIDENCE")

    # --------------------------------------------------------
    # 5. Type compatibility
    # --------------------------------------------------------

    compatible = type_compatible(record)

    if compatible:
        reasons.append("TYPE_COMPATIBLE")
    else:
        reasons.append("TYPE_INCOMPATIBLE")

    # --------------------------------------------------------
    # 6. Relation-specific semantic score
    # --------------------------------------------------------

    semantic_score, semantic_reasons = relation_specific_check(
        record
    )

    reasons.extend(semantic_reasons)

    # --------------------------------------------------------
    # 7. Composite score
    # --------------------------------------------------------

    # We deliberately give argument structure and semantics
    # more weight than the SVM margin.

    margin_score = min(margin / 1.0, 1.0)

    composite = (
        0.20 * margin_score
        + 0.20 * eq
        + 0.25 * dq
        + 0.35 * semantic_score
    )

    # Type incompatibility is a strong warning,
    # but not an automatic rejection because the AnnoCTR
    # ontology does not perfectly match AZERG's ontology.

    if not compatible:
        composite -= 0.20
        composite = max(0.0, composite)

    # --------------------------------------------------------
    # 8. Final status
    # --------------------------------------------------------

    direction = args.get("direction")

    # Hard rejection conditions
    if not args.get("trigger_found"):
        status = "REJECTED"
        reasons.append("NO_TRIGGER")

    elif direction == "reverse":
        status = "REJECTED"
        reasons.append("RELATION_DIRECTION_REVERSED")

    elif not source_present or not target_present:
        status = "REJECTED"
        reasons.append("ENTITY_NOT_GROUNDED_IN_EVIDENCE")

    elif semantic_score < 0.30:
        status = "REJECTED"
        reasons.append("WEAK_RELATION_SEMANTICS")

    elif composite >= SUPPORTED_THRESHOLD:
        status = "SUPPORTED"

    elif composite >= WEAK_THRESHOLD:
        status = "WEAK"

    else:
        status = "REJECTED"

    return {
        "validation_status": status,
        "validation_score": round(composite, 4),
        "semantic_score": round(semantic_score, 4),
        "evidence_score": round(eq, 4),
        "dependency_score": round(dq, 4),
        "type_compatible": compatible,
        "reasons": reasons
    }


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------

print("=" * 75)
print("RELATION-SPECIFIC VALIDATION")
print("=" * 75)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Records loaded: {len(records)}")

output = []

status_counts = Counter()
relation_status = Counter()
reason_counts = Counter()

for i, record in enumerate(records, 1):

    result = validate(record)

    new_record = dict(record)
    new_record["relation_validation"] = result

    output.append(new_record)

    status_counts[
        result["validation_status"]
    ] += 1

    relation_status[
        (
            record["predicted_relation"],
            result["validation_status"]
        )
    ] += 1

    for reason in result["reasons"]:
        reason_counts[reason] += 1

    if i % 500 == 0:
        print(f"Processed {i}/{len(records)}")


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in output:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)

print(f"Total records: {len(records)}")

print()
print("Validation status:")

for status, count in status_counts.most_common():
    print(
        f"  {status:12s}: "
        f"{count:5d} "
        f"({count / len(records):.2%})"
    )

print()
print("Relation × validation status:")

relations = sorted(
    set(r["predicted_relation"] for r in records)
)

for relation in relations:

    supported = relation_status[
        (relation, "SUPPORTED")
    ]

    weak = relation_status[
        (relation, "WEAK")
    ]

    rejected = relation_status[
        (relation, "REJECTED")
    ]

    total = supported + weak + rejected

    print(
        f"  {relation:25s} "
        f"Supported={supported:4d} "
        f"Weak={weak:4d} "
        f"Rejected={rejected:4d} "
        f"Total={total:4d}"
    )

print()
print("Top validation reasons:")

for reason, count in reason_counts.most_common():
    print(
        f"  {reason:40s}: {count}"
    )

print()
print("Saved to:")
print(OUTPUT_FILE)