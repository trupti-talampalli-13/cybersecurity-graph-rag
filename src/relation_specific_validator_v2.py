import json
import re
from collections import Counter

INPUT_FILE = "data/annoctr_relation_predictions_arguments_v2.jsonl"
OUTPUT_FILE = "data/annoctr_relation_predictions_validated_v2.jsonl"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Margin is evidence, NOT a hard acceptance condition.
HIGH_MARGIN = 0.70
MEDIUM_MARGIN = 0.30

# Final score thresholds
SUPPORTED_THRESHOLD = 0.72
PLAUSIBLE_THRESHOLD = 0.45


# ------------------------------------------------------------
# Relation-specific trigger vocabulary
# ------------------------------------------------------------

RELATION_TRIGGERS = {
    "uses": {
        "use", "uses", "used", "using",
        "utilize", "utilizes", "utilized", "utilizing",
        "employ", "employs", "employed", "employing",
        "leverage", "leverages", "leveraged",
        "deploy", "deploys", "deployed", "deploying",
        "execute", "executes", "executed", "executing",
        "run", "runs", "ran",
        "load", "loads", "loaded", "loading"
    },

    "targets": {
        "target", "targets", "targeted",
        "attack", "attacks", "attacked",
        "focus", "focuses", "focused",
        "aim", "aims", "aimed",
        "victimize", "victimizes", "victimized"
    },

    "originates-from": {
        "originate", "originates", "originated",
        "associate", "associated", "associates",
        "link", "linked", "links",
        "trace", "traced"
    },

    "located-at": {
        "locate", "located", "locates",
        "base", "based",
        "operate", "operates", "operated", "operating",
        "reside", "resides", "resided"
    },

    "downloads": {
        "download", "downloads", "downloaded", "downloading",
        "retrieve", "retrieves", "retrieved", "retrieving",
        "fetch", "fetches", "fetched", "fetching"
    },

    "communicates-with": {
        "communicate", "communicates", "communicated",
        "connect", "connects", "connected",
        "contact", "contacts", "contacted",
        "talk", "talks", "talked"
    },

    "exfiltrates-to": {
        "exfiltrate", "exfiltrates", "exfiltrated",
        "upload", "uploads", "uploaded",
        "transfer", "transfers", "transferred",
        "send", "sends", "sent",
        "steal", "steals", "stole", "stolen"
    },

    "attributed-to": {
        "attribute", "attributed", "attributes",
        "believe", "believed", "believes",
        "credit", "credited", "credits",
        "associate", "associated", "associates",
        "link", "linked", "links"
    },

    "delivers": {
        "deliver", "delivers", "delivered", "delivering"
    },

    "impersonates": {
        "impersonate", "impersonates", "impersonated"
    },

    "variant-of": {
        "variant", "variants"
    },

    "exploits": {
        "exploit", "exploits", "exploited", "exploiting"
    }
}


# ------------------------------------------------------------
# Type compatibility
# ------------------------------------------------------------

TYPE_RULES = {

    "uses": {
        "source": {
            "GROUP", "ORG", "MALWARE", "TOOL",
            "TECHNIQUE", "CON", "TACTIC"
        },
        "target": {
            "MALWARE", "TOOL", "TECHNIQUE",
            "CON", "TACTIC", "ORG"
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
    }
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def norm(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def trigger_matches_relation(trigger, relation):
    if not trigger:
        return False

    trigger = norm(trigger)

    return trigger in RELATION_TRIGGERS.get(
        relation, set()
    )


def get_original_evidence(record):
    """
    IMPORTANT:
    The original prediction's 'text' is the evidence.
    The V2 sentence is the sentence selected for argument analysis.
    """

    text = record.get("text")

    if text:
        return text

    args = record.get("relation_arguments_v2", {})

    return args.get("sentence")


def evidence_contains_entity(text, entity):
    if not text or not entity:
        return False

    return norm(entity) in norm(text)


def evidence_quality(record):
    """
    Score whether usable evidence exists.
    This is independent of whether arguments were resolved.
    """

    text = get_original_evidence(record)

    if not text:
        return 0.0

    score = 1.0

    words = text.split()

    if len(words) <= 5:
        score -= 0.25

    source_present = evidence_contains_entity(
        text,
        record.get("source_mention")
    )

    target_present = evidence_contains_entity(
        text,
        record.get("target_mention")
    )

    if not source_present:
        score -= 0.25

    if not target_present:
        score -= 0.25

    return max(0.0, score)


def has_relation_trigger_in_evidence(record):

    relation = record["predicted_relation"]
    text = get_original_evidence(record)

    if not text:
        return False

    triggers = RELATION_TRIGGERS.get(
        relation,
        set()
    )

    words = re.findall(
        r"[A-Za-z][A-Za-z-]*",
        text.lower()
    )

    return any(
        word in triggers
        for word in words
    )


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


# ------------------------------------------------------------
# Argument analysis
# ------------------------------------------------------------

def argument_score(record):

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    direction = args.get("direction")

    if direction == "forward":
        return 1.0

    if direction == "reverse":
        return 0.0

    return 0.5


def argument_status(record):

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    direction = args.get("direction")

    if direction == "forward":
        return "FORWARD"

    if direction == "reverse":
        return "REVERSE"

    return "UNRESOLVED"


def trigger_score(record):

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    trigger = args.get("trigger")

    if not trigger:
        return 0.0

    relation = record["predicted_relation"]

    if trigger_matches_relation(
        trigger,
        relation
    ):
        return 1.0

    return 0.0


# ------------------------------------------------------------
# Relation-specific semantic reasoning
# ------------------------------------------------------------

def semantic_relation_score(record):

    relation = record["predicted_relation"]

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    direction = args.get("direction")

    trigger = args.get("trigger")

    text = get_original_evidence(record)

    score = 0.0
    reasons = []

    if not text:
        return 0.0, ["NO_EVIDENCE"]

    # --------------------------------------------------------
    # No trigger
    # --------------------------------------------------------

    if not trigger:
        reasons.append(
            "NO_EXTRACTED_TRIGGER"
        )

        return 0.15, reasons

    # --------------------------------------------------------
    # Wrong direction
    # --------------------------------------------------------

    if direction == "reverse":

        reasons.append(
            "PREDICTED_DIRECTION_REVERSED"
        )

        return 0.05, reasons

    # --------------------------------------------------------
    # Unresolved arguments
    # --------------------------------------------------------

    if direction == "unknown":

        reasons.append(
            "ARGUMENTS_NOT_RESOLVED"
        )

        # Evidence exists, but we don't know
        # whether source/target participate.
        return 0.35, reasons

    # --------------------------------------------------------
    # Forward argument structure
    # --------------------------------------------------------

    if direction == "forward":

        score += 0.45

        reasons.append(
            "SOURCE_TARGET_DIRECTION_SUPPORTED"
        )

    # --------------------------------------------------------
    # Trigger
    # --------------------------------------------------------

    if trigger_matches_relation(
        trigger,
        relation
    ):

        score += 0.25

        reasons.append(
            "RELATION_TRIGGER_MATCH"
        )

    # --------------------------------------------------------
    # Relation-specific checks
    # --------------------------------------------------------

    if relation == "uses":

        if direction == "forward":
            score += 0.25
            reasons.append(
                "ACTIVE_USE_RELATION"
            )

    elif relation == "targets":

        if direction == "forward":
            score += 0.25
            reasons.append(
                "ACTIVE_TARGET_RELATION"
            )

    elif relation == "downloads":

        if direction == "forward":
            score += 0.25
            reasons.append(
                "ACTIVE_DOWNLOAD_RELATION"
            )

    elif relation == "communicates-with":

        if direction == "forward":
            score += 0.20
            reasons.append(
                "ACTIVE_COMMUNICATION_RELATION"
            )

    elif relation == "exfiltrates-to":

        if direction == "forward":
            score += 0.25
            reasons.append(
                "ACTIVE_EXFILTRATION_RELATION"
            )

    elif relation == "attributed-to":

        if direction == "forward":
            score += 0.20
            reasons.append(
                "ATTRIBUTION_RELATION"
            )

    elif relation in {
        "originates-from",
        "located-at"
    }:

        if direction == "forward":
            score += 0.20
            reasons.append(
                "LOCATION_RELATION"
            )

    else:

        score += 0.15

    return min(score, 1.0), reasons


# ------------------------------------------------------------
# SVM confidence
# ------------------------------------------------------------

def margin_score(margin):

    margin = float(margin)

    if margin >= HIGH_MARGIN:
        return 1.0

    if margin <= 0:
        return 0.0

    return margin / HIGH_MARGIN


# ------------------------------------------------------------
# Main validation
# ------------------------------------------------------------

def validate(record):

    relation = record["predicted_relation"]

    margin = float(
        record.get("margin", 0.0)
    )

    args = record.get(
        "relation_arguments_v2",
        {}
    )

    evidence = get_original_evidence(record)

    # --------------------------------------------------------
    # Basic evidence
    # --------------------------------------------------------

    eq = evidence_quality(record)

    source_present = evidence_contains_entity(
        evidence,
        record.get("source_mention")
    )

    target_present = evidence_contains_entity(
        evidence,
        record.get("target_mention")
    )

    trigger_present = has_relation_trigger_in_evidence(
        record
    )

    # --------------------------------------------------------
    # Argument information
    # --------------------------------------------------------

    arg_status = argument_status(record)
    arg_score = argument_score(record)

    # --------------------------------------------------------
    # Type compatibility
    # --------------------------------------------------------

    compatible = type_compatible(record)

    # --------------------------------------------------------
    # Semantic relation score
    # --------------------------------------------------------

    semantic_score, semantic_reasons = (
        semantic_relation_score(record)
    )

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    m_score = margin_score(margin)

    # --------------------------------------------------------
    # Composite score
    # --------------------------------------------------------

    composite = (
        0.15 * m_score
        + 0.20 * eq
        + 0.20 * arg_score
        + 0.40 * semantic_score
        + 0.05 * (1.0 if compatible else 0.0)
    )

    reasons = []

    # --------------------------------------------------------
    # Evidence reasons
    # --------------------------------------------------------

    if evidence:
        reasons.append(
            "EVIDENCE_AVAILABLE"
        )
    else:
        reasons.append(
            "NO_USABLE_EVIDENCE"
        )

    if source_present:
        reasons.append(
            "SOURCE_IN_EVIDENCE"
        )
    else:
        reasons.append(
            "SOURCE_NOT_IN_EVIDENCE"
        )

    if target_present:
        reasons.append(
            "TARGET_IN_EVIDENCE"
        )
    else:
        reasons.append(
            "TARGET_NOT_IN_EVIDENCE"
        )

    if trigger_present:
        reasons.append(
            "TRIGGER_PRESENT_IN_EVIDENCE"
        )
    else:
        reasons.append(
            "TRIGGER_NOT_PRESENT_IN_EVIDENCE"
        )

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    if margin >= HIGH_MARGIN:
        reasons.append(
            "HIGH_SVM_MARGIN"
        )

    elif margin >= MEDIUM_MARGIN:
        reasons.append(
            "MEDIUM_SVM_MARGIN"
        )

    else:
        reasons.append(
            "LOW_SVM_MARGIN"
        )

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    if arg_status == "FORWARD":
        reasons.append(
            "FORWARD_ARGUMENTS"
        )

    elif arg_status == "REVERSE":
        reasons.append(
            "REVERSE_ARGUMENTS"
        )

    else:
        reasons.append(
            "ARGUMENTS_UNRESOLVED"
        )

    # --------------------------------------------------------
    # Types
    # --------------------------------------------------------

    if compatible:
        reasons.append(
            "TYPE_COMPATIBLE"
        )
    else:
        reasons.append(
            "TYPE_INCOMPATIBLE"
        )

    reasons.extend(
        semantic_reasons
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    # --------------------------------------------------------
    # DEFINITE UNSUPPORTED
    # --------------------------------------------------------

    if arg_status == "REVERSE":

        status = "UNSUPPORTED"

        reasons.append(
            "RELATION_DIRECTION_CONTRADICTED"
        )

    elif (
        source_present
        and target_present
        and semantic_score < 0.15
    ):

        status = "UNSUPPORTED"

        reasons.append(
            "SEMANTIC_RELATION_NOT_SUPPORTED"
        )

    elif (
        source_present
        and target_present
        and trigger_present
        and arg_status == "FORWARD"
        and semantic_score >= 0.70
        and composite >= SUPPORTED_THRESHOLD
    ):

        status = "SUPPORTED"

    # --------------------------------------------------------
    # PLAUSIBLE
    # --------------------------------------------------------

    elif evidence and (
        composite >= PLAUSIBLE_THRESHOLD
        or arg_status == "FORWARD"
        or trigger_present
    ):

        status = "PLAUSIBLE"

    # --------------------------------------------------------
    # UNVERIFIED
    # --------------------------------------------------------

    else:

        status = "UNVERIFIED"

    return {
        "status": status,
        "validation_score": round(
            composite,
            4
        ),
        "semantic_score": round(
            semantic_score,
            4
        ),
        "evidence_score": round(
            eq,
            4
        ),
        "argument_score": round(
            arg_score,
            4
        ),
        "margin_score": round(
            m_score,
            4
        ),
        "argument_status": arg_status,
        "type_compatible": compatible,
        "source_in_evidence": source_present,
        "target_in_evidence": target_present,
        "trigger_in_evidence": trigger_present,
        "reasons": reasons
    }


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------

print("=" * 75)
print("RELATION-SPECIFIC VALIDATION V2")
print("=" * 75)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

records = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():
            records.append(
                json.loads(line)
            )

print(
    f"Records loaded: {len(records)}"
)

output = []

status_counts = Counter()
relation_status = Counter()
reason_counts = Counter()

for i, record in enumerate(
    records,
    1
):

    validation = validate(
        record
    )

    new_record = dict(
        record
    )

    new_record[
        "relation_validation_v2"
    ] = validation

    output.append(
        new_record
    )

    status = validation["status"]

    status_counts[status] += 1

    relation_status[
        (
            record["predicted_relation"],
            status
        )
    ] += 1

    for reason in validation[
        "reasons"
    ]:

        reason_counts[reason] += 1

    if i % 500 == 0:

        print(
            f"Processed {i}/{len(records)}"
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

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
# Statistics
# ------------------------------------------------------------

print()
print("=" * 75)
print("VALIDATION V2 COMPLETE")
print("=" * 75)

print(
    f"Total records: {len(records)}"
)

print()
print("Validation status:")

for status, count in (
    status_counts
    .most_common()
):

    print(
        f"  {status:12s}: "
        f"{count:5d} "
        f"({count / len(records):.2%})"
    )


print()
print(
    "Relation × validation status:"
)

relations = sorted(
    set(
        r["predicted_relation"]
        for r in records
    )
)

for relation in relations:

    supported = relation_status[
        (relation, "SUPPORTED")
    ]

    plausible = relation_status[
        (relation, "PLAUSIBLE")
    ]

    unsupported = relation_status[
        (relation, "UNSUPPORTED")
    ]

    unverified = relation_status[
        (relation, "UNVERIFIED")
    ]

    total = (
        supported
        + plausible
        + unsupported
        + unverified
    )

    print(
        f"  {relation:25s} "
        f"Supported={supported:4d} "
        f"Plausible={plausible:4d} "
        f"Unsupported={unsupported:4d} "
        f"Unverified={unverified:4d} "
        f"Total={total:4d}"
    )


print()
print(
    "Top validation reasons:"
)

for reason, count in (
    reason_counts
    .most_common()
):

    print(
        f"  {reason:40s}: {count}"
    )


print()
print("Saved to:")
print(OUTPUT_FILE)