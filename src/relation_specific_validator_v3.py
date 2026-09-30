import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import spacy


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path("data/annoctr_relation_predictions_arguments_v2.jsonl")
OUTPUT_PATH = Path("data/annoctr_relation_predictions_validated_v3.jsonl")


# ============================================================
# LOAD SPACY
# ============================================================

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("ERROR: spaCy model 'en_core_web_sm' is not installed.")
    print("Run:")
    print("python -m spacy download en_core_web_sm")
    raise


# ============================================================
# RELATION TRIGGERS
# ============================================================

RELATION_TRIGGERS = {
    "uses": [
        r"\buse\b",
        r"\buses\b",
        r"\bused\b",
        r"\busing\b",
        r"\bemploy\b",
        r"\bemploys\b",
        r"\bemployed\b",
        r"\bdeploy\b",
        r"\bdeploys\b",
        r"\bdeployed\b",
        r"\bleverage\b",
        r"\bleverages\b",
        r"\bleveraged\b",
        r"\butilize\b",
        r"\butilizes\b",
        r"\butilized\b",
    ],

    "targets": [
        r"\btarget\b",
        r"\btargets\b",
        r"\btargeted\b",
        r"\btargeting\b",
        r"\battack\b",
        r"\battacks\b",
        r"\battacked\b",
        r"\baimed at\b",
        r"\baims at\b",
    ],

    "attributed-to": [
        r"\battributed to\b",
        r"\battribution to\b",
        r"\bcredited to\b",
    ],

    "originates-from": [
        r"\boriginates from\b",
        r"\boriginated from\b",
        r"\boriginating from\b",
        r"\bbased in\b",
        r"\boriginated in\b",
    ],

    "communicates-with": [
        r"\bcommunicate with\b",
        r"\bcommunicates with\b",
        r"\bcommunicated with\b",
        r"\bcommunicating with\b",
        r"\bconnect with\b",
        r"\bconnects with\b",
        r"\bconnected with\b",
        r"\bconnect to\b",
        r"\bconnects to\b",
        r"\bconnected to\b",
        r"\bcontact\b",
        r"\bcontacts\b",
        r"\bbeacon\b",
        r"\bbeacons\b",
    ],

    "exfiltrates-to": [
        r"\bexfiltrate\b",
        r"\bexfiltrates\b",
        r"\bexfiltrated\b",
        r"\bexfiltrating\b",
        r"\bsend\b",
        r"\bsends\b",
        r"\bsent\b",
        r"\bupload\b",
        r"\buploads\b",
        r"\buploaded\b",
        r"\btransfer\b",
        r"\btransfers\b",
        r"\btransferred\b",
    ],

    "located-at": [
        r"\blocated in\b",
        r"\blocated at\b",
        r"\bbased in\b",
        r"\bheadquartered in\b",
        r"\bresident in\b",
        r"\bresides in\b",
    ],

    "downloads": [
        r"\bdownload\b",
        r"\bdownloads\b",
        r"\bdownloaded\b",
        r"\bdownloading\b",
        r"\bretrieve\b",
        r"\bretrieves\b",
        r"\bfetch\b",
        r"\bfetches\b",
    ],

    "delivers": [
        r"\bdeliver\b",
        r"\bdelivers\b",
        r"\bdelivered\b",
        r"\bdelivering\b",
    ],

    "drops": [
        r"\bdrop\b",
        r"\bdrops\b",
        r"\bdropped\b",
        r"\bdropping\b",
    ],

    "impersonates": [
        r"\bimpersonate\b",
        r"\bimpersonates\b",
        r"\bimpersonated\b",
        r"\bposing as\b",
        r"\bmasquerade as\b",
    ],

    "variant-of": [
        r"\bvariant of\b",
        r"\bversion of\b",
        r"\bderived from\b",
    ],

    "exploits": [
        r"\bexploit\b",
        r"\bexploits\b",
        r"\bexploited\b",
        r"\bexploiting\b",
    ],

    "owns": [
        r"\bowns\b",
        r"\bowned by\b",
        r"\bownership\b",
    ],

    "indicates": [
        r"\bindicates\b",
        r"\bindicate\b",
        r"\bindicated\b",
        r"\bsuggests\b",
        r"\bsuggested\b",
    ],

    "authored-by": [
        r"\bauthored by\b",
        r"\bwritten by\b",
        r"\bcreated by\b",
    ],
}


# ============================================================
# NEGATION / UNCERTAINTY
# ============================================================

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bwithout\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
    r"\bunlikely\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bpotential\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuspected\b",
    r"\bsuspected to\b",
    r"\bno evidence\b",
    r"\bcannot independently confirm\b",
    r"\bunclear\b",
    r"\buncertain\b",
]


# ============================================================
# COMPARISON / SIMILARITY
# ============================================================

COMPARISON_PATTERNS = [
    r"\bsame .* as\b",
    r"\bsimilar to\b",
    r"\bsimilarly\b",
    r"\bsame manner\b",
    r"\bsame way\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\bas .* as\b",
    r"\blike\b",
]


# ============================================================
# REPORTING / OBSERVER CONTEXT
# ============================================================

REPORTING_PATTERNS = [
    r"\breported by\b",
    r"\bobserved by\b",
    r"\bfound by\b",
    r"\bidentified by\b",
    r"\baccording to\b",
    r"\bresearchers\b",
    r"\banalysts\b",
    r"\binvestigators\b",
    r"\bresearch from\b",
    r"\bresearchers at\b",
    r"\bthe report\b",
    r"\bthe researchers\b",
]


# ============================================================
# TABLE / CAPTION
# ============================================================

TABLE_PATTERNS = [
    r"\|",
    r"^\s*(figure|fig\.|table|illustration|exhibit)\b",
    r"^\s*source\s*:",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, patterns):
    text = text.lower()

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def find_relation_trigger(sentence, relation):
    patterns = RELATION_TRIGGERS.get(relation, [])

    for pattern in patterns:
        match = re.search(pattern, sentence.lower())

        if match:
            return match.group(0), match.start(), match.end()

    return None, None, None


def get_sentence_containing_entities(text, source, target):
    """
    Find the best sentence containing both source and target.
    """

    if not text:
        return None

    doc = nlp(text)

    source_l = source.lower()
    target_l = target.lower()

    candidates = []

    for sent in doc.sents:
        s = sent.text.strip()
        sl = s.lower()

        if source_l in sl and target_l in sl:
            candidates.append(s)

    if candidates:
        # Prefer shortest sentence containing both entities.
        return min(candidates, key=len)

    # If no sentence contains both, find sentence containing either.
    for sent in doc.sents:
        s = sent.text.strip()
        sl = s.lower()

        if source_l in sl or target_l in sl:
            return s

    return None


def entity_in_sentence(entity, sentence):
    if not entity or not sentence:
        return False

    return entity.lower() in sentence.lower()


def trigger_in_sentence(relation, sentence):
    if not sentence:
        return None

    trigger, start, end = find_relation_trigger(sentence, relation)

    if trigger:
        return {
            "trigger": trigger,
            "start": start,
            "end": end,
        }

    return None


def dependency_argument_check(sentence, source, target, relation):
    """
    Lightweight dependency-aware validation.

    This does NOT claim perfect semantic parsing.
    It only detects strong subject/object/passive patterns.
    """

    if not sentence:
        return {
            "status": "UNKNOWN",
            "score": 0.0,
            "structure": None,
        }

    doc = nlp(sentence)

    source_token = None
    target_token = None

    for token in doc:
        if source.lower() == token.text.lower():
            source_token = token

        if target.lower() == token.text.lower():
            target_token = token

    # Multi-token entities may not match one token exactly.
    # Fall back to substring logic.
    if source_token is None or target_token is None:
        return {
            "status": "UNKNOWN",
            "score": 0.0,
            "structure": None,
        }

    # Direct subject -> object
    if source_token.dep_ in {"nsubj", "nsubjpass"}:

        if target_token.dep_ in {"dobj", "obj", "attr", "pobj"}:
            return {
                "status": "FORWARD",
                "score": 1.0,
                "structure": "SOURCE_SUBJECT_TARGET_OBJECT",
            }

    # Passive:
    # target is grammatical subject, source is agent via "by"
    if target_token.dep_ == "nsubjpass":
        if source_token.dep_ in {"agent", "pobj"}:
            return {
                "status": "REVERSE_PASSIVE",
                "score": 1.0,
                "structure": "TARGET_PASSIVE_SUBJECT_SOURCE_AGENT",
            }

    return {
        "status": "UNKNOWN",
        "score": 0.0,
        "structure": None,
    }


def margin_score(margin):
    """
    Margin is used only as supporting evidence.
    It is NOT a hard gate.
    """

    try:
        margin = float(margin)
    except Exception:
        return 0.0

    if margin >= 1.0:
        return 1.0

    if margin >= 0.7:
        return 0.8

    if margin >= 0.4:
        return 0.6

    if margin >= 0.2:
        return 0.4

    return 0.2


def type_compatibility(source_type, target_type, relation):
    """
    Conservative type compatibility.
    This is only a supporting signal.
    """

    source_type = (source_type or "").upper()
    target_type = (target_type or "").upper()

    if relation == "uses":
        allowed = {
            "GROUP": {
                "MALWARE", "TOOL", "TECHNIQUE", "CON", "ORG"
            },
            "MALWARE": {
                "TOOL", "TECHNIQUE", "CON", "MALWARE"
            },
            "ORG": {
                "TOOL", "MALWARE", "TECHNIQUE", "CON"
            },
        }

        return target_type in allowed.get(source_type, set())

    if relation == "targets":
        return source_type in {
            "GROUP", "MALWARE", "ORG", "TOOL"
        }

    if relation == "communicates-with":
        return True

    if relation == "located-at":
        return target_type in {"LOC", "ORG"}

    if relation == "originates-from":
        return target_type in {"LOC", "ORG"}

    return True


# ============================================================
# SEMANTIC CHECKS
# ============================================================

def check_negation(sentence, trigger_start):
    if not sentence:
        return False

    lower = sentence.lower()

    # Look around the relation trigger rather than blindly treating
    # every "not" in a long paragraph as relation negation.
    left = max(0, trigger_start - 80)
    context = lower[left:trigger_start + 80]

    return contains_any(context, NEGATION_PATTERNS)


def check_comparison(sentence):
    if not sentence:
        return False

    return contains_any(sentence, COMPARISON_PATTERNS)


def check_reporting_context(sentence):
    if not sentence:
        return False

    return contains_any(sentence, REPORTING_PATTERNS)


def check_table_caption(sentence, original_text):
    text = sentence or original_text or ""

    for pattern in TABLE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def check_uses_semantics(sentence, source, target, trigger):
    """
    Extra semantic checks for the highly frequent USES relation.
    """

    if not sentence or not trigger:
        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    trigger_pos = lower.find(trigger.lower())

    if trigger_pos == -1:
        return "UNKNOWN"

    after = lower[trigger_pos + len(trigger):]

    # --------------------------------------------------------
    # 1. Target occurs after "same ... as"
    # --------------------------------------------------------

    if re.search(
        r"\bsame\b.{0,80}\bas\s+" + re.escape(target_l),
        after
    ):
        return "COMPARISON"

    # --------------------------------------------------------
    # 2. Target is part of "used in X incidents"
    # --------------------------------------------------------

    if re.search(
        r"\bused\b.{0,40}\bin\s+" + re.escape(target_l),
        after
    ):
        return "CONTEXT"

    # --------------------------------------------------------
    # 3. Target follows a purpose phrase
    # --------------------------------------------------------

    if re.search(
        r"\bto\s+\w+\b.{0,50}" + re.escape(target_l),
        after
    ):
        return "PURPOSE_CONTEXT"

    # --------------------------------------------------------
    # 4. Explicit comparison
    # --------------------------------------------------------

    if re.search(
        r"\b(as|like|similar to|same)\b.{0,80}" +
        re.escape(target_l),
        after
    ):
        return "COMPARISON"

    # --------------------------------------------------------
    # 5. Strong object-like pattern
    # --------------------------------------------------------

    if re.search(
        r"\b(?:use|uses|used|using|employ|employs|employed|"
        r"deploy|deploys|deployed|leverage|leverages|leveraged|"
        r"utilize|utilizes|utilized)\b.{0,60}" +
        re.escape(target_l),
        lower
    ):
        return "DIRECT_OBJECT_CANDIDATE"

    return "UNKNOWN"


# ============================================================
# RECORD VALIDATION
# ============================================================

def validate_record(record):

    # --------------------------------------------------------
    # IMPORTANT: support both V2 schemas
    # --------------------------------------------------------

    source = record.get(
        "source",
        record.get("source_mention", "")
    )

    target = record.get(
        "target",
        record.get("target_mention", "")
    )

    relation = record.get(
        "relation",
        record.get("predicted_relation", "")
    )

    source_type = record.get(
        "source_type",
        ""
    )

    target_type = record.get(
        "target_type",
        ""
    )

    original_text = normalize(
        record.get("original_text",
                   record.get("text", ""))
    )

    # Existing sentence from argument extractor
    existing_sentence = normalize(
        record.get("sentence", "")
    )

    # --------------------------------------------------------
    # Evidence sentence
    # --------------------------------------------------------

    sentence = existing_sentence

    if not sentence:
        sentence = get_sentence_containing_entities(
            original_text,
            source,
            target
        )

    # --------------------------------------------------------
    # Basic flags
    # --------------------------------------------------------

    source_present = entity_in_sentence(source, sentence)
    target_present = entity_in_sentence(target, sentence)

    trigger_info = trigger_in_sentence(
        relation,
        sentence
    )

    trigger_present = trigger_info is not None

    trigger = (
        trigger_info["trigger"]
        if trigger_info
        else None
    )

    trigger_start = (
        trigger_info["start"]
        if trigger_info
        else 0
    )

    # --------------------------------------------------------
    # Context checks
    # --------------------------------------------------------

    negated = check_negation(
        sentence,
        trigger_start
    )

    comparison = check_comparison(sentence)

    reporting = check_reporting_context(sentence)

    table_caption = check_table_caption(
        sentence,
        original_text
    )

    uncertainty = contains_any(
        sentence or "",
        UNCERTAINTY_PATTERNS
    )

    # --------------------------------------------------------
    # Dependency
    # --------------------------------------------------------

    dependency = dependency_argument_check(
        sentence,
        source,
        target,
        relation
    )

    # --------------------------------------------------------
    # Type compatibility
    # --------------------------------------------------------

    type_ok = type_compatibility(
        source_type,
        target_type,
        relation
    )

    # --------------------------------------------------------
    # SVM margin
    # --------------------------------------------------------

    margin = record.get(
        "margin",
        record.get("decision_margin", 0.0)
    )

    try:
        margin = float(margin)
    except Exception:
        margin = 0.0

    m_score = margin_score(margin)

    # --------------------------------------------------------
    # Relation-specific semantic checks
    # --------------------------------------------------------

    semantic_status = "UNKNOWN"

    if relation == "uses":
        semantic_status = check_uses_semantics(
            sentence,
            source,
            target,
            trigger
        )

    # --------------------------------------------------------
    # Reason collection
    # --------------------------------------------------------

    reasons = []

    if source_present:
        reasons.append("SOURCE_IN_SENTENCE")
    else:
        reasons.append("SOURCE_NOT_IN_SENTENCE")

    if target_present:
        reasons.append("TARGET_IN_SENTENCE")
    else:
        reasons.append("TARGET_NOT_IN_SENTENCE")

    if trigger_present:
        reasons.append("RELATION_TRIGGER_IN_SENTENCE")
    else:
        reasons.append("NO_RELATION_TRIGGER")

    if dependency["status"] == "FORWARD":
        reasons.append("FORWARD_DEPENDENCY_STRUCTURE")

    elif dependency["status"] == "REVERSE_PASSIVE":
        reasons.append("REVERSE_PASSIVE_STRUCTURE")

    else:
        reasons.append("DEPENDENCY_STRUCTURE_UNRESOLVED")

    if type_ok:
        reasons.append("TYPE_COMPATIBLE")
    else:
        reasons.append("TYPE_INCOMPATIBLE")

    if negated:
        reasons.append("NEGATED_RELATION")

    if comparison:
        reasons.append("COMPARISON_OR_SIMILARITY_CONTEXT")

    if reporting:
        reasons.append("REPORTING_OR_OBSERVER_CONTEXT")

    if uncertainty:
        reasons.append("UNCERTAINTY_LANGUAGE")

    if table_caption:
        reasons.append("TABLE_OR_CAPTION_CONTEXT")

    if semantic_status == "DIRECT_OBJECT_CANDIDATE":
        reasons.append("DIRECT_USE_OBJECT")

    elif semantic_status == "COMPARISON":
        reasons.append("USE_COMPARISON_CONTEXT")

    elif semantic_status == "CONTEXT":
        reasons.append("USE_CONTEXT_NOT_DIRECT_OBJECT")

    elif semantic_status == "PURPOSE_CONTEXT":
        reasons.append("USE_PURPOSE_CONTEXT")

    if margin >= 1.0:
        reasons.append("VERY_HIGH_SVM_MARGIN")
    elif margin >= 0.7:
        reasons.append("HIGH_SVM_MARGIN")
    elif margin >= 0.4:
        reasons.append("MODERATE_SVM_MARGIN")
    else:
        reasons.append("LOW_SVM_MARGIN")

    # ========================================================
    # STATUS DECISION
    # ========================================================

    status = "UNVERIFIED"

    # --------------------------------------------------------
    # HARD UNSUPPORTED CONDITIONS
    # --------------------------------------------------------

    if not source_present or not target_present:
        status = "UNVERIFIED"

    elif table_caption:
        status = "UNSUPPORTED"

    elif negated:
        status = "UNSUPPORTED"

    elif comparison:
        status = "UNSUPPORTED"

    elif semantic_status in {
        "COMPARISON",
        "CONTEXT",
        "PURPOSE_CONTEXT"
    }:
        status = "UNSUPPORTED"

    elif dependency["status"] == "REVERSE_PASSIVE":
        status = "UNSUPPORTED"

    elif relation == "uses" and semantic_status == "UNKNOWN":
        status = "UNVERIFIED"

    # --------------------------------------------------------
    # STRONG SUPPORTED
    # --------------------------------------------------------

    elif (
        trigger_present
        and dependency["status"] == "FORWARD"
        and type_ok
        and not reporting
        and not uncertainty
        and (
            semantic_status == "DIRECT_OBJECT_CANDIDATE"
            or relation != "uses"
        )
    ):
        status = "SUPPORTED"

    # --------------------------------------------------------
    # PLAUSIBLE
    # --------------------------------------------------------

    elif (
        trigger_present
        and source_present
        and target_present
        and not negated
        and not comparison
        and not table_caption
        and not uncertainty
    ):
        status = "PLAUSIBLE"

    # --------------------------------------------------------
    # Otherwise
    # --------------------------------------------------------

    else:
        status = "UNVERIFIED"

    # ========================================================
    # OUTPUT
    # ========================================================

    output = dict(record)

    output.update({
        "source": source,
        "target": target,
        "relation": relation,
        "evidence_sentence": sentence,
        "trigger": trigger,

        "source_in_sentence": source_present,
        "target_in_sentence": target_present,
        "trigger_in_sentence": trigger_present,

        "negated": negated,
        "comparison_context": comparison,
        "reporting_context": reporting,
        "uncertainty_context": uncertainty,
        "table_caption_context": table_caption,

        "dependency_status": dependency["status"],
        "dependency_score": dependency["score"],
        "dependency_structure": dependency["structure"],

        "semantic_status": semantic_status,

        "type_compatible": type_ok,

        "margin": margin,
        "margin_score": m_score,

        "status": status,
        "validation_reasons": reasons,
    })

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RELATION-SPECIFIC VALIDATOR V3")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    print(f"Input records: {len(records)}")

    results = []

    status_counts = Counter()
    relation_status = defaultdict(Counter)
    reason_counts = Counter()

    for i, record in enumerate(records):

        try:
            result = validate_record(record)

            results.append(result)

            status = result["status"]
            relation = result["relation"]

            status_counts[status] += 1
            relation_status[relation][status] += 1

            for reason in result["validation_reasons"]:
                reason_counts[reason] += 1

        except Exception as e:

            result = dict(record)

            result.update({
                "status": "UNVERIFIED",
                "validation_error": str(e),
            })

            results.append(result)

            status_counts["UNVERIFIED"] += 1

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(records)}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(results)

    print()
    print("=" * 70)
    print("STATUS SUMMARY")
    print("=" * 70)

    for status in [
        "SUPPORTED",
        "PLAUSIBLE",
        "UNSUPPORTED",
        "UNVERIFIED",
    ]:

        count = status_counts[status]

        pct = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{status:15s}: "
            f"{count:5d} "
            f"({pct:6.2f}%)"
        )

    # ========================================================
    # RELATION × STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("RELATION × STATUS")
    print("=" * 70)

    all_relations = sorted(relation_status.keys())

    for relation in all_relations:

        counts = relation_status[relation]

        print(
            f"{relation:20s} "
            f"SUPPORTED={counts['SUPPORTED']:4d} "
            f"PLAUSIBLE={counts['PLAUSIBLE']:4d} "
            f"UNSUPPORTED={counts['UNSUPPORTED']:4d} "
            f"UNVERIFIED={counts['UNVERIFIED']:4d}"
        )

    # ========================================================
    # REASONS
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION REASONS")
    print("=" * 70)

    for reason, count in reason_counts.most_common():

        print(
            f"{reason:40s}: {count}"
        )

    # ========================================================
    # SUPPORTED EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("SUPPORTED EXAMPLES")
    print("=" * 70)

    supported = [
        r for r in results
        if r["status"] == "SUPPORTED"
    ]

    for i, r in enumerate(supported[:30], 1):

        print()
        print(
            f"[{i}] "
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
        )

        print(
            f"    Types: "
            f"{r.get('source_type')} -> "
            f"{r.get('target_type')}"
        )

        print(
            f"    Margin: "
            f"{r.get('margin')}"
        )

        print(
            f"    Trigger: "
            f"{r.get('trigger')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

        print(
            f"    Reasons: "
            f"{', '.join(r.get('validation_reasons', []))}"
        )

    # ========================================================
    # PLAUSIBLE EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("PLAUSIBLE EXAMPLES")
    print("=" * 70)

    plausible = [
        r for r in results
        if r["status"] == "PLAUSIBLE"
    ]

    for i, r in enumerate(plausible[:20], 1):

        print()
        print(
            f"[{i}] "
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

        print(
            f"    Reasons: "
            f"{', '.join(r.get('validation_reasons', []))}"
        )

    # ========================================================
    # UNSUPPORTED EXAMPLES
    # ========================================================

    print()
    print("=" * 70)
    print("UNSUPPORTED EXAMPLES")
    print("=" * 70)

    unsupported = [
        r for r in results
        if r["status"] == "UNSUPPORTED"
    ]

    for i, r in enumerate(unsupported[:20], 1):

        print()
        print(
            f"[{i}] "
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

        print(
            f"    Reasons: "
            f"{', '.join(r.get('validation_reasons', []))}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()