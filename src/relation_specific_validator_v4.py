import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import spacy


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/annoctr_relation_predictions_arguments_v2.jsonl"
)

OUTPUT_PATH = Path(
    "data/annoctr_relation_predictions_validated_v4.jsonl"
)


# ============================================================
# LOAD NLP MODEL
# ============================================================

print("=" * 70)
print("RELATION VALIDATOR V4")
print("=" * 70)

nlp = spacy.load("en_core_web_sm")


# ============================================================
# RELATION TRIGGERS
# ============================================================

TRIGGERS = {
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
        r"\bbeacon\b",
        r"\bbeacons\b",
        r"\bcontact\b",
        r"\bcontacts\b",
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

    "exploits": [
        r"\bexploit\b",
        r"\bexploits\b",
        r"\bexploited\b",
        r"\bexploiting\b",
    ],

    "impersonates": [
        r"\bimpersonate\b",
        r"\bimpersonates\b",
        r"\bimpersonated\b",
        r"\bposing as\b",
        r"\bmasquerade as\b",
    ],

    "located-at": [
        r"\blocated in\b",
        r"\blocated at\b",
        r"\bbased in\b",
        r"\bheadquartered in\b",
        r"\bresident in\b",
        r"\bresides in\b",
    ],

    "originates-from": [
        r"\boriginates from\b",
        r"\boriginated from\b",
        r"\boriginated in\b",
        r"\bbased in\b",
    ],

    "attributed-to": [
        r"\battributed to\b",
        r"\battribution to\b",
        r"\bcredited to\b",
    ],

    "variant-of": [
        r"\bvariant of\b",
        r"\bversion of\b",
        r"\bderived from\b",
    ],

    "owns": [
        r"\bowns\b",
        r"\bowned by\b",
        r"\bownership\b",
    ],

    "authored-by": [
        r"\bauthored by\b",
        r"\bwritten by\b",
        r"\bcreated by\b",
    ],

    "indicates": [
        r"\bindicates\b",
        r"\bindicate\b",
        r"\bindicated\b",
        r"\bsuggests\b",
        r"\bsuggested\b",
    ],
}


# ============================================================
# NEGATION
# ============================================================

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bwithout\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoesn't\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
]


# ============================================================
# UNCERTAINTY
# ============================================================

UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bpotential\b",
    r"\bpotentially\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuspected\b",
    r"\bsuspected to\b",
    r"\bunclear\b",
    r"\buncertain\b",
    r"\bno evidence\b",
    r"\bcannot independently confirm\b",
]


# ============================================================
# COMPARISON
# ============================================================

COMPARISON_PATTERNS = [
    r"\bsame\b.{0,80}\bas\b",
    r"\bsimilar to\b",
    r"\bsimilarly\b",
    r"\bsame manner\b",
    r"\bsame way\b",
    r"\bcompared to\b",
    r"\bcompared with\b",
]


# ============================================================
# REPORTING / OBSERVER
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
]


# ============================================================
# TABLE / CAPTION
# ============================================================

def is_table_or_caption(text):
    if not text:
        return False

    stripped = text.strip()

    if "|" in stripped:
        return True

    if re.match(
        r"^(figure|fig\.|table|illustration|exhibit)\b",
        stripped,
        re.IGNORECASE
    ):
        return True

    return False


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def find_trigger(sentence, relation):
    if not sentence:
        return None

    for pattern in TRIGGERS.get(relation, []):

        match = re.search(
            pattern,
            sentence,
            re.IGNORECASE
        )

        if match:
            return {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            }

    return None


def entity_span(sentence, entity):
    """
    Find character span of an entity.
    """

    if not sentence or not entity:
        return None

    match = re.search(
        re.escape(entity),
        sentence,
        re.IGNORECASE
    )

    if match:
        return match.start(), match.end()

    return None


# ============================================================
# SENTENCE SELECTION
# ============================================================

def select_evidence_sentence(text, source, target):

    if not text:
        return None

    doc = nlp(text)

    source_l = source.lower()
    target_l = target.lower()

    both = []

    for sent in doc.sents:

        s = sent.text.strip()
        sl = s.lower()

        if source_l in sl and target_l in sl:
            both.append(s)

    if both:
        return min(both, key=len)

    return None


# ============================================================
# DEPENDENCY ANALYSIS
# ============================================================

def dependency_relation(
    sentence,
    source,
    target,
    relation,
    trigger
):

    if not sentence or not trigger:
        return {
            "status": "UNKNOWN",
            "structure": None,
            "score": 0.0
        }

    doc = nlp(sentence)

    source_tokens = []
    target_tokens = []

    source_l = source.lower()
    target_l = target.lower()

    for token in doc:

        if token.text.lower() in source_l.split():
            source_tokens.append(token)

        if token.text.lower() in target_l.split():
            target_tokens.append(token)

    if not source_tokens or not target_tokens:

        return {
            "status": "UNKNOWN",
            "structure": None,
            "score": 0.0
        }

    source_token = source_tokens[0]
    target_token = target_tokens[0]

    # --------------------------------------------------------
    # Find predicate associated with trigger
    # --------------------------------------------------------

    trigger_doc = nlp(sentence)

    trigger_token = None

    for token in trigger_doc:

        if token.text.lower() == trigger.lower():
            trigger_token = token
            break

    # --------------------------------------------------------
    # Direct active subject -> object
    # --------------------------------------------------------

    if source_token.dep_ in {"nsubj", "nsubjpass"}:

        if target_token.dep_ in {
            "dobj",
            "obj",
            "attr"
        }:

            return {
                "status": "FORWARD",
                "structure": "SUBJECT_OBJECT",
                "score": 1.0
            }

    # --------------------------------------------------------
    # Passive:
    #
    # Malware is used by Group
    #
    # grammatical subject = target
    # agent = source
    # --------------------------------------------------------

    if target_token.dep_ == "nsubjpass":

        if source_token.dep_ in {
            "pobj",
            "agent"
        }:

            return {
                "status": "REVERSE",
                "structure": "PASSIVE_AGENT",
                "score": 1.0
            }

    # --------------------------------------------------------
    # Prepositional argument
    # --------------------------------------------------------

    if target_token.dep_ == "pobj":

        head = target_token.head

        if head.text.lower() in {
            "with",
            "to",
            "at",
            "from",
            "against",
            "by"
        }:

            return {
                "status": "FORWARD_PREPOSITION",
                "structure": (
                    f"PREPOSITION_{head.text.upper()}"
                ),
                "score": 0.8
            }

    return {
        "status": "UNKNOWN",
        "structure": None,
        "score": 0.0
    }


# ============================================================
# SPECIAL USES ANALYSIS
# ============================================================

def analyze_uses(sentence, source, target, trigger):

    if not sentence or not trigger:
        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    trigger_match = re.search(
        re.escape(trigger),
        lower
    )

    if not trigger_match:
        return "UNKNOWN"

    trigger_end = trigger_match.end()

    after = lower[trigger_end:]

    # --------------------------------------------------------
    # Passive construction
    #
    # X is used by Y
    # --------------------------------------------------------

    passive_pattern = (
        re.escape(source_l)
        + r".{0,50}\b(is|was|are|were|has been|have been)"
        r"\s+used\b.{0,50}\bby\s+"
        + re.escape(target_l)
    )

    if re.search(
        passive_pattern,
        lower
    ):
        return "PASSIVE_REVERSE"

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if re.search(
        r"\bsame\b.{0,80}\bas\s+"
        + re.escape(target_l),
        after
    ):
        return "COMPARISON"

    if re.search(
        r"\bsimilar\b.{0,80}"
        + re.escape(target_l),
        lower
    ):
        return "COMPARISON"

    # --------------------------------------------------------
    # "used in X incidents"
    # --------------------------------------------------------

    if re.search(
        r"\bused\b.{0,50}\bin\s+"
        + re.escape(target_l),
        lower
    ):
        return "CONTEXT"

    # --------------------------------------------------------
    # "used by X"
    #
    # Source itself is the thing being used.
    # --------------------------------------------------------

    if re.search(
        r"\bused\b.{0,30}\bby\s+"
        + re.escape(target_l),
        lower
    ):
        return "PASSIVE_REVERSE"

    # --------------------------------------------------------
    # Direct target immediately after use verb
    # --------------------------------------------------------

    if re.search(
        r"\b(use|uses|used|using|employ|employs|"
        r"employed|deploy|deploys|deployed|leverage|"
        r"leverages|leveraged|utilize|utilizes|utilized)\b"
        r"(?:\s+\w+){0,8}\s+"
        + re.escape(target_l),
        lower
    ):
        return "DIRECT"

    return "UNKNOWN"


# ============================================================
# TARGETS ANALYSIS
# ============================================================

def analyze_targets(sentence, source, target):

    if not sentence:
        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    # --------------------------------------------------------
    # Explicit against construction
    # --------------------------------------------------------

    if re.search(
        re.escape(source_l)
        + r".{0,80}\b(?:target|targets|targeted|"
        r"attacks|attacked|attack)\b.{0,50}"
        + re.escape(target_l),
        lower
    ):
        return "DIRECT"

    if re.search(
        re.escape(source_l)
        + r".{0,80}\bagainst\s+"
        + re.escape(target_l),
        lower
    ):
        return "DIRECT"

    return "UNKNOWN"


# ============================================================
# COMMUNICATION ANALYSIS
# ============================================================

def analyze_communication(sentence, source, target):

    if not sentence:
        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    pattern = (
        re.escape(source_l)
        + r".{0,60}"
        r"\b(communicate|communicates|communicated|"
        r"connect|connects|connected|contact|contacts|"
        r"beacon|beacons)\b"
        r".{0,60}"
        + re.escape(target_l)
    )

    if re.search(pattern, lower):
        return "DIRECT"

    return "UNKNOWN"


# ============================================================
# TYPE COMPATIBILITY
# ============================================================

def type_compatible(source_type, target_type, relation):

    source_type = (source_type or "").upper()
    target_type = (target_type or "").upper()

    if relation == "uses":

        allowed = {
            "GROUP": {
                "MALWARE",
                "TOOL",
                "TECHNIQUE",
                "CON"
            },

            "MALWARE": {
                "TOOL",
                "TECHNIQUE",
                "CON"
            },

            "ORG": {
                "TOOL",
                "MALWARE",
                "TECHNIQUE",
                "CON"
            },

            "TOOL": {
                "TOOL",
                "TECHNIQUE",
                "CON"
            }
        }

        return target_type in allowed.get(
            source_type,
            set()
        )

    if relation == "targets":

        return source_type in {
            "GROUP",
            "MALWARE",
            "ORG",
            "TOOL"
        }

    return True


# ============================================================
# VALIDATE RECORD
# ============================================================

def validate_record(record):

    # --------------------------------------------------------
    # SUPPORT BOTH SCHEMAS
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
        record.get(
            "original_text",
            record.get("text", "")
        )
    )

    existing_sentence = normalize(
        record.get("sentence", "")
    )

    # --------------------------------------------------------
    # Prefer extractor sentence only if it contains both
    # entities.
    # --------------------------------------------------------

    if (
        existing_sentence
        and source.lower() in existing_sentence.lower()
        and target.lower() in existing_sentence.lower()
    ):

        sentence = existing_sentence

    else:

        sentence = select_evidence_sentence(
            original_text,
            source,
            target
        )

    # --------------------------------------------------------
    # Basic evidence
    # --------------------------------------------------------

    source_present = (
        bool(sentence)
        and source.lower() in sentence.lower()
    )

    target_present = (
        bool(sentence)
        and target.lower() in sentence.lower()
    )

    trigger_info = find_trigger(
        sentence,
        relation
    )

    trigger = (
        trigger_info["text"]
        if trigger_info
        else None
    )

    trigger_present = trigger is not None

    # --------------------------------------------------------
    # Context flags
    # --------------------------------------------------------

    negated = False

    if sentence and trigger_info:

        left = max(
            0,
            trigger_info["start"] - 80
        )

        context = sentence[
            left:trigger_info["end"] + 80
        ]

        negated = any(
            re.search(
                pattern,
                context,
                re.IGNORECASE
            )
            for pattern in NEGATION_PATTERNS
        )

    comparison = (
        bool(sentence)
        and any(
            re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )
            for pattern in COMPARISON_PATTERNS
        )
    )

    reporting = (
        bool(sentence)
        and any(
            re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )
            for pattern in REPORTING_PATTERNS
        )
    )

    uncertainty = (
        bool(sentence)
        and any(
            re.search(
                pattern,
                sentence,
                re.IGNORECASE
            )
            for pattern in UNCERTAINTY_PATTERNS
        )
    )

    table_caption = is_table_or_caption(
        sentence
    )

    # --------------------------------------------------------
    # Dependency
    # --------------------------------------------------------

    dependency = dependency_relation(
        sentence,
        source,
        target,
        relation,
        trigger
    )

    # --------------------------------------------------------
    # Relation-specific semantic analysis
    # --------------------------------------------------------

    semantic = "UNKNOWN"

    if relation == "uses":

        semantic = analyze_uses(
            sentence,
            source,
            target,
            trigger
        )

    elif relation == "targets":

        semantic = analyze_targets(
            sentence,
            source,
            target
        )

    elif relation == "communicates-with":

        semantic = analyze_communication(
            sentence,
            source,
            target
        )

    # --------------------------------------------------------
    # Type
    # --------------------------------------------------------

    type_ok = type_compatible(
        source_type,
        target_type,
        relation
    )

    # --------------------------------------------------------
    # Margin
    # --------------------------------------------------------

    margin = record.get(
        "margin",
        record.get(
            "decision_margin",
            0.0
        )
    )

    try:
        margin = float(margin)
    except Exception:
        margin = 0.0

    # ========================================================
    # STATUS
    # ========================================================

    status = "UNVERIFIED"

    reasons = []

    # --------------------------------------------------------
    # Evidence availability
    # --------------------------------------------------------

    if source_present:
        reasons.append(
            "SOURCE_IN_SENTENCE"
        )

    else:
        reasons.append(
            "SOURCE_NOT_IN_SENTENCE"
        )

    if target_present:
        reasons.append(
            "TARGET_IN_SENTENCE"
        )

    else:
        reasons.append(
            "TARGET_NOT_IN_SENTENCE"
        )

    if trigger_present:
        reasons.append(
            "RELATION_TRIGGER_IN_SENTENCE"
        )

    else:
        reasons.append(
            "NO_RELATION_TRIGGER"
        )

    # --------------------------------------------------------
    # Context reasons
    # --------------------------------------------------------

    if negated:
        reasons.append(
            "NEGATED_RELATION"
        )

    if comparison:
        reasons.append(
            "COMPARISON_OR_SIMILARITY_CONTEXT"
        )

    if reporting:
        reasons.append(
            "REPORTING_OR_OBSERVER_CONTEXT"
        )

    if uncertainty:
        reasons.append(
            "UNCERTAINTY_LANGUAGE"
        )

    if table_caption:
        reasons.append(
            "TABLE_OR_CAPTION_CONTEXT"
        )

    if type_ok:
        reasons.append(
            "TYPE_COMPATIBLE"
        )

    else:
        reasons.append(
            "TYPE_INCOMPATIBLE"
        )

    # --------------------------------------------------------
    # Semantic reasons
    # --------------------------------------------------------

    if semantic == "DIRECT":
        reasons.append(
            "RELATION_SPECIFIC_DIRECT_PATTERN"
        )

    elif semantic == "PASSIVE_REVERSE":
        reasons.append(
            "PASSIVE_DIRECTION_REVERSED"
        )

    elif semantic == "COMPARISON":
        reasons.append(
            "COMPARISON_RELATION_NOT_DIRECT"
        )

    elif semantic == "CONTEXT":
        reasons.append(
            "CONTEXT_NOT_DIRECT_RELATION"
        )

    # --------------------------------------------------------
    # Dependency reasons
    # --------------------------------------------------------

    if dependency["status"] == "FORWARD":

        reasons.append(
            "DEPENDENCY_FORWARD"
        )

    elif dependency["status"] == "REVERSE":

        reasons.append(
            "DEPENDENCY_REVERSE"
        )

    else:

        reasons.append(
            "DEPENDENCY_UNRESOLVED"
        )

    # ========================================================
    # HARD REJECTION
    # ========================================================

    if table_caption:

        status = "UNSUPPORTED"

    elif negated:

        status = "UNSUPPORTED"

    elif comparison:

        status = "UNSUPPORTED"

    elif semantic in {
        "COMPARISON",
        "CONTEXT",
        "PASSIVE_REVERSE"
    }:

        status = "UNSUPPORTED"

    # ========================================================
    # STRONG SUPPORT
    # ========================================================

    elif (
        source_present
        and target_present
        and trigger_present
        and semantic == "DIRECT"
        and not reporting
        and not uncertainty
        and type_ok
        and dependency["status"] == "FORWARD"
    ):

        status = "SUPPORTED"

    # ========================================================
    # PLAUSIBLE
    # ========================================================

    elif (
        source_present
        and target_present
        and trigger_present
        and not negated
        and not comparison
        and not table_caption
    ):

        status = "PLAUSIBLE"

    # ========================================================
    # OTHERWISE
    # ========================================================

    else:

        status = "UNVERIFIED"

    # ========================================================
    # OUTPUT RECORD
    # ========================================================

    result = dict(record)

    result.update({

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

        "semantic_analysis": semantic,

        "dependency_status":
            dependency["status"],

        "dependency_structure":
            dependency["structure"],

        "dependency_score":
            dependency["score"],

        "type_compatible":
            type_ok,

        "svm_margin":
            margin,

        "status":
            status,

        "validation_reasons":
            reasons
    })

    return result


# ============================================================
# MAIN
# ============================================================

def main():

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
                records.append(
                    json.loads(line)
                )

    print(
        f"Input records: {len(records)}"
    )

    results = []

    status_counts = Counter()

    relation_status = defaultdict(
        Counter
    )

    reason_counts = Counter()

    # ========================================================
    # PROCESS
    # ========================================================

    for i, record in enumerate(records):

        try:

            result = validate_record(
                record
            )

            results.append(result)

            status_counts[
                result["status"]
            ] += 1

            relation_status[
                result["relation"]
            ][
                result["status"]
            ] += 1

            for reason in result[
                "validation_reasons"
            ]:

                reason_counts[
                    reason
                ] += 1

        except Exception as e:

            result = dict(record)

            result.update({
                "status": "UNVERIFIED",
                "validation_error": str(e)
            })

            results.append(result)

            status_counts[
                "UNVERIFIED"
            ] += 1

        if (i + 1) % 500 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(records)}"
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

    print()
    print("=" * 70)
    print("STATUS SUMMARY")
    print("=" * 70)

    total = len(results)

    for status in [
        "SUPPORTED",
        "PLAUSIBLE",
        "UNSUPPORTED",
        "UNVERIFIED"
    ]:

        count = status_counts[
            status
        ]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{status:15s}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # RELATION STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("RELATION × STATUS")
    print("=" * 70)

    for relation in sorted(
        relation_status.keys()
    ):

        c = relation_status[
            relation
        ]

        print(
            f"{relation:20s} "
            f"SUPPORTED={c['SUPPORTED']:4d} "
            f"PLAUSIBLE={c['PLAUSIBLE']:4d} "
            f"UNSUPPORTED={c['UNSUPPORTED']:4d} "
            f"UNVERIFIED={c['UNVERIFIED']:4d}"
        )

    # ========================================================
    # REASONS
    # ========================================================

    print()
    print("=" * 70)
    print("TOP VALIDATION REASONS")
    print("=" * 70)

    for reason, count in (
        reason_counts.most_common()
    ):

        print(
            f"{reason:45s}: {count}"
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

    for i, r in enumerate(
        supported[:30],
        1
    ):

        print()

        print(
            f"[{i}] "
            f"{r['source']} "
            f"--{r['relation']}--> "
            f"{r['target']}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

        print(
            f"    Semantic: "
            f"{r.get('semantic_analysis')}"
        )

        print(
            f"    Dependency: "
            f"{r.get('dependency_status')}"
        )

        print(
            f"    Margin: "
            f"{r.get('svm_margin')}"
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

    for i, r in enumerate(
        plausible[:20],
        1
    ):

        print()

        print(
            f"[{i}] "
            f"{r['source']} "
            f"--{r['relation']}--> "
            f"{r['target']}"
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

    for i, r in enumerate(
        unsupported[:20],
        1
    ):

        print()

        print(
            f"[{i}] "
            f"{r['source']} "
            f"--{r['relation']}--> "
            f"{r['target']}"
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
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print("V4 COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()