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
    "data/annoctr_relation_predictions_validated_v5.jsonl"
)


# ============================================================
# LOAD SPACY
# ============================================================

print("=" * 70)
print("RELATION VALIDATOR V5")
print("=" * 70)

nlp = spacy.load("en_core_web_sm")


# ============================================================
# RELATION VERBS
# ============================================================

RELATION_VERBS = {

    "uses": {
        "use", "uses", "used", "using",
        "employ", "employs", "employed",
        "deploy", "deploys", "deployed",
        "leverage", "leverages", "leveraged",
        "utilize", "utilizes", "utilized"
    },

    "targets": {
        "target", "targets", "targeted", "targeting",
        "attack", "attacks", "attacked",
        "aim", "aims", "aimed"
    },

    "communicates-with": {
        "communicate", "communicates", "communicated",
        "connect", "connects", "connected",
        "contact", "contacts",
        "beacon", "beacons"
    },

    "exploits": {
        "exploit", "exploits", "exploited", "exploiting"
    },

    "downloads": {
        "download", "downloads", "downloaded", "downloading",
        "retrieve", "retrieves",
        "fetch", "fetches"
    },

    "delivers": {
        "deliver", "delivers", "delivered", "delivering"
    },

    "drops": {
        "drop", "drops", "dropped", "dropping"
    },

    "impersonates": {
        "impersonate", "impersonates", "impersonated"
    },

    "owns": {
        "own", "owns", "owned"
    },

    "indicates": {
        "indicate", "indicates", "indicated",
        "suggest", "suggests", "suggested"
    }
}


# ============================================================
# PREPOSITIONAL RELATIONS
# ============================================================

PREPOSITION_RELATIONS = {

    "targets": {
        "against"
    },

    "communicates-with": {
        "with", "to"
    },

    "exfiltrates-to": {
        "to"
    },

    "originates-from": {
        "from", "in"
    },

    "located-at": {
        "in", "at"
    },

    "attributed-to": {
        "to"
    }
}


# ============================================================
# NEGATION
# ============================================================

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "without",
    "cannot",
    "can't",
    "doesn't",
    "don't",
    "didn't",
    "wasn't",
    "weren't",
    "isn't",
    "aren't"
}


# ============================================================
# COMPARISON
# ============================================================

COMPARISON_PATTERNS = [
    r"\bsame\b.{0,80}\bas\b",
    r"\bsimilar\b",
    r"\bcomparison\b",
    r"\bcompared\b",
    r"\bsimilarly\b",
    r"\bin the same manner\b",
    r"\bin the same way\b"
]


# ============================================================
# REPORTING / OBSERVER
# ============================================================

REPORTING_PATTERNS = [
    r"\breported by\b",
    r"\bobserved by\b",
    r"\bidentified by\b",
    r"\bfound by\b",
    r"\baccording to\b",
    r"\bresearchers\b",
    r"\banalysts\b",
    r"\binvestigators\b"
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
    r"\bunclear\b",
    r"\buncertain\b",
    r"\bno evidence\b",
    r"\bcannot independently confirm\b"
]


# ============================================================
# HELPERS
# ============================================================

def normalize(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def contains_pattern(text, patterns):

    if not text:
        return False

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):
            return True

    return False


def entity_in_text(entity, text):

    if not entity or not text:
        return False

    return (
        entity.lower()
        in
        text.lower()
    )


# ============================================================
# SENTENCE SELECTION
# ============================================================

def select_sentence(text, source, target):

    if not text:
        return None

    doc = nlp(text)

    source_l = source.lower()
    target_l = target.lower()

    # Best case: both candidates in same sentence
    for sent in doc.sents:

        s = sent.text.strip()
        sl = s.lower()

        if (
            source_l in sl
            and target_l in sl
        ):
            return s

    return None


# ============================================================
# FIND RELATION VERB
# ============================================================

def find_relation_verbs(doc, relation):

    allowed = RELATION_VERBS.get(
        relation,
        set()
    )

    matches = []

    for token in doc:

        if token.lemma_.lower() in allowed:

            matches.append(token)

    return matches


# ============================================================
# ENTITY TOKEN MATCHING
# ============================================================

def entity_tokens(doc, entity):

    entity_words = entity.lower().split()

    matches = []

    for i in range(
        len(doc)
    ):

        window = doc[
            i:i + len(entity_words)
        ]

        words = [
            token.text.lower()
            for token in window
        ]

        if words == entity_words:

            matches.extend(
                window
            )

    return matches


# ============================================================
# FIND SUBJECTS OF PREDICATE
# ============================================================

def get_subjects(predicate):

    subjects = []

    for child in predicate.children:

        if child.dep_ in {
            "nsubj",
            "nsubjpass",
            "csubj"
        }:

            subjects.append(child)

    return subjects


# ============================================================
# FIND OBJECTS OF PREDICATE
# ============================================================

def get_objects(predicate):

    objects = []

    for child in predicate.children:

        if child.dep_ in {
            "dobj",
            "obj",
            "attr",
            "oprd"
        }:

            objects.append(child)

    return objects


# ============================================================
# FIND PREPOSITIONAL OBJECTS
# ============================================================

def get_prepositional_objects(
    predicate,
    allowed_prepositions
):

    results = []

    for child in predicate.children:

        if child.dep_ == "prep":

            prep = child.text.lower()

            if prep not in allowed_prepositions:
                continue

            for grandchild in child.children:

                if grandchild.dep_ == "pobj":

                    results.append(
                        (
                            prep,
                            grandchild
                        )
                    )

    return results


# ============================================================
# TOKEN BELONGS TO ENTITY
# ============================================================

def token_matches_entity(
    token,
    entity_tokens_list
):

    for entity_token in entity_tokens_list:

        if token.i == entity_token.i:
            return True

        # descendants / compounds
        subtree = list(
            token.subtree
        )

        if entity_token in subtree:
            return True

    return False


# ============================================================
# DEPENDENCY ARGUMENT ANALYSIS
# ============================================================

def analyze_predicate(
    sentence,
    source,
    target,
    relation
):

    if not sentence:

        return {
            "status": "NO_SENTENCE",
            "structure": None,
            "predicate": None,
            "source_argument": False,
            "target_argument": False,
            "score": 0.0
        }

    doc = nlp(sentence)

    source_tokens = entity_tokens(
        doc,
        source
    )

    target_tokens = entity_tokens(
        doc,
        target
    )

    if not source_tokens or not target_tokens:

        return {
            "status": "ENTITIES_NOT_TOKENIZED",
            "structure": None,
            "predicate": None,
            "source_argument": False,
            "target_argument": False,
            "score": 0.0
        }

    predicates = find_relation_verbs(
        doc,
        relation
    )

    if not predicates:

        return {
            "status": "NO_PREDICATE",
            "structure": None,
            "predicate": None,
            "source_argument": False,
            "target_argument": False,
            "score": 0.0
        }

    best = None

    # --------------------------------------------------------
    # Evaluate every possible predicate
    # --------------------------------------------------------

    for predicate in predicates:

        subjects = get_subjects(
            predicate
        )

        objects = get_objects(
            predicate
        )

        source_is_subject = any(
            token_matches_entity(
                s,
                source_tokens
            )
            for s in subjects
        )

        target_is_object = any(
            token_matches_entity(
                o,
                target_tokens
            )
            for o in objects
        )

        # ----------------------------------------------------
        # Passive construction
        #
        # AsyncRAT is used by cybercriminals
        #
        # target = passive subject
        # source = agent
        # ----------------------------------------------------

        target_is_passive_subject = any(
            token_matches_entity(
                s,
                target_tokens
            )
            and s.dep_ == "nsubjpass"
            for s in subjects
        )

        source_is_agent = False

        for child in predicate.children:

            if child.dep_ == "agent":

                for agent_child in child.children:

                    if token_matches_entity(
                        agent_child,
                        source_tokens
                    ):
                        source_is_agent = True

        # Some spaCy parses make the "by" noun
        # a pobj rather than an agent.
        for child in predicate.children:

            if child.dep_ == "prep" and child.text.lower() == "by":

                for pobj in child.children:

                    if pobj.dep_ == "pobj":

                        if token_matches_entity(
                            pobj,
                            source_tokens
                        ):
                            source_is_agent = True

        # ----------------------------------------------------
        # Prepositional relation
        # ----------------------------------------------------

        prep_results = get_prepositional_objects(
            predicate,
            PREPOSITION_RELATIONS.get(
                relation,
                set()
            )
        )

        target_is_prep_object = False
        prep_used = None

        for prep, pobj in prep_results:

            if token_matches_entity(
                pobj,
                target_tokens
            ):

                target_is_prep_object = True
                prep_used = prep

        # ----------------------------------------------------
        # Score candidate predicate
        # ----------------------------------------------------

        score = 0.0
        status = "UNKNOWN"
        structure = None

        if (
            source_is_subject
            and target_is_object
        ):

            score = 1.0
            status = "FORWARD"
            structure = "SUBJECT_OBJECT"

        elif (
            target_is_passive_subject
            and source_is_agent
        ):

            score = 1.0
            status = "PASSIVE_REVERSE"
            structure = "PASSIVE_AGENT"

        elif (
            source_is_subject
            and target_is_prep_object
        ):

            score = 0.9
            status = "FORWARD_PREPOSITION"
            structure = (
                "SUBJECT_PREPOSITION_OBJECT"
                f"_{prep_used}"
            )

        elif target_is_prep_object:

            score = 0.6
            status = "PREPOSITION_ONLY"
            structure = (
                f"PREPOSITION_{prep_used}"
            )

        if best is None or score > best["score"]:

            best = {
                "status": status,
                "structure": structure,
                "predicate": predicate.text,
                "source_argument":
                    source_is_subject
                    or source_is_agent,
                "target_argument":
                    target_is_object
                    or target_is_prep_object
                    or target_is_passive_subject,
                "score": score,
                "predicate_index":
                    predicate.i
            }

    return best


# ============================================================
# SPECIAL USES SEMANTICS
# ============================================================

def analyze_uses_context(
    sentence,
    source,
    target
):

    if not sentence:

        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    # --------------------------------------------------------
    # Explicit passive:
    #
    # X is used by Y
    # --------------------------------------------------------

    passive = re.search(
        re.escape(source_l)
        + r".{0,80}"
        r"\b(?:is|was|are|were|has been|have been)"
        r"\s+used\b"
        r".{0,50}"
        r"\bby\s+"
        + re.escape(target_l),
        lower
    )

    if passive:

        return "PASSIVE_REVERSE"

    # --------------------------------------------------------
    # "used in ransomware incidents"
    # --------------------------------------------------------

    if re.search(
        r"\bused\b.{0,50}\bin\s+"
        + re.escape(target_l),
        lower
    ):

        return "CONTEXT"

    # --------------------------------------------------------
    # Same X as Y
    # --------------------------------------------------------

    if re.search(
        r"\bsame\b.{0,80}\bas\s+"
        + re.escape(target_l),
        lower
    ):

        return "COMPARISON"

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    if (
        "similar" in lower
        and target_l in lower
    ):

        return "COMPARISON"

    return "UNKNOWN"


# ============================================================
# NEGATION AROUND PREDICATE
# ============================================================

def predicate_negated(
    doc,
    predicate
):

    # Direct dependency negation
    for child in predicate.children:

        if child.dep_ == "neg":

            return True

    # Nearby textual negation
    start = max(
        0,
        predicate.i - 4
    )

    end = min(
        len(doc),
        predicate.i + 5
    )

    window = doc[
        start:end
    ]

    for token in window:

        if token.text.lower() in NEGATION_WORDS:

            return True

    return False


# ============================================================
# TYPE COMPATIBILITY
# ============================================================

def type_compatible(
    source_type,
    target_type,
    relation
):

    source_type = (
        source_type or ""
    ).upper()

    target_type = (
        target_type or ""
    ).upper()

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
# VALIDATE ONE RECORD
# ============================================================

def validate_record(record):

    # --------------------------------------------------------
    # FIELD COMPATIBILITY
    # --------------------------------------------------------

    source = record.get(
        "source",
        record.get(
            "source_mention",
            ""
        )
    )

    target = record.get(
        "target",
        record.get(
            "target_mention",
            ""
        )
    )

    relation = record.get(
        "relation",
        record.get(
            "predicted_relation",
            ""
        )
    )

    source_type = record.get(
        "source_type",
        ""
    )

    target_type = record.get(
        "target_type",
        ""
    )

    text = normalize(
        record.get(
            "original_text",
            record.get(
                "text",
                ""
            )
        )
    )

    existing_sentence = normalize(
        record.get(
            "sentence",
            ""
        )
    )

    # --------------------------------------------------------
    # Evidence sentence
    # --------------------------------------------------------

    if (
        existing_sentence
        and entity_in_text(
            source,
            existing_sentence
        )
        and entity_in_text(
            target,
            existing_sentence
        )
    ):

        sentence = existing_sentence

    else:

        sentence = select_sentence(
            text,
            source,
            target
        )

    # --------------------------------------------------------
    # Basic evidence
    # --------------------------------------------------------

    source_present = entity_in_text(
        source,
        sentence
    )

    target_present = entity_in_text(
        target,
        sentence
    )

    # --------------------------------------------------------
    # Dependency / predicate analysis
    # --------------------------------------------------------

    dependency = analyze_predicate(
        sentence,
        source,
        target,
        relation
    )

    # --------------------------------------------------------
    # Negation
    # --------------------------------------------------------

    negated = False

    if sentence:

        doc = nlp(sentence)

        predicates = find_relation_verbs(
            doc,
            relation
        )

        for predicate in predicates:

            if predicate_negated(
                doc,
                predicate
            ):

                negated = True
                break

    # --------------------------------------------------------
    # Comparison / reporting / uncertainty
    # --------------------------------------------------------

    comparison = contains_pattern(
        sentence,
        COMPARISON_PATTERNS
    )

    reporting = contains_pattern(
        sentence,
        REPORTING_PATTERNS
    )

    uncertainty = contains_pattern(
        sentence,
        UNCERTAINTY_PATTERNS
    )

    table = (
        "|"
        in
        (sentence or "")
    )

    # --------------------------------------------------------
    # Relation-specific analysis
    # --------------------------------------------------------

    semantic = "UNKNOWN"

    if relation == "uses":

        semantic = analyze_uses_context(
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
    # REASONS
    # ========================================================

    reasons = []

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

    if dependency["predicate"]:

        reasons.append(
            "RELATION_PREDICATE_FOUND"
        )

    else:

        reasons.append(
            "NO_RELATION_PREDICATE"
        )

    if dependency["status"] == "FORWARD":

        reasons.append(
            "PREDICATE_SUBJECT_OBJECT_MATCH"
        )

    elif dependency["status"] == "PASSIVE_REVERSE":

        reasons.append(
            "PASSIVE_ARGUMENT_REVERSAL"
        )

    elif dependency["status"] == "FORWARD_PREPOSITION":

        reasons.append(
            "PREDICATE_PREPOSITION_ARGUMENT_MATCH"
        )

    else:

        reasons.append(
            "PREDICATE_ARGUMENTS_UNRESOLVED"
        )

    if type_ok:

        reasons.append(
            "TYPE_COMPATIBLE"
        )

    else:

        reasons.append(
            "TYPE_INCOMPATIBLE"
        )

    if negated:

        reasons.append(
            "NEGATED_RELATION"
        )

    if comparison:

        reasons.append(
            "COMPARISON_CONTEXT"
        )

    if reporting:

        reasons.append(
            "REPORTING_CONTEXT"
        )

    if uncertainty:

        reasons.append(
            "UNCERTAINTY_CONTEXT"
        )

    if table:

        reasons.append(
            "TABLE_CONTEXT"
        )

    if semantic == "DIRECT":

        reasons.append(
            "DIRECT_USE_CONTEXT"
        )

    elif semantic == "PASSIVE_REVERSE":

        reasons.append(
            "PASSIVE_USE_CONTEXT"
        )

    elif semantic == "COMPARISON":

        reasons.append(
            "COMPARISON_USE_CONTEXT"
        )

    elif semantic == "CONTEXT":

        reasons.append(
            "CONTEXT_USE_NOT_RELATION"
        )

    # ========================================================
    # STATUS DECISION
    # ========================================================

    status = "UNVERIFIED"

    # --------------------------------------------------------
    # Hard negative cases
    # --------------------------------------------------------

    if table:

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

    # --------------------------------------------------------
    # Strong predicate-centered support
    # --------------------------------------------------------

    elif (
        dependency["status"] == "FORWARD"
        and dependency["source_argument"]
        and dependency["target_argument"]
        and source_present
        and target_present
        and type_ok
        and not reporting
        and not uncertainty
    ):

        # For USES specifically, require an actual
        # predicate argument match.
        if relation == "uses":

            status = "SUPPORTED"

        else:

            status = "SUPPORTED"

    # --------------------------------------------------------
    # Prepositional direct relation
    # --------------------------------------------------------

    elif (
        dependency["status"]
        == "FORWARD_PREPOSITION"
        and source_present
        and target_present
        and not negated
        and not comparison
        and not table
    ):

        status = "PLAUSIBLE"

    # --------------------------------------------------------
    # General evidence
    # --------------------------------------------------------

    elif (
        source_present
        and target_present
        and dependency["predicate"]
        and not negated
        and not comparison
        and not table
    ):

        status = "PLAUSIBLE"

    else:

        status = "UNVERIFIED"

    # ========================================================
    # OUTPUT
    # ========================================================

    result = dict(record)

    result.update({

        "source":
            source,

        "target":
            target,

        "relation":
            relation,

        "evidence_sentence":
            sentence,

        "predicate":
            dependency["predicate"],

        "predicate_status":
            dependency["status"],

        "predicate_structure":
            dependency["structure"],

        "predicate_score":
            dependency["score"],

        "source_argument":
            dependency["source_argument"],

        "target_argument":
            dependency["target_argument"],

        "semantic_analysis":
            semantic,

        "negated":
            negated,

        "comparison_context":
            comparison,

        "reporting_context":
            reporting,

        "uncertainty_context":
            uncertainty,

        "table_context":
            table,

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
            f"Input not found: {INPUT_PATH}"
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

    predicate_counts = Counter()

    # ========================================================
    # PROCESS
    # ========================================================

    for i, record in enumerate(
        records
    ):

        try:

            result = validate_record(
                record
            )

            results.append(
                result
            )

            status_counts[
                result["status"]
            ] += 1

            relation_status[
                result["relation"]
            ][
                result["status"]
            ] += 1

            predicate_counts[
                result["predicate_status"]
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

                "status":
                    "UNVERIFIED",

                "validation_error":
                    str(e)
            })

            results.append(
                result
            )

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
    # STATUS SUMMARY
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

    for relation in sorted(
        relation_status
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
    # PREDICATE STATUS
    # ========================================================

    print()
    print("=" * 70)
    print("PREDICATE ANALYSIS")
    print("=" * 70)

    for key, count in (
        predicate_counts.most_common()
    ):

        print(
            f"{key:35s}: {count}"
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
    # SUPPORTED
    # ========================================================

    print()
    print("=" * 70)
    print("SUPPORTED EXAMPLES")
    print("=" * 70)

    supported = [
        r
        for r in results
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
            f"    Predicate: "
            f"{r.get('predicate')}"
        )

        print(
            f"    Structure: "
            f"{r.get('predicate_structure')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

        print(
            f"    Margin: "
            f"{r.get('svm_margin')}"
        )

    # ========================================================
    # PLAUSIBLE
    # ========================================================

    print()
    print("=" * 70)
    print("PLAUSIBLE EXAMPLES")
    print("=" * 70)

    plausible = [
        r
        for r in results
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
            f"    Predicate: "
            f"{r.get('predicate')}"
        )

        print(
            f"    Structure: "
            f"{r.get('predicate_structure')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
        )

    # ========================================================
    # UNSUPPORTED
    # ========================================================

    print()
    print("=" * 70)
    print("UNSUPPORTED EXAMPLES")
    print("=" * 70)

    unsupported = [
        r
        for r in results
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
            f"    Predicate: "
            f"{r.get('predicate')}"
        )

        print(
            f"    Structure: "
            f"{r.get('predicate_structure')}"
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
    print("V5 COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()