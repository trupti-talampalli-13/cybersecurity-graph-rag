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
    "data/annoctr_relation_predictions_validated_v6.jsonl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("RELATION VALIDATOR V6")
print("=" * 70)

nlp = spacy.load("en_core_web_sm")


# ============================================================
# RELATION VOCABULARY
# ============================================================

RELATION_VERBS = {

    "uses": {
        "use",
        "uses",
        "used",
        "using",
        "employ",
        "employs",
        "employed",
        "deploy",
        "deploys",
        "deployed",
        "leverage",
        "leverages",
        "leveraged",
        "utilize",
        "utilizes",
        "utilized"
    },

    "targets": {
        "target",
        "targets",
        "targeted",
        "targeting",
        "attack",
        "attacks",
        "attacked",
        "attacking"
    },

    "communicates-with": {
        "communicate",
        "communicates",
        "communicated",
        "connect",
        "connects",
        "connected",
        "contact",
        "contacts",
        "contacted",
        "beacon",
        "beacons"
    },

    "exploits": {
        "exploit",
        "exploits",
        "exploited",
        "exploiting"
    },

    "downloads": {
        "download",
        "downloads",
        "downloaded",
        "downloading",
        "retrieve",
        "retrieves",
        "retrieved",
        "fetch",
        "fetches",
        "fetched"
    },

    "delivers": {
        "deliver",
        "delivers",
        "delivered",
        "delivering"
    },

    "drops": {
        "drop",
        "drops",
        "dropped",
        "dropping"
    },

    "impersonates": {
        "impersonate",
        "impersonates",
        "impersonated"
    },

    "owns": {
        "own",
        "owns",
        "owned"
    },

    "indicates": {
        "indicate",
        "indicates",
        "indicated",
        "suggest",
        "suggests",
        "suggested"
    }
}


# ============================================================
# PREPOSITION RELATIONS
# ============================================================

PREPOSITION_RELATIONS = {

    "targets": {
        "against",
        "at"
    },

    "communicates-with": {
        "with",
        "to"
    },

    "exfiltrates-to": {
        "to"
    },

    "originates-from": {
        "from",
        "in"
    },

    "located-at": {
        "in",
        "at"
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
# CONTEXT PATTERNS
# ============================================================

COMPARISON_PATTERNS = [
    r"\bsame\b.{0,100}\bas\b",
    r"\bsimilar\b",
    r"\bsimilarly\b",
    r"\bcomparison\b",
    r"\bcompared\b",
    r"\bin the same manner\b",
    r"\bin the same way\b",
    r"\banalogous\b"
]

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
    r"\buncertain\b"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def contains_any(text, patterns):

    if not text:
        return False

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return True

    return False


def entity_in_text(entity, text):

    if not entity or not text:
        return False

    return (
        entity.lower()
        in text.lower()
    )


# ============================================================
# ENTITY SPANS
# ============================================================

def find_entity_spans(doc, entity):

    """
    Find exact token spans corresponding to an entity mention.

    Unlike the previous versions, this does not reduce an
    entity to a single token.
    """

    if not entity:
        return []

    target_words = [
        x.lower()
        for x in entity.split()
    ]

    spans = []

    for i in range(
        len(doc)
    ):

        if i + len(target_words) > len(doc):
            break

        window = doc[
            i:i + len(target_words)
        ]

        words = [
            token.text.lower()
            for token in window
        ]

        if words == target_words:

            spans.append(
                window
            )

    return spans


# ============================================================
# TOKEN/SPAN MATCH
# ============================================================

def span_matches(
    span,
    entity_spans
):

    if not span:
        return False

    span_indices = {
        token.i
        for token in span
    }

    for entity_span in entity_spans:

        entity_indices = {
            token.i
            for token in entity_span
        }

        if span_indices & entity_indices:
            return True

    return False


# ============================================================
# EXPAND ONLY COORDINATION
# ============================================================

def expand_coordination(token):

    """
    Expand:

        Microsoft and Google

    but NOT:

        individuals employed at defense companies

    This is critical.
    """

    tokens = {token.i}

    # coordinated siblings
    for child in token.children:

        if child.dep_ == "conj":

            tokens.add(child.i)

    # token itself may be a conjunction child
    if token.dep_ == "conj":

        tokens.add(token.head.i)

        for sibling in token.head.children:

            if sibling.dep_ == "conj":
                tokens.add(sibling.i)

    return sorted(tokens)


# ============================================================
# ARGUMENT HEAD MATCH
# ============================================================

def argument_matches_entity_head(
    argument_token,
    entity_spans
):

    """
    Match the candidate entity against the ACTUAL argument
    head or its coordination.

    We deliberately do NOT search the entire subtree.
    """

    candidate_indices = set(
        expand_coordination(
            argument_token
        )
    )

    for entity_span in entity_spans:

        entity_indices = {
            token.i
            for token in entity_span
        }

        if entity_indices & candidate_indices:
            return True

    return False


# ============================================================
# DIRECT SUBJECTS
# ============================================================

def get_direct_subjects(
    predicate
):

    return [
        child
        for child in predicate.children
        if child.dep_ in {
            "nsubj",
            "nsubjpass",
            "csubj"
        }
    ]


# ============================================================
# DIRECT OBJECTS
# ============================================================

def get_direct_objects(
    predicate
):

    return [
        child
        for child in predicate.children
        if child.dep_ in {
            "obj",
            "dobj",
            "attr",
            "oprd"
        }
    ]


# ============================================================
# PASSIVE AGENT
# ============================================================

def get_passive_agents(
    predicate
):

    agents = []

    for child in predicate.children:

        # Universal Dependencies / spaCy style
        if child.dep_ == "agent":

            for grandchild in child.children:

                if grandchild.dep_ == "pobj":

                    agents.append(
                        grandchild
                    )

        # Some spaCy parses represent:
        # used by X
        # directly as prep -> pobj.
        if (
            child.dep_ == "prep"
            and child.text.lower() == "by"
        ):

            for grandchild in child.children:

                if grandchild.dep_ == "pobj":

                    agents.append(
                        grandchild
                    )

    return agents


# ============================================================
# PREPOSITIONAL OBJECTS
# ============================================================

def get_prepositional_objects(
    predicate,
    allowed_prepositions
):

    results = []

    for child in predicate.children:

        if child.dep_ != "prep":
            continue

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
# FIND SENTENCE
# ============================================================

def find_best_sentence(
    text,
    source,
    target,
    existing_sentence=None
):

    existing_sentence = normalize(
        existing_sentence
    )

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

        return existing_sentence

    if not text:
        return None

    doc = nlp(text)

    best = None

    for sent in doc.sents:

        sentence = sent.text.strip()

        source_found = entity_in_text(
            source,
            sentence
        )

        target_found = entity_in_text(
            target,
            sentence
        )

        if (
            source_found
            and target_found
        ):

            # Prefer shorter sentence containing both
            if (
                best is None
                or len(sentence) < len(best)
            ):

                best = sentence

    return best


# ============================================================
# FIND PREDICATE
# ============================================================

def find_predicates(
    doc,
    relation
):

    allowed = RELATION_VERBS.get(
        relation,
        set()
    )

    predicates = []

    for token in doc:

        lemma = token.lemma_.lower()

        if lemma not in allowed:
            continue

        if token.pos_ not in {
            "VERB",
            "AUX"
        }:

            continue

        predicates.append(
            token
        )

    return predicates


# ============================================================
# RELATIVE CLAUSE CHECK
# ============================================================

def relative_clause_info(
    predicate
):

    """
    Detect:

        threat actor that deploys Conti

    The subject of deploys is the relative
    clause's antecedent, not the previous arbitrary entity.
    """

    info = {
        "relative_clause": False,
        "antecedent": None
    }

    if predicate.dep_ != "relcl":
        return info

    info["relative_clause"] = True

    # The head of the relative clause is its antecedent.
    info["antecedent"] = predicate.head

    return info


# ============================================================
# PREDICATE ARGUMENT ANALYSIS
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
            "predicate": None,
            "structure": None,
            "score": 0.0,
            "source_argument": False,
            "target_argument": False,
            "predicate_index": None,
            "argument_details": {}
        }

    doc = nlp(sentence)

    source_spans = find_entity_spans(
        doc,
        source
    )

    target_spans = find_entity_spans(
        doc,
        target
    )

    if not source_spans or not target_spans:

        return {
            "status": "ENTITIES_NOT_TOKENIZED",
            "predicate": None,
            "structure": None,
            "score": 0.0,
            "source_argument": False,
            "target_argument": False,
            "predicate_index": None,
            "argument_details": {}
        }

    predicates = find_predicates(
        doc,
        relation
    )

    if not predicates:

        return {
            "status": "NO_PREDICATE",
            "predicate": None,
            "structure": None,
            "score": 0.0,
            "source_argument": False,
            "target_argument": False,
            "predicate_index": None,
            "argument_details": {}
        }

    best = None

    # ========================================================
    # CHECK EACH PREDICATE
    # ========================================================

    for predicate in predicates:

        subjects = get_direct_subjects(
            predicate
        )

        objects = get_direct_objects(
            predicate
        )

        agents = get_passive_agents(
            predicate
        )

        prep_objects = get_prepositional_objects(
            predicate,
            PREPOSITION_RELATIONS.get(
                relation,
                set()
            )
        )

        source_subject = any(
            argument_matches_entity_head(
                subject,
                source_spans
            )
            for subject in subjects
        )

        target_object = any(
            argument_matches_entity_head(
                obj,
                target_spans
            )
            for obj in objects
        )

        source_agent = any(
            argument_matches_entity_head(
                agent,
                source_spans
            )
            for agent in agents
        )

        target_passive_subject = any(
            (
                subject.dep_ == "nsubjpass"
                and argument_matches_entity_head(
                    subject,
                    target_spans
                )
            )
            for subject in subjects
        )

        target_preposition = None

        for prep, pobj in prep_objects:

            if argument_matches_entity_head(
                pobj,
                target_spans
            ):

                target_preposition = prep

                break

        # ----------------------------------------------------
        # Relative clause
        # ----------------------------------------------------

        rel_info = relative_clause_info(
            predicate
        )

        # ----------------------------------------------------
        # Decide structure
        # ----------------------------------------------------

        status = "UNKNOWN"
        structure = None
        score = 0.0

        # Active:
        #
        # TA456 targets defense
        #
        if (
            source_subject
            and target_object
        ):

            status = "FORWARD"
            structure = "DIRECT_SUBJECT_OBJECT"
            score = 1.0

        # Passive:
        #
        # NetWireRC is used by APT actors
        #
        elif (
            target_passive_subject
            and source_agent
        ):

            status = "PASSIVE_FORWARD"
            structure = "PASSIVE_TARGET_AGENT"
            score = 1.0

        # Subject + explicit preposition:
        #
        # actor communicates with server
        #
        elif (
            source_subject
            and target_preposition
        ):

            status = "FORWARD_PREPOSITION"
            structure = (
                "SUBJECT_PREPOSITION_OBJECT"
            )
            score = 0.95

        # Explicit passive agent but candidate direction
        # is reversed.
        elif (
            source_subject
            and any(
                argument_matches_entity_head(
                    agent,
                    target_spans
                )
                for agent in agents
            )
        ):

            status = "PASSIVE_REVERSE"
            structure = "SUBJECT_PASSIVE_AGENT"
            score = 0.95

        # Candidate is passive subject while target is
        # the agent.
        elif (
            target_passive_subject
            and any(
                argument_matches_entity_head(
                    agent,
                    target_spans
                )
                for agent in agents
            )
        ):

            status = "PASSIVE_REVERSE"
            structure = "PASSIVE_SUBJECT_AGENT"
            score = 0.95

        # Preposition exists but source isn't predicate subject.
        elif target_preposition:

            status = "PREPOSITION_ONLY"
            structure = (
                f"PREPOSITION_{target_preposition}"
            )
            score = 0.50

        # ----------------------------------------------------
        # Keep best result
        # ----------------------------------------------------

        candidate = {

            "status": status,

            "predicate":
                predicate.text,

            "structure":
                structure,

            "score":
                score,

            "source_argument":
                source_subject or source_agent,

            "target_argument":
                target_object
                or target_passive_subject
                or bool(target_preposition),

            "predicate_index":
                predicate.i,

            "relative_clause":
                rel_info["relative_clause"],

            "relative_antecedent":
                (
                    rel_info["antecedent"].text
                    if rel_info["antecedent"]
                    else None
                ),

            "argument_details": {

                "subjects": [
                    x.text
                    for x in subjects
                ],

                "objects": [
                    x.text
                    for x in objects
                ],

                "agents": [
                    x.text
                    for x in agents
                ],

                "prepositional_objects": [
                    {
                        "preposition": prep,
                        "object": pobj.text
                    }
                    for prep, pobj
                    in prep_objects
                ]
            }
        }

        if (
            best is None
            or candidate["score"]
            > best["score"]
        ):

            best = candidate

    return best


# ============================================================
# RELATION-SPECIFIC SEMANTIC CHECK
# ============================================================

def semantic_relation_check(
    sentence,
    source,
    target,
    relation
):

    if not sentence:

        return "UNKNOWN"

    lower = sentence.lower()

    source_l = source.lower()
    target_l = target.lower()

    # --------------------------------------------------------
    # USES
    # --------------------------------------------------------

    if relation == "uses":

        # "X is used by Y"
        reverse_passive = re.search(
            re.escape(source_l)
            + r".{0,80}"
            + r"\b(?:is|was|are|were|has been|have been)"
            + r"\s+used\b"
            + r".{0,80}"
            + r"\bby\s+"
            + re.escape(target_l),
            lower
        )

        if reverse_passive:

            return "PASSIVE_REVERSE"

        # "used in ransomware incidents"
        contextual_use = re.search(
            r"\bused\b.{0,50}\bin\s+"
            + re.escape(target_l),
            lower
        )

        if contextual_use:

            return "CONTEXT"

        # "same ... as"
        if re.search(
            r"\bsame\b.{0,100}\bas\s+"
            + re.escape(target_l),
            lower
        ):

            return "COMPARISON"

    return "UNKNOWN"


# ============================================================
# NEGATION
# ============================================================

def has_predicate_negation(
    doc,
    predicate
):

    for child in predicate.children:

        if child.dep_ == "neg":
            return True

    # textual fallback
    start = max(
        0,
        predicate.i - 4
    )

    end = min(
        len(doc),
        predicate.i + 5
    )

    for token in doc[
        start:end
    ]:

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
                "MALWARE",
                "TOOL",
                "TECHNIQUE",
                "CON"
            },

            "TOOL": {
                "TOOL",
                "TECHNIQUE",
                "CON"
            }
        }

        return (
            target_type
            in
            allowed.get(
                source_type,
                set()
            )
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
            "text",
            record.get(
                "original_text",
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

    # ========================================================
    # SENTENCE
    # ========================================================

    sentence = find_best_sentence(
        text,
        source,
        target,
        existing_sentence
    )

    source_present = entity_in_text(
        source,
        sentence
    )

    target_present = entity_in_text(
        target,
        sentence
    )

    # ========================================================
    # DEPENDENCY
    # ========================================================

    dependency = analyze_predicate(
        sentence,
        source,
        target,
        relation
    )

    # ========================================================
    # PREDICATE NEGATION
    # ========================================================

    negated = False

    if sentence:

        doc = nlp(sentence)

        predicates = find_predicates(
            doc,
            relation
        )

        for predicate in predicates:

            if has_predicate_negation(
                doc,
                predicate
            ):

                negated = True
                break

    # ========================================================
    # CONTEXT
    # ========================================================

    comparison = contains_any(
        sentence,
        COMPARISON_PATTERNS
    )

    reporting = contains_any(
        sentence,
        REPORTING_PATTERNS
    )

    uncertainty = contains_any(
        sentence,
        UNCERTAINTY_PATTERNS
    )

    table = (
        "|" in (sentence or "")
    )

    semantic = semantic_relation_check(
        sentence,
        source,
        target,
        relation
    )

    type_ok = type_compatible(
        source_type,
        target_type,
        relation
    )

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
            "DIRECT_PREDICATE_ARGUMENT_MATCH"
        )

    elif dependency["status"] == "PASSIVE_FORWARD":

        reasons.append(
            "PASSIVE_AGENT_ARGUMENT_MATCH"
        )

    elif dependency["status"] == "FORWARD_PREPOSITION":

        reasons.append(
            "DIRECT_PREPOSITION_ARGUMENT_MATCH"
        )

    elif dependency["status"] == "PASSIVE_REVERSE":

        reasons.append(
            "PASSIVE_DIRECTION_REVERSED"
        )

    else:

        reasons.append(
            "DIRECT_ARGUMENTS_UNRESOLVED"
        )

    if type_ok:

        reasons.append(
            "TYPE_COMPATIBLE"
        )

    else:

        reasons.append(
            "TYPE_INCOMPATIBLE"
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

    if negated:

        reasons.append(
            "NEGATED_RELATION"
        )

    if semantic == "PASSIVE_REVERSE":

        reasons.append(
            "PASSIVE_USE_CONTEXT"
        )

    elif semantic == "CONTEXT":

        reasons.append(
            "USE_CONTEXT_NOT_RELATION"
        )

    elif semantic == "COMPARISON":

        reasons.append(
            "COMPARISON_USE_CONTEXT"
        )

    # ========================================================
    # STATUS
    # ========================================================

    status = "UNVERIFIED"

    # --------------------------------------------------------
    # Hard negatives
    # --------------------------------------------------------

    if table:

        status = "UNSUPPORTED"

    elif negated:

        status = "UNSUPPORTED"

    elif comparison:

        status = "UNSUPPORTED"

    elif semantic in {
        "PASSIVE_REVERSE",
        "CONTEXT",
        "COMPARISON"
    }:

        status = "UNSUPPORTED"

    elif (
        dependency["status"]
        == "PASSIVE_REVERSE"
    ):

        status = "UNSUPPORTED"

    # --------------------------------------------------------
    # High precision direct relation
    # --------------------------------------------------------

    elif (
        dependency["status"]
        == "FORWARD"
        and dependency["source_argument"]
        and dependency["target_argument"]
        and source_present
        and target_present
        and type_ok
        and not reporting
        and not uncertainty
    ):

        status = "SUPPORTED"

    # --------------------------------------------------------
    # Direct preposition
    # --------------------------------------------------------

    elif (
        dependency["status"]
        == "FORWARD_PREPOSITION"
        and source_present
        and target_present
        and not reporting
        and not uncertainty
    ):

        status = "SUPPORTED"

    # --------------------------------------------------------
    # Passive forward
    # --------------------------------------------------------

    elif (
        dependency["status"]
        == "PASSIVE_FORWARD"
        and source_present
        and target_present
        and type_ok
        and not reporting
        and not uncertainty
    ):

        status = "SUPPORTED"

    # --------------------------------------------------------
    # Predicate found but argument uncertain
    # --------------------------------------------------------

    elif (
        source_present
        and target_present
        and dependency["predicate"]
        and not comparison
        and not table
    ):

        status = "PLAUSIBLE"

    # --------------------------------------------------------
    # Otherwise
    # --------------------------------------------------------

    else:

        status = "UNVERIFIED"

    # ========================================================
    # OUTPUT RECORD
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

        "predicate_index":
            dependency["predicate_index"],

        "predicate_argument_details":
            dependency["argument_details"],

        "relative_clause":
            dependency.get(
                "relative_clause",
                False
            ),

        "relative_antecedent":
            dependency.get(
                "relative_antecedent"
            ),

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
            result["status"]
        ] += 1

        relation_status[
            result.get(
                "relation",
                ""
            )
        ][
            result["status"]
        ] += 1

        predicate_counts[
            result.get(
                "predicate_status",
                "ERROR"
            )
        ] += 1

        for reason in result.get(
            "validation_reasons",
            []
        ):

            reason_counts[
                reason
            ] += 1

        if (
            (i + 1) % 500 == 0
        ):

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
    print("PREDICATE STATUS")
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
    print("TOP REASONS")
    print("=" * 70)

    for reason, count in (
        reason_counts.most_common(25)
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
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
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
            f"    Arguments: "
            f"{r.get('predicate_argument_details')}"
        )

        print(
            f"    Evidence: "
            f"{r.get('evidence_sentence')}"
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
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
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
            f"{r.get('source')} "
            f"--{r.get('relation')}--> "
            f"{r.get('target')}"
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
    print("V6 COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()