import json
import re
from collections import Counter


INPUT_FILE = "data/annoctr_relation_predictions.jsonl"
OUTPUT_FILE = "data/annoctr_relation_predictions_syntax.jsonl"

MODEL_NAME = "en_core_web_sm"

RELATIONS_TO_VALIDATE = {
    "uses",
    "targets",
    "originates-from",
    "located-at",
    "exfiltrates-to",
    "communicates-with",
    "downloads",
    "attributed-to",
}


# ============================================================
# LOAD SPACY
# ============================================================

print("=" * 70)
print("SYNTACTIC RELATION VALIDATION")
print("=" * 70)

print("\nLoading spaCy model...")

import spacy

nlp = spacy.load(MODEL_NAME)

print("spaCy model loaded successfully.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    # Markdown links:
    # [visible text](url) -> visible text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove common URL/image artifacts
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# ENTITY MATCHING
# ============================================================

def find_entity_tokens(doc, mention):

    mention_norm = normalize_text(
        mention
    )

    if not mention_norm:
        return []

    mention_words = mention_norm.split()

    matches = []

    tokens = list(doc)

    normalized_tokens = [
        normalize_text(token.text)
        for token in tokens
    ]

    for i in range(
        len(tokens) - len(mention_words) + 1
    ):

        candidate = normalized_tokens[
            i:i + len(mention_words)
        ]

        if candidate == mention_words:

            matches.append(
                list(
                    range(
                        i,
                        i + len(mention_words)
                    )
                )
            )

    return matches


# ============================================================
# SENTENCE MATCHING
# ============================================================

def same_sentence(
    source_token,
    target_token
):

    return (
        source_token.sent.start
        == target_token.sent.start
        and
        source_token.sent.end
        == target_token.sent.end
    )


# ============================================================
# DEPENDENCY PATH
# ============================================================

def get_dependency_path(
    source_token,
    target_token
):

    source_ancestors = {}

    current = source_token

    while True:

        source_ancestors[
            current.i
        ] = current

        if current.head == current:
            break

        current = current.head


    target_chain = []

    current = target_token

    while True:

        target_chain.append(
            current
        )

        if current.head == current:
            break

        current = current.head


    common = None

    for token in target_chain:

        if token.i in source_ancestors:

            common = token
            break


    if common is None:
        return []


    path = []

    current = source_token

    while current.i != common.i:

        path.append(
            (
                current,
                "UP"
            )
        )

        current = current.head


    path.append(
        (
            common,
            "COMMON"
        )
    )


    downward = []

    current = target_token

    while current.i != common.i:

        downward.append(
            (
                current,
                "DOWN"
            )
        )

        current = current.head


    path.extend(
        reversed(
            downward
        )
    )

    return path


# ============================================================
# SYNTACTIC FEATURES
# ============================================================

def extract_features(
    doc,
    source_token,
    target_token
):

    same_sent = same_sentence(
        source_token,
        target_token
    )


    distance = abs(
        source_token.i
        -
        target_token.i
    )


    source_dep = source_token.dep_

    target_dep = target_token.dep_


    source_is_subject = (
        source_dep in {
            "nsubj",
            "nsubjpass",
            "csubj",
            "csubjpass",
            "agent"
        }
    )


    target_is_object = (
        target_dep in {
            "dobj",
            "obj",
            "iobj",
            "pobj",
            "attr",
            "oprd"
        }
    )


    source_is_agent = (
        source_dep == "agent"
    )


    target_is_prep_object = (
        target_dep == "pobj"
    )


    # Is there a verb between the entities?
    start = min(
        source_token.i,
        target_token.i
    )

    end = max(
        source_token.i,
        target_token.i
    )

    verb_between = any(
        doc[i].pos_ == "VERB"
        for i in range(
            start,
            end + 1
        )
    )


    # Do they share the same syntactic head?
    shared_head = (
        source_token.head.i
        ==
        target_token.head.i
    )


    # Dependency path
    path = get_dependency_path(
        source_token,
        target_token
    )


    path_string = " -> ".join(
        f"{token.text}/{token.dep_}"
        for token, direction in path
    )


    # Does the path contain a verb?
    path_contains_verb = any(
        token.pos_ == "VERB"
        for token, direction in path
    )


    return {

        "same_sentence":
            same_sent,

        "token_distance":
            distance,

        "source_dependency":
            source_dep,

        "target_dependency":
            target_dep,

        "source_is_subject":
            source_is_subject,

        "source_is_agent":
            source_is_agent,

        "target_is_object":
            target_is_object,

        "target_is_prep_object":
            target_is_prep_object,

        "verb_between":
            verb_between,

        "shared_head":
            shared_head,

        "path_contains_verb":
            path_contains_verb,

        "dependency_path":
            path_string
    }


# ============================================================
# SYNTAX SCORE
# ============================================================

def calculate_syntax_score(
    features
):

    """
    Heuristic plausibility score.

    This is NOT a probability.

    It is only used for analysis at this stage.
    """

    score = 0.0


    # Same sentence is important.
    if features[
        "same_sentence"
    ]:

        score += 0.20


    # Subject-like source.
    if features[
        "source_is_subject"
    ]:

        score += 0.25


    # Agent-like source.
    if features[
        "source_is_agent"
    ]:

        score += 0.15


    # Object-like target.
    if features[
        "target_is_object"
    ]:

        score += 0.25


    # Prepositional object can represent a relation.
    if features[
        "target_is_prep_object"
    ]:

        score += 0.10


    # Verb between the entities.
    if features[
        "verb_between"
    ]:

        score += 0.05


    # Keep nearby entities slightly favored.
    if (
        features["token_distance"]
        <= 15
    ):

        score += 0.05


    # Dependency path contains a verb.
    if features[
        "path_contains_verb"
    ]:

        score += 0.05


    return round(
        min(
            score,
            1.0
        ),
        4
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading relation predictions...")

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
    f"Predictions loaded : "
    f"{len(records)}"
)


# ============================================================
# VALIDATION
# ============================================================

validated_records = []

relation_scores = Counter()

score_values = []

validated_count = 0
not_validated_count = 0
parse_error_count = 0
entity_not_found_count = 0


for index, record in enumerate(
    records
):

    relation = record.get(
        "predicted_relation"
    )

    # IMPORTANT:
    # The actual prediction schema uses "text"
    evidence = record.get(
        "text",
        ""
    )

    source = record.get(
        "source_mention",
        ""
    )

    target = record.get(
        "target_mention",
        ""
    )


    # --------------------------------------------------------
    # Relations outside validation set
    # --------------------------------------------------------

    if relation not in RELATIONS_TO_VALIDATE:

        record[
            "syntax_validation_status"
        ] = "not_validated"

        record[
            "syntax_score"
        ] = None

        validated_records.append(
            record
        )

        not_validated_count += 1

        continue


    # --------------------------------------------------------
    # Parse evidence
    # --------------------------------------------------------

    try:

        doc = nlp(
            evidence
        )

    except Exception:

        record[
            "syntax_validation_status"
        ] = "parse_error"

        record[
            "syntax_score"
        ] = 0.0

        validated_records.append(
            record
        )

        parse_error_count += 1

        continue


    # --------------------------------------------------------
    # Locate entities
    # --------------------------------------------------------

    source_matches = find_entity_tokens(
        doc,
        source
    )

    target_matches = find_entity_tokens(
        doc,
        target
    )


    if not source_matches:

        record[
            "syntax_validation_status"
        ] = "source_not_found"

        record[
            "syntax_score"
        ] = 0.0

        validated_records.append(
            record
        )

        entity_not_found_count += 1

        continue


    if not target_matches:

        record[
            "syntax_validation_status"
        ] = "target_not_found"

        record[
            "syntax_score"
        ] = 0.0

        validated_records.append(
            record
        )

        entity_not_found_count += 1

        continue


    # --------------------------------------------------------
    # Find best occurrence pair
    # --------------------------------------------------------

    best_score = -1
    best_features = None

    for source_indices in source_matches:

        for target_indices in target_matches:

            source_token = doc[
                source_indices[0]
            ]

            target_token = doc[
                target_indices[0]
            ]

            features = extract_features(
                doc,
                source_token,
                target_token
            )

            score = calculate_syntax_score(
                features
            )

            if score > best_score:

                best_score = score
                best_features = features


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    record[
        "syntax_score"
    ] = best_score

    record[
        "syntax_features"
    ] = best_features

    record[
        "syntax_validation_status"
    ] = "validated"


    validated_records.append(
        record
    )

    validated_count += 1

    score_values.append(
        best_score
    )

    relation_scores[
        relation
    ] += 1


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in validated_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SYNTACTIC VALIDATION RESULTS")
print("=" * 70)

print(
    f"Total predictions       : "
    f"{len(records)}"
)

print(
    f"Successfully validated  : "
    f"{validated_count}"
)

print(
    f"Not validated            : "
    f"{not_validated_count}"
)

print(
    f"Parse errors             : "
    f"{parse_error_count}"
)

print(
    f"Entity not found         : "
    f"{entity_not_found_count}"
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SYNTAX SCORE DISTRIBUTION")
print("=" * 70)

bins = [
    (0.0, 0.19),
    (0.20, 0.39),
    (0.40, 0.59),
    (0.60, 0.79),
    (0.80, 1.00),
]

for low, high in bins:

    count = sum(
        low <= score <= high
        for score in score_values
    )

    print(
        f"{low:.2f} - {high:.2f} : "
        f"{count}"
    )


# ============================================================
# RELATION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATED RELATIONS")
print("=" * 70)

for relation, count in (
    relation_scores.most_common()
):

    print(
        f"{relation:<25}"
        f"{count}"
    )


# ============================================================
# HIGH-SCORE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("HIGH SYNTAX-SCORE EXAMPLES")
print("=" * 70)

high_examples = [

    r
    for r in validated_records

    if r.get(
        "syntax_score"
    ) is not None

    and r.get(
        "syntax_score"
    ) >= 0.70
]


for record in high_examples[:20]:

    print("\n----------------------------------------")

    print(
        f"{record['source_mention']}"
        f" --"
        f"{record['predicted_relation']}"
        f"--> "
        f"{record['target_mention']}"
    )

    print(
        f"Syntax score: "
        f"{record['syntax_score']}"
    )

    print(
        f"Margin: "
        f"{record['margin']}"
    )

    print(
        f"Evidence: "
        f"{record['text']}"
    )


# ============================================================
# LOW-SCORE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("LOW SYNTAX-SCORE EXAMPLES")
print("=" * 70)

low_examples = [

    r
    for r in validated_records

    if r.get(
        "syntax_score"
    ) is not None

    and r.get(
        "syntax_score"
    ) < 0.40
]


for record in low_examples[:20]:

    print("\n----------------------------------------")

    print(
        f"{record['source_mention']}"
        f" --"
        f"{record['predicted_relation']}"
        f"--> "
        f"{record['target_mention']}"
    )

    print(
        f"Syntax score: "
        f"{record['syntax_score']}"
    )

    print(
        f"Margin: "
        f"{record['margin']}"
    )

    print(
        f"Evidence: "
        f"{record['text']}"
    )


# ============================================================
# IMPORTANT KNOWN EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("KNOWN SANITY-CHECK EXAMPLES")
print("=" * 70)

for record in validated_records:

    source = record.get(
        "source_mention"
    )

    target = record.get(
        "target_mention"
    )

    if (
        source == "Microsoft"
        and
        target == "Google"
    ):

        print("\nMicrosoft → Google")

        print(
            "Relation:",
            record.get(
                "predicted_relation"
            )
        )

        print(
            "Syntax score:",
            record.get(
                "syntax_score"
            )
        )

        print(
            "Features:",
            record.get(
                "syntax_features"
            )
        )

        print(
            "Evidence:",
            record.get(
                "text"
            )
        )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)