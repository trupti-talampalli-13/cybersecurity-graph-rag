import json
import re
from pathlib import Path
from collections import Counter

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/annoctr_relation_predictions.jsonl")
OUTPUT_FILE = Path("data/annoctr_relation_predictions_linguistic.jsonl")

MODEL_NAME = "en_core_web_sm"

# Only relations currently produced by the SVM that we want
# to analyze linguistically.
SUPPORTED_RELATIONS = {
    "uses",
    "targets",
    "originates-from",
    "located-at",
    "exfiltrates-to",
    "communicates-with",
    "downloads",
    "attributed-to",
    "delivers",
    "impersonates",
    "variant-of",
    "exploits",
}


# ============================================================
# RELATION-SPECIFIC TRIGGERS
# ============================================================

RELATION_TRIGGERS = {
    "uses": [
        "use",
        "uses",
        "used",
        "using",
        "utilize",
        "utilizes",
        "utilized",
        "utilizing",
        "employ",
        "employs",
        "employed",
        "leverages",
        "leveraged",
        "deploy",
        "deploys",
        "deployed",
        "execute",
        "executes",
        "executed",
        "load",
        "loads",
        "loaded",
        "run",
        "runs",
        "ran",
        "runs",
    ],

    "targets": [
        "target",
        "targets",
        "targeted",
        "targeting",
        "attack",
        "attacks",
        "attacked",
        "attacking",
        "victimize",
        "victimizes",
        "victimized",
        "focus",
        "focuses",
        "focused",
        "focusing",
        "aim",
        "aims",
        "aimed",
    ],

    "originates-from": [
        "originates",
        "originated",
        "originating",
        "origin",
        "comes",
        "came",
        "based",
        "associated",
        "linked",
        "originates from",
        "originated from",
        "based in",
        "based from",
        "associated with",
        "linked to",
    ],

    "located-at": [
        "located",
        "locates",
        "based",
        "resides",
        "operates",
        "headquartered",
        "located in",
        "based in",
        "resides in",
        "operates in",
        "headquartered in",
    ],

    "downloads": [
        "download",
        "downloads",
        "downloaded",
        "downloading",
        "retrieve",
        "retrieves",
        "retrieved",
        "retrieving",
        "fetch",
        "fetches",
        "fetched",
        "fetching",
        "obtain",
        "obtains",
        "obtained",
        "obtaining",
    ],

    "communicates-with": [
        "communicate",
        "communicates",
        "communicated",
        "communicating",
        "connect",
        "connects",
        "connected",
        "connecting",
        "contact",
        "contacts",
        "contacted",
        "contacting",
        "talk",
        "talks",
        "talked",
        "talking",
    ],

    "exfiltrates-to": [
        "exfiltrate",
        "exfiltrates",
        "exfiltrated",
        "exfiltrating",
        "steal",
        "steals",
        "stole",
        "stolen",
        "upload",
        "uploads",
        "uploaded",
        "uploading",
        "transfer",
        "transfers",
        "transferred",
        "transferring",
        "send",
        "sends",
        "sent",
        "sending",
    ],

    "attributed-to": [
        "attribute",
        "attributes",
        "attributed",
        "attributing",
        "associate",
        "associates",
        "associated",
        "associating",
        "link",
        "links",
        "linked",
        "linking",
        "believe",
        "believes",
        "believed",
        "suspect",
        "suspects",
        "suspected",
    ],

    "delivers": [
        "deliver",
        "delivers",
        "delivered",
        "delivering",
        "drop",
        "drops",
        "dropped",
        "dropping",
    ],

    "impersonates": [
        "impersonate",
        "impersonates",
        "impersonated",
        "impersonating",
        "masquerade",
        "masquerades",
        "masqueraded",
        "masquerading",
    ],

    "variant-of": [
        "variant",
        "variants",
        "variation",
        "variations",
        "version",
        "versions",
        "variant of",
        "version of",
    ],

    "exploits": [
        "exploit",
        "exploits",
        "exploited",
        "exploiting",
        "abuse",
        "abuses",
        "abused",
        "abusing",
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def find_entity_spans(doc, entity_text):
    """
    Find occurrences of entity_text in the spaCy document.

    Returns:
        list of (start_token, end_token)
    """

    entity_text = normalize_text(entity_text)

    if not entity_text:
        return []

    spans = []

    for start in range(len(doc)):
        for end in range(start + 1, len(doc) + 1):

            candidate = normalize_text(
                doc[start:end].text
            )

            if candidate == entity_text:
                spans.append((start, end))

            # Once the candidate is already much longer than
            # the target, stop expanding.
            if len(candidate) > len(entity_text) + 20:
                break

    return spans


def token_span_to_text(doc, span):
    start, end = span
    return doc[start:end].text


# ============================================================
# ENTITY LOCATION
# ============================================================

def locate_entities(doc, source_text, target_text):
    """
    Locate source and target mentions in the parsed sentence.

    Returns:
        source_span
        target_span
        entity_found
    """

    source_spans = find_entity_spans(doc, source_text)
    target_spans = find_entity_spans(doc, target_text)

    if not source_spans or not target_spans:
        return None, None, False

    # Prefer the closest source-target pair.
    best_pair = None
    best_distance = float("inf")

    for source_span in source_spans:
        for target_span in target_spans:

            source_start = source_span[0]
            target_start = target_span[0]

            # Do not require a particular ordering.
            distance = abs(source_start - target_start)

            if distance < best_distance:
                best_distance = distance
                best_pair = (
                    source_span,
                    target_span
                )

    return best_pair[0], best_pair[1], True


# ============================================================
# SENTENCE INFORMATION
# ============================================================

def get_sentence_for_span(doc, start, end):

    for sent in doc.sents:

        if start >= sent.start and end <= sent.end:
            return sent

    return None


# ============================================================
# TRIGGER DETECTION
# ============================================================

def find_relation_triggers(sentence, relation):

    triggers = RELATION_TRIGGERS.get(relation, [])

    found = []

    sentence_lower = normalize_text(sentence.text)

    for trigger in triggers:

        trigger_lower = trigger.lower()

        if trigger_lower in sentence_lower:

            found.append(trigger)

    return sorted(set(found))


def find_trigger_tokens(sentence, relation):

    trigger_words = {
        t.lower()
        for t in RELATION_TRIGGERS.get(relation, [])
        if " " not in t
    }

    found = []

    for token in sentence:

        if token.text.lower() in trigger_words:

            found.append({
                "text": token.text,
                "lemma": token.lemma_,
                "dep": token.dep_,
                "pos": token.pos_,
                "index": token.i,
            })

    return found


# ============================================================
# DEPENDENCY FEATURES
# ============================================================

def get_entity_head(doc, span):

    start, end = span

    # For multi-token entities, use the syntactic root
    # inside the span when possible.
    tokens = list(doc[start:end])

    if len(tokens) == 1:
        return tokens[0]

    for token in tokens:

        if token.head.i < start or token.head.i >= end:
            return token

    return tokens[0]


def dependency_path(source_token, target_token):

    source_ancestors = [source_token] + list(source_token.ancestors)
    target_ancestors = [target_token] + list(target_token.ancestors)

    source_map = {
        token.i: token
        for token in source_ancestors
    }

    common = None

    for token in target_ancestors:

        if token.i in source_map:
            common = token
            break

    if common is None:
        return []

    source_path = []
    token = source_token

    while token.i != common.i:
        source_path.append(token)
        token = token.head

    source_path.append(common)

    target_path = []
    token = target_token

    while token.i != common.i:
        target_path.append(token)
        token = token.head

    target_path.append(common)

    full_path = source_path + list(reversed(target_path[:-1]))

    return full_path


def path_to_string(path):

    parts = []

    for token in path:

        parts.append(
            f"{token.text}/{token.dep_}/{token.pos_}"
        )

    return " -> ".join(parts)


def path_contains_verb(path):

    return any(
        token.pos_ in {"VERB", "AUX"}
        for token in path
    )


def path_contains_subject(path):

    return any(
        token.dep_ in {
            "nsubj",
            "nsubjpass",
            "csubj",
            "agent",
        }
        for token in path
    )


def path_contains_object(path):

    return any(
        token.dep_ in {
            "dobj",
            "obj",
            "pobj",
            "iobj",
        }
        for token in path
    )


def path_contains_preposition(path):

    return any(
        token.dep_ == "prep"
        for token in path
    )


# ============================================================
# SOURCE / TARGET GRAMMATICAL ROLES
# ============================================================

def grammatical_roles(source_token, target_token):

    source_roles = []
    target_roles = []

    important_deps = {
        "nsubj",
        "nsubjpass",
        "dobj",
        "obj",
        "pobj",
        "iobj",
        "prep",
        "agent",
        "attr",
        "appos",
        "conj",
        "compound",
        "poss",
    }

    if source_token.dep_ in important_deps:
        source_roles.append(source_token.dep_)

    if target_token.dep_ in important_deps:
        target_roles.append(target_token.dep_)

    return source_roles, target_roles


# ============================================================
# RELATION-SPECIFIC LINGUISTIC SCORE
# ============================================================

def compute_linguistic_score(
    relation,
    trigger_found,
    trigger_is_path_related,
    source_is_subject,
    target_is_object,
    passive,
    path_has_verb,
    path_has_prep,
    distance,
):
    """
    This is intentionally an interpretable heuristic.

    IMPORTANT:
    This score is for analysis only.
    It is NOT used to filter the KG.
    """

    score = 0.0

    # Strongest signal:
    # relation-specific trigger exists.
    if trigger_found:
        score += 0.30

    # Trigger lies on / near dependency path.
    if trigger_is_path_related:
        score += 0.20

    # Source behaves like grammatical subject.
    if source_is_subject:
        score += 0.15

    # Target behaves like grammatical object.
    if target_is_object:
        score += 0.15

    # A verb connects the two entities.
    if path_has_verb:
        score += 0.10

    # Prepositional construction can be meaningful
    # for several CTI relations.
    if path_has_prep:
        score += 0.05

    # Shorter dependency distance is better.
    if distance <= 2:
        score += 0.05
    elif distance <= 4:
        score += 0.03

    # Passive voice can be important for
    # relations such as targets / attributed-to.
    if passive and relation in {
        "targets",
        "attributed-to",
        "delivers",
        "exploits",
    }:
        score += 0.05

    return min(score, 1.0)


# ============================================================
# ANALYZE ONE RECORD
# ============================================================

def analyze_record(record, nlp):

    text = record.get("text", "")

    source = record.get("source_mention", "")
    target = record.get("target_mention", "")

    relation = record.get("predicted_relation", "")

    result = dict(record)

    result["linguistic_analysis"] = {
        "parsed": False,
        "entity_found": False,
        "source_found": False,
        "target_found": False,
        "source_token": None,
        "target_token": None,
        "source_dep": None,
        "target_dep": None,
        "source_pos": None,
        "target_pos": None,
        "source_is_subject": False,
        "target_is_object": False,
        "passive_voice": False,
        "same_sentence": False,
        "token_distance": None,
        "dependency_distance": None,
        "dependency_path": [],
        "dependency_path_text": "",
        "path_has_verb": False,
        "path_has_subject": False,
        "path_has_object": False,
        "path_has_preposition": False,
        "relation_triggers": [],
        "trigger_tokens": [],
        "trigger_on_dependency_path": False,
        "linguistic_score": 0.0,
    }

    if not text:
        return result

    try:
        doc = nlp(text)
        result["linguistic_analysis"]["parsed"] = True
    except Exception:
        return result

    source_span, target_span, found = locate_entities(
        doc,
        source,
        target
    )

    if not found:
        return result

    result["linguistic_analysis"]["entity_found"] = True
    result["linguistic_analysis"]["source_found"] = True
    result["linguistic_analysis"]["target_found"] = True

    source_token = get_entity_head(doc, source_span)
    target_token = get_entity_head(doc, target_span)

    source_sent = get_sentence_for_span(
        doc,
        source_span[0],
        source_span[1]
    )

    target_sent = get_sentence_for_span(
        doc,
        target_span[0],
        target_span[1]
    )

    if source_sent is None or target_sent is None:
        return result

    same_sentence = (
        source_sent.start == target_sent.start
    )

    result["linguistic_analysis"]["same_sentence"] = same_sentence

    # If entities occur in different sentences, a direct
    # dependency relation cannot exist.
    if not same_sentence:
        result["linguistic_analysis"]["linguistic_score"] = 0.0
        return result

    distance = abs(
        source_token.i - target_token.i
    )

    result["linguistic_analysis"]["token_distance"] = distance

    source_dep_distance = len(
        list(source_token.ancestors)
    )

    target_dep_distance = len(
        list(target_token.ancestors)
    )

    path = dependency_path(
        source_token,
        target_token
    )

    dependency_distance = len(path)

    result["linguistic_analysis"]["dependency_distance"] = (
        dependency_distance
    )

    result["linguistic_analysis"]["dependency_path"] = [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "dep": token.dep_,
            "pos": token.pos_,
            "index": token.i,
        }
        for token in path
    ]

    result["linguistic_analysis"]["dependency_path_text"] = (
        path_to_string(path)
    )

    path_has_verb = path_contains_verb(path)
    path_has_subject = path_contains_subject(path)
    path_has_object = path_contains_object(path)
    path_has_prep = path_contains_preposition(path)

    result["linguistic_analysis"]["path_has_verb"] = (
        path_has_verb
    )

    result["linguistic_analysis"]["path_has_subject"] = (
        path_has_subject
    )

    result["linguistic_analysis"]["path_has_object"] = (
        path_has_object
    )

    result["linguistic_analysis"]["path_has_preposition"] = (
        path_has_prep
    )

    result["linguistic_analysis"]["source_token"] = (
        source_token.text
    )

    result["linguistic_analysis"]["target_token"] = (
        target_token.text
    )

    result["linguistic_analysis"]["source_dep"] = (
        source_token.dep_
    )

    result["linguistic_analysis"]["target_dep"] = (
        target_token.dep_
    )

    result["linguistic_analysis"]["source_pos"] = (
        source_token.pos_
    )

    result["linguistic_analysis"]["target_pos"] = (
        target_token.pos_
    )

    source_roles, target_roles = grammatical_roles(
        source_token,
        target_token
    )

    source_is_subject = any(
        role in {
            "nsubj",
            "nsubjpass",
            "csubj",
            "agent",
        }
        for role in source_roles
    )

    target_is_object = any(
        role in {
            "dobj",
            "obj",
            "pobj",
            "iobj",
        }
        for role in target_roles
    )

    result["linguistic_analysis"]["source_is_subject"] = (
        source_is_subject
    )

    result["linguistic_analysis"]["target_is_object"] = (
        target_is_object
    )

    passive = any(
        token.dep_ == "nsubjpass"
        for token in source_sent
    )

    result["linguistic_analysis"]["passive_voice"] = passive

    relation_triggers = find_relation_triggers(
        source_sent,
        relation
    )

    result["linguistic_analysis"]["relation_triggers"] = (
        relation_triggers
    )

    trigger_tokens = find_trigger_tokens(
        source_sent,
        relation
    )

    result["linguistic_analysis"]["trigger_tokens"] = (
        trigger_tokens
    )

    path_indices = {
        token.i
        for token in path
    }

    trigger_on_path = any(
        item["index"] in path_indices
        for item in trigger_tokens
    )

    result["linguistic_analysis"]["trigger_on_dependency_path"] = (
        trigger_on_path
    )

    score = compute_linguistic_score(
        relation=relation,
        trigger_found=len(relation_triggers) > 0,
        trigger_is_path_related=trigger_on_path,
        source_is_subject=source_is_subject,
        target_is_object=target_is_object,
        passive=passive,
        path_has_verb=path_has_verb,
        path_has_prep=path_has_prep,
        distance=distance,
    )

    result["linguistic_analysis"]["linguistic_score"] = round(
        score,
        4
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RELATION-SPECIFIC LINGUISTIC ANALYSIS")
    print("=" * 70)

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"Loading spaCy model: {MODEL_NAME}")

    nlp = spacy.load(MODEL_NAME)

    # Disable components that are not required.
    # We need parser + tokenizer.
    enabled = [
        pipe
        for pipe in nlp.pipe_names
        if pipe not in {
            "ner",
            "textcat",
            "textcat_multilabel",
        }
    ]

    print("spaCy pipeline:", nlp.pipe_names)
    print()

    records = load_jsonl(INPUT_FILE)

    print(f"Predictions loaded: {len(records)}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    parsed_count = 0
    entity_found_count = 0
    same_sentence_count = 0

    score_bins = Counter()
    relation_counts = Counter()
    trigger_counts = Counter()

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    for index, record in enumerate(records, start=1):

        result = analyze_record(
            record,
            nlp
        )

        results.append(result)

        analysis = result["linguistic_analysis"]

        if analysis["parsed"]:
            parsed_count += 1

        if analysis["entity_found"]:
            entity_found_count += 1

        if analysis["same_sentence"]:
            same_sentence_count += 1

        relation = record.get(
            "predicted_relation",
            "UNKNOWN"
        )

        relation_counts[relation] += 1

        if analysis["relation_triggers"]:

            trigger_counts[relation] += 1

        score = analysis["linguistic_score"]

        if score < 0.20:
            score_bins["0.00-0.19"] += 1

        elif score < 0.40:
            score_bins["0.20-0.39"] += 1

        elif score < 0.60:
            score_bins["0.40-0.59"] += 1

        elif score < 0.80:
            score_bins["0.60-0.79"] += 1

        else:
            score_bins["0.80-1.00"] += 1

        if index % 500 == 0:

            print(
                f"Processed {index}/{len(records)}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Predictions              : {len(records)}"
    )

    print(
        f"Parsed                   : {parsed_count}"
    )

    print(
        f"Entities located         : {entity_found_count}"
    )

    print(
        f"Same-sentence pairs      : {same_sentence_count}"
    )

    print()
    print("Linguistic score bins:")

    for bucket in [
        "0.00-0.19",
        "0.20-0.39",
        "0.40-0.59",
        "0.60-0.79",
        "0.80-1.00",
    ]:

        print(
            f"  {bucket}: "
            f"{score_bins[bucket]}"
        )

    print()
    print("Relation counts:")

    for relation, count in relation_counts.most_common():

        trigger_count = trigger_counts[relation]

        percentage = (
            100 * trigger_count / count
            if count
            else 0
        )

        print(
            f"  {relation:20s} "
            f"{count:5d} predictions | "
            f"{trigger_count:5d} with trigger "
            f"({percentage:5.1f}%)"
        )

    print()
    print(
        f"Saved linguistic analysis to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()