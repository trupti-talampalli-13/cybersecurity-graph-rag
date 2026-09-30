import json
from pathlib import Path
from collections import Counter

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/annoctr_relation_predictions_linguistic.jsonl"
)

OUTPUT_FILE = Path(
    "data/annoctr_relation_predictions_arguments.jsonl"
)

MODEL_NAME = "en_core_web_sm"


# ============================================================
# RELATION TRIGGERS
# ============================================================

RELATION_TRIGGERS = {
    "uses": {
        "use", "uses", "used", "using",
        "utilize", "utilizes", "utilized", "utilizing",
        "employ", "employs", "employed",
        "leverage", "leverages", "leveraged",
        "deploy", "deploys", "deployed",
        "execute", "executes", "executed",
        "load", "loads", "loaded",
        "run", "runs", "ran"
    },

    "targets": {
        "target", "targets", "targeted", "targeting",
        "attack", "attacks", "attacked", "attacking",
        "victimize", "victimizes", "victimized",
        "focus", "focuses", "focused",
        "aim", "aims", "aimed"
    },

    "originates-from": {
        "originate", "originates", "originated",
        "come", "comes", "came",
        "base", "based",
        "associate", "associated",
        "link", "linked"
    },

    "located-at": {
        "locate", "located",
        "base", "based",
        "reside", "resides",
        "operate", "operates",
        "headquarter", "headquartered"
    },

    "downloads": {
        "download", "downloads", "downloaded",
        "retrieve", "retrieves", "retrieved",
        "fetch", "fetches", "fetched",
        "obtain", "obtains", "obtained"
    },

    "communicates-with": {
        "communicate", "communicates", "communicated",
        "connect", "connects", "connected",
        "contact", "contacts", "contacted",
        "talk", "talks", "talked"
    },

    "exfiltrates-to": {
        "exfiltrate", "exfiltrates", "exfiltrated",
        "steal", "steals", "stole", "stolen",
        "upload", "uploads", "uploaded",
        "transfer", "transfers", "transferred",
        "send", "sends", "sent"
    },

    "attributed-to": {
        "attribute", "attributes", "attributed",
        "associate", "associates", "associated",
        "link", "links", "linked",
        "believe", "believes", "believed",
        "suspect", "suspects", "suspected"
    },

    "delivers": {
        "deliver", "delivers", "delivered",
        "drop", "drops", "dropped"
    },

    "impersonates": {
        "impersonate", "impersonates", "impersonated",
        "masquerade", "masquerades", "masqueraded"
    },

    "variant-of": {
        "variant", "variants",
        "variation", "variations",
        "version", "versions"
    },

    "exploits": {
        "exploit", "exploits", "exploited",
        "abuse", "abuses", "abused"
    }
}


# ============================================================
# LOAD
# ============================================================

def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    return " ".join(
        text.lower().split()
    )


# ============================================================
# ENTITY MATCHING
# ============================================================

def find_entity_tokens(doc, entity_text):

    target = normalize(entity_text)

    if not target:
        return []

    target_tokens = target.split()

    matches = []

    for i in range(
        len(doc) - len(target_tokens) + 1
    ):

        candidate = [
            doc[j].text.lower()
            for j in range(
                i,
                i + len(target_tokens)
            )
        ]

        if candidate == target_tokens:

            matches.append(
                doc[i:i + len(target_tokens)]
            )

    return matches


def choose_entity_span(
    doc,
    entity_text,
    preferred_sentence=None
):

    matches = find_entity_tokens(
        doc,
        entity_text
    )

    if not matches:
        return None

    if preferred_sentence is not None:

        same_sentence = [
            span
            for span in matches
            if (
                span.start >= preferred_sentence.start
                and span.end <= preferred_sentence.end
            )
        ]

        if same_sentence:
            return same_sentence[0]

    return matches[0]


# ============================================================
# HEAD OF ENTITY
# ============================================================

def get_head(span):

    if span is None:
        return None

    tokens = list(span)

    if len(tokens) == 1:
        return tokens[0]

    token_indices = {
        token.i
        for token in tokens
    }

    # Find the syntactic root inside the entity.
    for token in tokens:

        if token.head.i not in token_indices:
            return token

    return tokens[0]


# ============================================================
# SENTENCE
# ============================================================

def get_sentence(doc, token):

    if token is None:
        return None

    for sent in doc.sents:

        if (
            token.i >= sent.start
            and token.i < sent.end
        ):
            return sent

    return None


# ============================================================
# FIND TRIGGERS
# ============================================================

def find_triggers(sentence, relation):

    if sentence is None:
        return []

    allowed = RELATION_TRIGGERS.get(
        relation,
        set()
    )

    triggers = []

    for token in sentence:

        if token.lemma_.lower() in allowed:
            triggers.append(token)

        elif token.text.lower() in allowed:
            triggers.append(token)

    # Remove duplicate token indices.
    unique = {}

    for token in triggers:
        unique[token.i] = token

    return list(unique.values())


# ============================================================
# DEPENDENCY RELATIONSHIP
# ============================================================

def get_subjects(trigger):

    subjects = []

    for child in trigger.children:

        if child.dep_ in {
            "nsubj",
            "nsubjpass",
            "csubj",
        }:

            subjects.append(child)

    # Handle passive constructions:
    #
    # Malware was used BY attackers
    #
    # The attacker is connected through "agent".
    for child in trigger.children:

        if child.dep_ == "agent":

            for grandchild in child.children:

                if grandchild.dep_ == "pobj":
                    subjects.append(grandchild)

    return subjects


def get_objects(trigger):

    objects = []

    for child in trigger.children:

        if child.dep_ in {
            "dobj",
            "obj",
            "iobj",
        }:

            objects.append(child)

        elif child.dep_ == "attr":

            objects.append(child)

        elif child.dep_ == "oprd":

            objects.append(child)

    return objects


def get_prepositional_objects(trigger):

    results = []

    for child in trigger.children:

        if child.dep_ != "prep":
            continue

        for grandchild in child.children:

            if grandchild.dep_ == "pobj":

                results.append(
                    {
                        "preposition": child.text,
                        "token": grandchild
                    }
                )

    return results


# ============================================================
# ANCESTOR / DESCENDANT SEARCH
# ============================================================

def find_nearest_relation_token(
    token,
    relation_triggers,
    max_distance=20
):

    if token is None:
        return None

    best = None
    best_distance = float("inf")

    for trigger in relation_triggers:

        distance = abs(
            token.i - trigger.i
        )

        if distance <= max_distance:

            if distance < best_distance:

                best = trigger
                best_distance = distance

    return best


# ============================================================
# ENTITY MATCHING
# ============================================================

def token_matches_span(
    token,
    span
):

    if span is None:
        return False

    return (
        token.i >= span.start
        and token.i < span.end
    )


def find_matching_entity(
    token,
    source_span,
    target_span
):

    if token_matches_span(
        token,
        source_span
    ):
        return "source"

    if token_matches_span(
        token,
        target_span
    ):
        return "target"

    return None


# ============================================================
# EXTRACT ARGUMENT
# ============================================================

def describe_token(token):

    if token is None:
        return None

    return {
        "text": token.text,
        "lemma": token.lemma_,
        "dep": token.dep_,
        "pos": token.pos_,
        "index": token.i,
    }


# ============================================================
# ANALYZE ONE RECORD
# ============================================================

def analyze_record(record, nlp):

    result = dict(record)

    source_text = record.get(
        "source_mention",
        ""
    )

    target_text = record.get(
        "target_mention",
        ""
    )

    relation = record.get(
        "predicted_relation",
        ""
    )

    text = record.get(
        "text",
        ""
    )

    analysis = record.get(
        "linguistic_analysis",
        {}
    )

    argument_info = {

        "trigger_found": False,

        "triggers": [],

        "selected_trigger": None,

        "trigger_lemma": None,

        "trigger_dep": None,

        "trigger_pos": None,

        "trigger_subjects": [],

        "trigger_objects": [],

        "trigger_prepositional_objects": [],

        "source_matches_trigger_subject": False,

        "source_matches_trigger_object": False,

        "source_matches_trigger_agent": False,

        "target_matches_trigger_subject": False,

        "target_matches_trigger_object": False,

        "target_matches_trigger_agent": False,

        "relation_direction": "unknown",

        "source_role": "none",

        "target_role": "none",

        "passive_construction": False,

        "argument_structure": "none",

        "argument_distance": None,

        "argument_dependency_path": [],

    }

    result["relation_arguments"] = argument_info

    if not text:
        return result

    try:

        doc = nlp(text)

    except Exception:

        return result

    # --------------------------------------------------------
    # Locate entities
    # --------------------------------------------------------

    source_span = choose_entity_span(
        doc,
        source_text
    )

    target_span = choose_entity_span(
        doc,
        target_text
    )

    if source_span is None or target_span is None:
        return result

    source_head = get_head(
        source_span
    )

    target_head = get_head(
        target_span
    )

    source_sentence = get_sentence(
        doc,
        source_head
    )

    target_sentence = get_sentence(
        doc,
        target_head
    )

    if (
        source_sentence is None
        or target_sentence is None
        or source_sentence.start != target_sentence.start
    ):
        return result

    sentence = source_sentence

    # --------------------------------------------------------
    # Find relation triggers
    # --------------------------------------------------------

    triggers = find_triggers(
        sentence,
        relation
    )

    if not triggers:
        return result

    argument_info["trigger_found"] = True

    argument_info["triggers"] = [
        describe_token(token)
        for token in triggers
    ]

    # --------------------------------------------------------
    # Select best trigger
    #
    # Prefer the trigger closest to the source-target
    # dependency structure.
    # --------------------------------------------------------

    pair_center = (
        source_head.i + target_head.i
    ) / 2

    selected_trigger = min(
        triggers,
        key=lambda token: abs(
            token.i - pair_center
        )
    )

    argument_info["selected_trigger"] = (
        describe_token(
            selected_trigger
        )
    )

    argument_info["trigger_lemma"] = (
        selected_trigger.lemma_
    )

    argument_info["trigger_dep"] = (
        selected_trigger.dep_
    )

    argument_info["trigger_pos"] = (
        selected_trigger.pos_
    )

    # --------------------------------------------------------
    # Find grammatical arguments
    # --------------------------------------------------------

    subjects = get_subjects(
        selected_trigger
    )

    objects = get_objects(
        selected_trigger
    )

    prepositional_objects = (
        get_prepositional_objects(
            selected_trigger
        )
    )

    argument_info["trigger_subjects"] = [
        describe_token(token)
        for token in subjects
    ]

    argument_info["trigger_objects"] = [
        describe_token(token)
        for token in objects
    ]

    argument_info[
        "trigger_prepositional_objects"
    ] = [

        {
            "preposition": item[
                "preposition"
            ],
            "token": describe_token(
                item["token"]
            )
        }

        for item in prepositional_objects
    ]

    # --------------------------------------------------------
    # Match source and target against arguments
    # --------------------------------------------------------

    for subject in subjects:

        role = find_matching_entity(
            subject,
            source_span,
            target_span
        )

        if role == "source":

            argument_info[
                "source_matches_trigger_subject"
            ] = True

            argument_info[
                "source_role"
            ] = "subject"

        elif role == "target":

            argument_info[
                "target_matches_trigger_subject"
            ] = True

            argument_info[
                "target_role"
            ] = "subject"

    for obj in objects:

        role = find_matching_entity(
            obj,
            source_span,
            target_span
        )

        if role == "source":

            argument_info[
                "source_matches_trigger_object"
            ] = True

            argument_info[
                "source_role"
            ] = "object"

        elif role == "target":

            argument_info[
                "target_matches_trigger_object"
            ] = True

            argument_info[
                "target_role"
            ] = "object"

    # --------------------------------------------------------
    # Passive agent
    # --------------------------------------------------------

    for child in selected_trigger.children:

        if child.dep_ == "agent":

            for grandchild in child.children:

                if grandchild.dep_ != "pobj":
                    continue

                role = find_matching_entity(
                    grandchild,
                    source_span,
                    target_span
                )

                if role == "source":

                    argument_info[
                        "source_matches_trigger_agent"
                    ] = True

                    argument_info[
                        "source_role"
                    ] = "agent"

                elif role == "target":

                    argument_info[
                        "target_matches_trigger_agent"
                    ] = True

                    argument_info[
                        "target_role"
                    ] = "agent"

    # --------------------------------------------------------
    # Passive construction
    # --------------------------------------------------------

    argument_info[
        "passive_construction"
    ] = any(
        token.dep_ == "nsubjpass"
        for token in sentence
    )

    # --------------------------------------------------------
    # Determine argument structure
    # --------------------------------------------------------

    source_role = argument_info[
        "source_role"
    ]

    target_role = argument_info[
        "target_role"
    ]

    if (
        source_role == "subject"
        and target_role == "object"
    ):

        argument_info[
            "argument_structure"
        ] = "SOURCE_SUBJECT_TARGET_OBJECT"

        argument_info[
            "relation_direction"
        ] = "forward"

    elif (
        source_role == "agent"
        and target_role == "subject"
    ):

        argument_info[
            "argument_structure"
        ] = "SOURCE_AGENT_TARGET_PASSIVE_SUBJECT"

        argument_info[
            "relation_direction"
        ] = "forward"

    elif (
        source_role == "object"
        and target_role == "subject"
    ):

        argument_info[
            "argument_structure"
        ] = "TARGET_SUBJECT_SOURCE_OBJECT"

        argument_info[
            "relation_direction"
        ] = "reverse"

    elif (
        source_role == "object"
        and target_role == "agent"
    ):

        argument_info[
            "argument_structure"
        ] = "SOURCE_OBJECT_TARGET_AGENT"

        argument_info[
            "relation_direction"
        ] = "reverse"

    elif source_role != "none" and target_role != "none":

        argument_info[
            "argument_structure"
        ] = (
            f"{source_role.upper()}_"
            f"{target_role.upper()}"
        )

    # --------------------------------------------------------
    # Distance between trigger and arguments
    # --------------------------------------------------------

    distances = []

    for token in [
        source_head,
        target_head
    ]:

        distances.append(
            abs(
                token.i
                - selected_trigger.i
            )
        )

    argument_info[
        "argument_distance"
    ] = max(distances)

    # --------------------------------------------------------
    # Dependency path from source to trigger
    # --------------------------------------------------------

    def dependency_path_to_ancestor(
        token,
        ancestor
    ):

        path = []

        current = token

        visited = set()

        while (
            current is not None
            and current.i not in visited
        ):

            visited.add(current.i)

            path.append(
                current
            )

            if current.i == ancestor.i:
                break

            current = current.head

        return path

    source_path = dependency_path_to_ancestor(
        source_head,
        selected_trigger
    )

    target_path = dependency_path_to_ancestor(
        target_head,
        selected_trigger
    )

    argument_info[
        "argument_dependency_path"
    ] = {

        "source_to_trigger": [
            describe_token(token)
            for token in source_path
        ],

        "target_to_trigger": [
            describe_token(token)
            for token in target_path
        ]
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("RELATION ARGUMENT EXTRACTION")
    print("=" * 75)

    print(
        f"Input : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(
        f"Loading spaCy model: {MODEL_NAME}"
    )

    nlp = spacy.load(
        MODEL_NAME
    )

    records = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(records)}"
    )

    print()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    trigger_count = Counter()
    direction_count = Counter()
    structure_count = Counter()

    successful = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, record in enumerate(
        records,
        start=1
    ):

        result = analyze_record(
            record,
            nlp
        )

        results.append(result)

        info = result[
            "relation_arguments"
        ]

        if info["trigger_found"]:

            successful += 1

            relation = record.get(
                "predicted_relation",
                "UNKNOWN"
            )

            trigger_count[
                relation
            ] += 1

            direction_count[
                info["relation_direction"]
            ] += 1

            structure_count[
                info["argument_structure"]
            ] += 1

        if index % 500 == 0:

            print(
                f"Processed "
                f"{index}/{len(records)}"
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
    print("=" * 75)
    print("EXTRACTION COMPLETE")
    print("=" * 75)

    print(
        f"Total records       : {len(records)}"
    )

    print(
        f"Trigger found       : {successful}"
    )

    print(
        f"Trigger not found   : "
        f"{len(records) - successful}"
    )

    print()
    print("Relation trigger counts:")

    for relation, count in (
        trigger_count.most_common()
    ):

        print(
            f"  {relation:20s}: {count}"
        )

    print()
    print("Relation directions:")

    for direction, count in (
        direction_count.most_common()
    ):

        print(
            f"  {direction:35s}: {count}"
        )

    print()
    print("Argument structures:")

    for structure, count in (
        structure_count.most_common(30)
    ):

        print(
            f"  {structure:45s}: {count}"
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()