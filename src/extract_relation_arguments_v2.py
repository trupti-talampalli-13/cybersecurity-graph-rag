import json
import re
import spacy
from collections import Counter

INPUT_FILE = "data/annoctr_relation_predictions_linguistic.jsonl"
OUTPUT_FILE = "data/annoctr_relation_predictions_arguments_v2.jsonl"

print("=" * 75)
print("RELATION ARGUMENT EXTRACTION V2")
print("=" * 75)
print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

# ---------------------------------------------------------------------
# Load spaCy
# ---------------------------------------------------------------------

print("Loading spaCy model: en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------
# Relation-specific triggers
# ---------------------------------------------------------------------

TRIGGERS = {
    "uses": {
        "use", "uses", "used", "using", "utilize", "utilizes",
        "utilized", "employ", "employs", "employed",
        "leverage", "leverages", "leveraged",
        "deploy", "deploys", "deployed",
        "run", "runs", "ran", "execute", "executes", "executed",
        "load", "loads", "loaded"
    },

    "targets": {
        "target", "targets", "targeted", "attack", "attacks",
        "attacked", "focus", "focuses", "focused",
        "aim", "aims", "aimed", "victimize", "victimizes",
        "victimized"
    },

    "originates-from": {
        "originate", "originates", "originated",
        "associate", "associated", "associates",
        "link", "linked", "links",
        "trace", "traced"
    },

    "located-at": {
        "locate", "located", "locates",
        "base", "based", "operate", "operates", "operated"
    },

    "downloads": {
        "download", "downloads", "downloaded",
        "retrieve", "retrieves", "retrieved",
        "fetch", "fetches", "fetched"
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
        "credit", "credited", "credits"
    },

    "delivers": {
        "deliver", "delivers", "delivered"
    },

    "impersonates": {
        "impersonate", "impersonates", "impersonated"
    },

    "variant-of": {
        "variant", "variants"
    },

    "exploits": {
        "exploit", "exploits", "exploited"
    }
}

TRIGGER_TO_RELATION = {}

for relation, words in TRIGGERS.items():
    for word in words:
        TRIGGER_TO_RELATION[word] = relation


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())


def entity_tokens(doc, entity_text):
    """
    Find token spans matching the entity surface form.
    """
    entity_doc = nlp.make_doc(entity_text)

    target = [t.text.lower() for t in entity_doc]

    if not target:
        return []

    matches = []

    for i in range(len(doc) - len(target) + 1):
        window = [t.text.lower() for t in doc[i:i + len(target)]]

        if window == target:
            matches.append(list(doc[i:i + len(target)]))

    return matches


def best_entity_span(doc, entity_text, preferred_index=None):
    """
    Pick the entity occurrence closest to preferred_index.
    """
    spans = entity_tokens(doc, entity_text)

    if not spans:
        return None

    if preferred_index is None:
        return spans[0]

    return min(
        spans,
        key=lambda span: abs(
            ((span[0].i + span[-1].i) / 2) - preferred_index
        )
    )


def subtree_text(token):
    """
    Return the complete dependency subtree rooted at token.
    """
    return " ".join(t.text for t in token.subtree)


def get_subjects(trigger):
    """
    Find active/passive subjects of a trigger.
    """
    subjects = []

    for child in trigger.children:

        if child.dep_ in {
            "nsubj",
            "csubj",
            "nsubjpass"
        }:
            subjects.append(child)

        # Passive agent:
        # was used BY cybercriminals
        if child.dep_ == "agent":
            for subchild in child.children:
                if subchild.dep_ == "pobj":
                    subjects.append(subchild)

    return subjects


def get_objects(trigger):
    """
    Find direct, indirect and predicate objects.
    """
    objects = []

    for child in trigger.children:

        if child.dep_ in {
            "dobj",
            "obj",
            "iobj",
            "attr",
            "oprd"
        }:
            objects.append(child)

        if child.dep_ == "prep":
            for subchild in child.children:
                if subchild.dep_ == "pobj":
                    objects.append(subchild)

    return objects


def span_contains(span, token):
    return span is not None and any(t.i == token.i for t in span)


def token_in_subtree(root, token):
    return any(t.i == token.i for t in root.subtree)


def find_trigger_candidates(sent):
    """
    Find relation triggers inside a sentence.
    """
    candidates = []

    for token in sent:

        lemma = token.lemma_.lower()
        text = token.text.lower()

        relation = TRIGGER_TO_RELATION.get(lemma)

        if relation is None:
            relation = TRIGGER_TO_RELATION.get(text)

        if relation:
            candidates.append({
                "token": token,
                "relation": relation
            })

    return candidates


def find_nearest_trigger(sent, source_span, target_span, predicted_relation):
    """
    Prefer triggers belonging to the predicted relation.

    Then prefer triggers whose dependency structure connects
    to one of the candidate entities.
    """
    candidates = find_trigger_candidates(sent)

    candidates = [
        x for x in candidates
        if x["relation"] == predicted_relation
    ]

    if not candidates:
        return None

    source_center = sum(t.i for t in source_span) / len(source_span)
    target_center = sum(t.i for t in target_span) / len(target_span)
    pair_center = (source_center + target_center) / 2

    scored = []

    for item in candidates:
        trigger = item["token"]

        score = abs(trigger.i - pair_center)

        # Strong bonus if source or target appears in
        # subject/object structure of trigger.
        subjects = get_subjects(trigger)
        objects = get_objects(trigger)

        if any(
            token_in_subtree(s, source_span[0])
            or token_in_subtree(s, target_span[0])
            for s in subjects
        ):
            score -= 10

        if any(
            token_in_subtree(o, source_span[0])
            or token_in_subtree(o, target_span[0])
            for o in objects
        ):
            score -= 10

        scored.append((score, item))

    scored.sort(key=lambda x: x[0])

    return scored[0][1]


def entity_matches_root(entity_span, root):
    """
    Check whether an entity span belongs to a dependency subtree.
    """
    if root is None:
        return False

    entity_ids = {t.i for t in entity_span}

    subtree_ids = {t.i for t in root.subtree}

    return bool(entity_ids & subtree_ids)


def relation_arguments(trigger, source_span, target_span):
    """
    Determine whether source/target correspond to grammatical
    arguments of the relation trigger.
    """

    subjects = get_subjects(trigger)
    objects = get_objects(trigger)

    source_subject = False
    target_subject = False

    source_object = False
    target_object = False

    source_agent = False
    target_agent = False

    for subj in subjects:

        if entity_matches_root(source_span, subj):
            source_subject = True

        if entity_matches_root(target_span, subj):
            target_subject = True

        # Passive agent
        if subj.dep_ == "pobj":
            if entity_matches_root(source_span, subj):
                source_agent = True

            if entity_matches_root(target_span, subj):
                target_agent = True

    for obj in objects:

        if entity_matches_root(source_span, obj):
            source_object = True

        if entity_matches_root(target_span, obj):
            target_object = True

    # -------------------------------------------------------------
    # Active: SOURCE -> trigger -> TARGET
    # -------------------------------------------------------------

    if source_subject and target_object:
        return {
            "direction": "forward",
            "structure": "SOURCE_SUBJECT_TARGET_OBJECT",
            "source_role": "subject",
            "target_role": "object"
        }

    # -------------------------------------------------------------
    # Reverse: TARGET -> trigger -> SOURCE
    # -------------------------------------------------------------

    if target_subject and source_object:
        return {
            "direction": "reverse",
            "structure": "TARGET_SUBJECT_SOURCE_OBJECT",
            "source_role": "object",
            "target_role": "subject"
        }

    # -------------------------------------------------------------
    # Passive:
    #
    # SOURCE is used by TARGET
    #
    # TARGET = agent
    # SOURCE = passive subject
    # -------------------------------------------------------------

    if target_agent and source_subject:
        return {
            "direction": "reverse",
            "structure": "SOURCE_PASSIVE_SUBJECT_TARGET_AGENT",
            "source_role": "patient",
            "target_role": "agent"
        }

    if source_agent and target_subject:
        return {
            "direction": "forward",
            "structure": "TARGET_PASSIVE_SUBJECT_SOURCE_AGENT",
            "source_role": "agent",
            "target_role": "patient"
        }

    return {
        "direction": "unknown",
        "structure": "none",
        "source_role": None,
        "target_role": None
    }


def dependency_path(source_span, target_span):
    """
    Approximate shortest dependency path between entity heads.
    """
    if not source_span or not target_span:
        return []

    source = source_span[0]
    target = target_span[0]

    source_ancestors = {source.i: source}

    cur = source
    while cur.head != cur:
        cur = cur.head
        source_ancestors[cur.i] = cur

    path_target = []
    cur = target

    while cur.i not in source_ancestors:
        path_target.append(cur)

        if cur.head == cur:
            return []

        cur = cur.head

    lca = source_ancestors[cur.i]

    path = []

    cur = source
    while cur.i != lca.i:
        path.append(cur)
        cur = cur.head

    path.append(lca)

    path.extend(reversed(path_target))

    return [
        f"{t.text}/{t.dep_}"
        for t in path
    ]


def analyze_record(record):

    text = normalize_text(record["text"])

    predicted_relation = record["predicted_relation"]

    source_text = record["source_mention"]
    target_text = record["target_mention"]

    doc = nlp(text)

    source_span = None
    target_span = None

    best_result = None

    # -------------------------------------------------------------
    # Search sentence by sentence
    # -------------------------------------------------------------

    for sent in doc.sents:

        sent_source = best_entity_span(
            sent,
            source_text
        )

        sent_target = best_entity_span(
            sent,
            target_text
        )

        if sent_source is None or sent_target is None:
            continue

        source_center = sum(t.i for t in sent_source) / len(sent_source)
        target_center = sum(t.i for t in sent_target) / len(sent_target)

        trigger_info = find_nearest_trigger(
            sent,
            sent_source,
            sent_target,
            predicted_relation
        )

        if trigger_info is None:
            continue

        trigger = trigger_info["token"]

        args = relation_arguments(
            trigger,
            sent_source,
            sent_target
        )

        path = dependency_path(
            sent_source,
            sent_target
        )

        result = {
            "sentence": sent.text,

            "source_span": [t.text for t in sent_source],
            "target_span": [t.text for t in sent_target],

            "trigger": trigger.text,
            "trigger_lemma": trigger.lemma_,
            "trigger_pos": trigger.pos_,
            "trigger_dep": trigger.dep_,

            "source_head": sent_source[0].text,
            "target_head": sent_target[0].text,

            "source_head_dep": sent_source[0].dep_,
            "target_head_dep": sent_target[0].dep_,

            "direction": args["direction"],
            "argument_structure": args["structure"],

            "source_role": args["source_role"],
            "target_role": args["target_role"],

            "dependency_path": path,

            "trigger_index": trigger.i,

            "argument_distance": abs(
                source_center - target_center
            )
        }

        # Prefer a case where arguments are actually resolved.
        if args["direction"] != "unknown":
            best_result = result
            break

        if best_result is None:
            best_result = result

    # -------------------------------------------------------------
    # No relation trigger
    # -------------------------------------------------------------

    if best_result is None:

        return {
            "trigger_found": False,
            "direction": "unknown",
            "argument_structure": "none",
            "source_role": None,
            "target_role": None,
            "trigger": None,
            "trigger_lemma": None,
            "sentence": None,
            "dependency_path": []
        }

    best_result["trigger_found"] = True

    return best_result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("Loading records...")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Records loaded: {len(records)}")
print()

output = []

trigger_count = 0
direction_counts = Counter()
structure_counts = Counter()
relation_trigger_counts = Counter()

for i, record in enumerate(records, 1):

    result = analyze_record(record)

    new_record = dict(record)
    new_record["relation_arguments_v2"] = result

    output.append(new_record)

    if result["trigger_found"]:
        trigger_count += 1
        relation_trigger_counts[
            record["predicted_relation"]
        ] += 1

    direction_counts[result["direction"]] += 1
    structure_counts[
        result["argument_structure"]
    ] += 1

    if i % 500 == 0:
        print(f"Processed {i}/{len(records)}")


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("EXTRACTION V2 COMPLETE")
print("=" * 75)

print(f"Total records       : {len(records)}")
print(f"Trigger found       : {trigger_count}")
print(f"Trigger not found   : {len(records) - trigger_count}")

print()
print("Relation trigger counts:")

for relation, count in relation_trigger_counts.most_common():
    print(f"  {relation:25s}: {count}")

print()
print("Relation directions:")

for direction, count in direction_counts.most_common():
    print(f"  {direction:35s}: {count}")

print()
print("Argument structures:")

for structure, count in structure_counts.most_common():
    print(f"  {structure:45s}: {count}")

print()
print("Saved to:")
print(OUTPUT_FILE)