# Audit of `extract_relations.py` relations

Review every row of `data/extracted_relations_audit_template.csv`. Each row is one
automatically extracted relation (`head` -> `relation` -> `tail`) with its evidence
sentence. The trigger, direction and confidence columns are system outputs, not labels.

Judge whether the **evidence sentence** supports the **exact** head, relation, tail and direction:

- `TRUE`: the sentence states the complete relation with this subject and object.
- `FALSE`: the sentence conflicts with it or shows a different subject, relation, object or direction.
- `UNCERTAIN`: the sentence does not settle it (indirect, ambiguous, needs outside context).

Do not mark `FALSE` just because the sentence does not state the relation; use `UNCERTAIN`.

Fill `error_category` for every non-`TRUE` row, using one of:
`WRONG_SUBJECT`, `WRONG_OBJECT`, `WRONG_DIRECTION`, `WRONG_RELATION_TYPE`,
`ENTITY_ERROR` (NER or linking), `NOT_ASSERTED` (negated, hypothetical, or comparison),
`OTHER`. Add a short `rationale`. Leave `reviewer` and `reviewed_at` empty unless you want them.

Example of why subject and object must be checked separately:
"TA415 used Cobalt Strike in a campaign against an entity in the airlines sector."
supports TA415 USES Cobalt Strike and TA415 TARGETS the airlines sector, but not
Cobalt Strike TARGETS the airlines sector.
