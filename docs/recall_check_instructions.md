# Recall check: instructions

Goal: measure how many of the relations that sentences really state the extractor finds (its recall). So you
write the relations down **without seeing** what the extractor found. Do not open `data/relations.jsonl`,
the trusted graph or any audit file before you finish.

File: `data/recall_check_sheet.csv` (30 random sentences from the gold-tagged reports). Save your work as
`data/recall_check_human.csv` (File, Save As, CSV UTF-8) and keep the original sheet unchanged.

For each sentence, fill the `gold_relations` column with every relation the sentence itself states. Write each
as

    head | RELATION | tail

and separate several relations with `;`. Use entity names exactly as they appear in the `entities` column.
Write `NONE` if the sentence states no relation. Use the `notes` column for anything unclear.

Relation types (head, then tail):

| RELATION | Head | Tail | Meaning |
|---|---|---|---|
| `USES` | threat actor or malware | malware or tool | the head deploys, delivers, drops, downloads or otherwise uses the tail |
| `USES_TECHNIQUE` | threat actor, malware or tool | technique or tactic | the head uses that technique or tactic |
| `EXPLOITS` | threat actor, malware or tool | CVE | the head exploits that vulnerability |
| `TARGETS` | threat actor or malware | organization, sector or location | the head attacks or focuses on that victim |

Rules:

- Use only what the sentence says. No outside knowledge, and no relations that need other sentences.
- Include a relation when the sentence states it, even if indirectly worded, as long as the subject and object are
  clear. Skip hedged or negated ones ("we have never seen X use Y"), comparisons, and mere co-occurrence.
- A list gives one relation per item: "Turla targets government, military and energy" gives three `TARGETS`.
- If a relation exists but one of its entities is not in the `entities` list, write it anyway using the
  words from the sentence and add "entity missing" in `notes`.
- If you are not sure, leave the relation out and mention it in `notes`.

Example, for "TA415 used Cobalt Strike in a campaign against an entity in the airlines sector." with entities
`TA415 [GROUP] ; Cobalt Strike [MALWARE] ; airlines sector [SECTOR]`:

    TA415 | USES | Cobalt Strike ; TA415 | TARGETS | airlines sector

(not `Cobalt Strike | TARGETS | airlines sector`: the tool does not do the targeting).

When you are done, run `python src/score_recall_check.py`. It reports recall, how many extractor relations on
these sentences are not in your list, and which of your relations were missed.
