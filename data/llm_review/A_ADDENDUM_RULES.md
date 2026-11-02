# Addendum rules (used only in folder A)

These fill gaps the instructions file does not cover. They were agreed by the three reviewing models
(Claude Sonnet 5.5, Claude Opus, and a third model) after cross-checking. The md file's own rules still apply.

1. EXPLOITS needs the exploiter as subject. A payload dropped via a CVE does not exploit it; the document, dropper or campaign does.
   -> UNS-001, 002, 003 = FALSE, WRONG_SUBJECT.
2. A packer applied to malware ("protected by", "UPX packed", "campaign using X packer") is not "the malware uses the packer".
   -> UNS-008, 010, 018 = UNCERTAIN, NOT_ASSERTED; REC-06 = NONE.
3. A type label is not a relation. "banking trojan", "ransomware", "RaaS" say what the malware is, not what it targets or does.
   -> REC-03, 10, 17, 21 (ransomware part), 26 = no relation.
4. EXPLOITS needs a CVE. An exploited software product (Kaseya VSA) is not a victim.  -> REC-21 = NONE.
5. Hedged, future-tense or unnamed-subject statements = UNCERTAIN.  -> UNS-006, 017.
6. Indirect wording counts when subject and object are clear (this is already in the md file).
   -> REC-01, REC-15, REC-27 (second relation), UNS-022.
7. Entity missing: write the relation and note it (already in the md file).  -> REC-15.
