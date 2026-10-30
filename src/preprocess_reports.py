"""Clean and preprocess AnnoCTR CTI reports into report- and segment-level tables.

Outputs (JSON Lines, under data/):
  reports.jsonl             one record per report: id, source, publication date, title, split
  processed_segments.jsonl  one record per sentence/paragraph segment with cleaned text,
                            provenance (report id, source, date, segment id) and entity spans

Run from the cti_graph_rag directory:  python src/preprocess_reports.py
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

CORPUS = Path("../anno-ctr-lrec-coling-2024/AnnoCTR")
SPLITS = ["train", "dev", "test"]  # gold splits; train_ext is a larger augmented set
REPORTS_OUT = Path("data/reports.jsonl")
SEGMENTS_OUT = Path("data/processed_segments.jsonl")

SOURCES = {
    "intel471": "Intel 471",
    "lab52": "Lab52",
    "proofpoint": "Proofpoint",
    "quoteintelligence": "QuoIntelligence",
    "zscaler": "Zscaler",
    "s2grupo": "S2 Grupo",
}

ID_RE = re.compile(r"^(?P<source>[a-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<slug>.+)$")
DCT_RE = re.compile(r'CREATION_TIME"[^>]*value="([^"]*)"')
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)\s]*\)")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MD_EMPH_RE = re.compile(r"(\*\*|__)(.+?)\1")
SPACE_RE = re.compile(r"\s+")
CVE_RE = re.compile(r"\bCVE\s*[-‑–]\s*\d{4}\s*[-‑–]\s*\d{4,7}\b", re.I)

QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "–": "-", "—": "-", " ": " ", "​": ""}


def clean_text(text):
    """Make prose uniform without touching content that entities/IOCs depend on."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(QUOTES.get(c, c) for c in text)
    text = MD_LINK_RE.sub(r"\1", text)           # [anchor](url) -> anchor
    text = MD_EMPH_RE.sub(r"\2", text)           # **bold** -> bold
    text = URL_RE.sub("[URL]", text)             # bare URLs -> placeholder
    text = re.sub(r"^\s*(?:#+|[-*>])\s+", "", text)  # heading / bullet / quote markers
    return SPACE_RE.sub(" ", text).strip()


def parse_report_id(report_id):
    m = ID_RE.match(report_id)
    if not m:
        return None, None, report_id
    return m["source"], m["date"], m["slug"]


def read_date_and_title(split, report_id):
    """Publication date from the TimeML document creation time; title = first text line."""
    date = None
    tml = CORPUS / "time_text_only" / split / f"{report_id}.tml"
    if tml.exists():
        m = DCT_RE.search(tml.read_text(encoding="utf-8"))
        date = m[1] if m else None
    title = None
    txt = CORPUS / "text" / split / f"{report_id}.txt"
    if txt.exists():
        first = txt.read_text(encoding="utf-8").strip().splitlines()[0]
        title = clean_text(first)
    return date, title


def bio_spans(tokens, tags):
    """Entity spans as (start_token, end_token_exclusive, type)."""
    spans, start, etype = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("I-") and start is not None and tag[2:] == etype:
            continue
        if start is not None:
            spans.append((start, i, etype))
            start, etype = None, None
        if tag.startswith("B-") or tag.startswith("I-"):  # stray I- starts a span
            start, etype = i, tag[2:]
    return spans


def char_offsets(text, tokens):
    """Character (start, end) of each token in the raw segment text, or None if not found."""
    offsets, pos = [], 0
    for tok in tokens:
        idx = text.find(tok, pos)
        if idx == -1:
            offsets.append(None)
            continue
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return offsets


def main():
    REPORTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    reports, stats = {}, Counter()
    seen = set()

    with open(SEGMENTS_OUT, "w", encoding="utf-8") as out:
        for split in SPLITS:
            with open(CORPUS / "ner_json" / f"{split}.json", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    report_id, seg = rec["id"].rsplit("__s", 1)
                    seg = int(seg)
                    stats["segments_in"] += 1

                    if report_id not in reports:
                        source_key, id_date, _ = parse_report_id(report_id)
                        date, title = read_date_and_title(split, report_id)
                        reports[report_id] = {
                            "report_id": report_id,
                            "source": SOURCES.get(source_key, source_key),
                            "publication_date": date or id_date,
                            "title": title,
                            "split": split,
                            "n_segments": 0,
                        }
                    rep = reports[report_id]

                    raw = rec["text"]
                    cleaned = clean_text(raw)
                    if not cleaned:
                        stats["dropped_empty"] += 1
                        continue
                    key = (report_id, cleaned)
                    duplicate = key in seen
                    seen.add(key)
                    stats["duplicates_flagged"] += duplicate

                    tokens = rec["tokens"]
                    offs = char_offsets(raw, tokens)
                    entities = []
                    for s, e, etype in bio_spans(tokens, rec["all_tags"]):
                        o_s, o_e = offs[s], offs[e - 1]
                        entities.append({
                            "text": " ".join(tokens[s:e]),
                            "type": etype,
                            "token_start": s,
                            "token_end": e,
                            "char_start": o_s[0] if o_s else None,
                            "char_end": o_e[1] if o_e else None,
                        })
                    # AnnoCTR has no CVE label, so add CVE ids found by regex
                    # (skipping any that overlap a gold span)
                    for m in CVE_RE.finditer(raw):
                        if any(e["char_start"] is not None and e["char_start"] < m.end()
                               and m.start() < e["char_end"] for e in entities):
                            continue
                        entities.append({
                            "text": m[0], "type": "CVE",
                            "token_start": None, "token_end": None,
                            "char_start": m.start(), "char_end": m.end(),
                        })
                        stats["cves_added"] += 1
                    entities.sort(key=lambda e: e["char_start"] if e["char_start"] is not None else -1)
                    stats["entities"] += len(entities)

                    rep["n_segments"] += 1
                    out.write(json.dumps({
                        "report_id": report_id,
                        "source": rep["source"],
                        "publication_date": rep["publication_date"],
                        "split": split,
                        "segment_id": seg,
                        "text": raw,
                        "clean_text": cleaned,
                        "duplicate_in_report": duplicate,
                        "entities": entities,
                    }, ensure_ascii=False) + "\n")

    with open(REPORTS_OUT, "w", encoding="utf-8") as f:
        for rep in reports.values():
            f.write(json.dumps(rep, ensure_ascii=False) + "\n")

    print(f"Reports: {len(reports)}  ({Counter(r['source'] for r in reports.values())})")
    print(f"Without date: {sum(not r['publication_date'] for r in reports.values())}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
