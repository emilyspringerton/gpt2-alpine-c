#!/usr/bin/env python3
"""
Corpus quality statistics for the Emily Prime training dataset.

Run before uploading to Colab to verify corpus health.

Usage:
    python3 scripts/corpus_stats.py [--corpus /tmp/emily-corpus.jsonl] [--verbose]
    python3 scripts/corpus_stats.py --provenance-audit --corpus var/snapshots/eps-....jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for English text."""
    return max(1, len(text) // 4)


def detect_duplicates(texts: list[str], threshold: float = 0.95) -> int:
    """Count near-duplicate records by exact prefix match (first 200 chars)."""
    seen_prefixes: set[str] = set()
    dupes = 0
    for t in texts:
        key = t[:200].strip().lower()
        if key in seen_prefixes:
            dupes += 1
        else:
            seen_prefixes.add(key)
    return dupes


def load_provenance_records(corpus_path: Path) -> list[dict]:
    """Read a snapshot corpus and return only records carrying a
    `_provenance` block (S146-01/02 snapshot format). Records without one
    (e.g. a general Track A corpus with no oracle-graded source) are
    silently excluded from the audit -- they have nothing to audit, not a
    finding.
    """
    records = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_provenance"):
                records.append(rec)
    return records


def provenance_breakdown(records: list[dict]) -> dict:
    """S146-05: verdict / source-pipeline / license-class breakdown for a
    snapshot corpus. "Source pipeline" stands in for HQ-SPEC-AI-103 §7's
    "oracle" metric -- the schema's `_provenance` block carries
    `oracle_case_id` (which case, not which oracle) and the record's own
    `_source` (e.g. "eps-headlines"), not a named oracle field; `_source`
    is the closest real proxy for "which oracle labeled this" without
    inventing a field the schema doesn't have.
    """
    verdicts: Counter = Counter()
    sources: Counter = Counter()
    license_classes: Counter = Counter()
    for rec in records:
        prov = rec.get("_provenance") or {}
        verdicts[prov.get("verdict", "unknown")] += 1
        sources[rec.get("_source", "unknown")] += 1
        license_classes[prov.get("license_class", "unknown")] += 1
    return {
        "verdicts": dict(verdicts),
        "sources": dict(sources),
        "license_classes": dict(license_classes),
    }


def load_tombstone_hashes(gpt2_root: Path) -> set[str]:
    """Same file S146-04's `load_eval_tombstones` reads
    (var/eval-tombstones.json); re-read here independently rather than
    imported from prime_directive_dataset.py so this audit can run standalone
    against any snapshot file, including ones built by a different tool
    (e.g. Go `fabledata`, S145-01) that never imports this Python module.
    Missing file -> empty set, not an error (no suite frozen yet is a valid,
    honest state -- same convention S146-04 itself uses).
    """
    path = gpt2_root / "var" / "eval-tombstones.json"
    if not path.exists():
        return set()
    entries = json.loads(path.read_text())
    return {e["sha256"] for e in entries if "sha256" in e}


def check_contamination(records: list[dict], tombstones: set[str]) -> list[str]:
    """Returns the list of source_event_hash values present in BOTH the
    snapshot and the frozen eval-tombstone set -- HQ-SPEC-AI-103 §7's
    "contamination audit results (must be zero findings)" metric. An eval
    record leaking into a training snapshot is exactly the failure
    apply_tombstones (S146-04) is supposed to prevent at build time; this
    audit is the independent check that it actually did.
    """
    if not tombstones:
        return []
    found = []
    for rec in records:
        h = (rec.get("_provenance") or {}).get("source_event_hash")
        if h and h in tombstones:
            found.append(h)
    return found


def run_provenance_audit(corpus_path: Path, gpt2_root: Path, verbose: bool) -> int:
    """Runs the S146-05 provenance audit and prints a report. Returns the
    process exit code: 0 clean, 1 if any tombstoned hash was found in the
    snapshot (a real contamination finding, not a warning to note and move
    past).
    """
    records = load_provenance_records(corpus_path)
    print(f"{'='*55}")
    print(f"  PROVENANCE AUDIT — {corpus_path.name}")
    print(f"{'='*55}")

    if not records:
        print("  No _provenance-bearing records found -- nothing to audit.")
        print("  (Not an error: a general Track A corpus has no oracle-graded")
        print("   source. Point --corpus at a snapshot .jsonl to audit one.)")
        print(f"{'='*55}")
        return 0

    breakdown = provenance_breakdown(records)
    print(f"  Records with provenance: {len(records)}")
    print()
    print("  By verdict:")
    for verdict, count in sorted(breakdown["verdicts"].items(), key=lambda kv: -kv[1]):
        print(f"    {verdict:<20} {count:>6}")
    print()
    print("  By source pipeline (oracle proxy):")
    for src, count in sorted(breakdown["sources"].items(), key=lambda kv: -kv[1]):
        print(f"    {src:<20} {count:>6}")
    print()
    print("  By license class:")
    for lic, count in sorted(breakdown["license_classes"].items(), key=lambda kv: -kv[1]):
        print(f"    {lic:<20} {count:>6}")

    tombstones = load_tombstone_hashes(gpt2_root)
    contaminated = check_contamination(records, tombstones)
    print()
    print(f"  Contamination check ({len(tombstones)} frozen eval hashes loaded):")
    if contaminated:
        print(f"    ✗ FAIL: {len(contaminated)} tombstoned hash(es) present in this snapshot")
        if verbose:
            for h in contaminated:
                print(f"      {h}")
        print(f"{'='*55}")
        return 1
    print("    ✓ zero findings")
    print(f"{'='*55}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Corpus quality stats")
    parser.add_argument("--corpus", default="/tmp/emily-corpus.jsonl",
                        help="Path to JSONL corpus (default: /tmp/emily-corpus.jsonl)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-source breakdown")
    parser.add_argument("--provenance-audit", action="store_true",
                        help="S146-05: audit a snapshot corpus's verdict/oracle/license-class "
                             "breakdown + eval-tombstone contamination check, instead of the "
                             "general quality stats below")
    parser.add_argument("--gpt2-root", default=None,
                        help="Repo root for --provenance-audit's tombstone lookup "
                             "(default: this script's own repo)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found: {corpus_path}", file=sys.stderr)
        print("  Run: python3 scripts/prime_directive_dataset.py --output /tmp/emily-corpus.jsonl",
              file=sys.stderr)
        sys.exit(1)

    if args.provenance_audit:
        gpt2_root = Path(args.gpt2_root) if args.gpt2_root else Path(__file__).resolve().parent.parent
        sys.exit(run_provenance_audit(corpus_path, gpt2_root, args.verbose))

    records = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("ERROR: corpus is empty", file=sys.stderr)
        sys.exit(1)

    # Classify record types
    lm_records = [r for r in records if "text" in r]
    instruct_records = [r for r in records if "prompt" in r and "completion" in r]
    other_records = [r for r in records if r not in lm_records and r not in instruct_records]

    # Extract all text
    all_texts: list[str] = []
    for r in lm_records:
        all_texts.append(r["text"])
    for r in instruct_records:
        all_texts.append(r["prompt"] + "\n" + r["completion"])

    total_chars = sum(len(t) for t in all_texts)
    total_tokens = sum(estimate_tokens(t) for t in all_texts)
    text_lengths = sorted(len(t) for t in all_texts)
    median_len = text_lengths[len(text_lengths) // 2]
    min_len = text_lengths[0]
    max_len = text_lengths[-1]

    # Short record count (< 100 chars)
    short_count = sum(1 for t in all_texts if len(t) < 100)

    # Duplicate detection
    dupe_count = detect_duplicates(all_texts)

    # Source breakdown
    source_counter: Counter = Counter()
    for r in records:
        src = r.get("_source", "unknown")
        source_counter[src] += 1

    # File info
    size_kb = corpus_path.stat().st_size / 1024

    print(f"{'='*55}")
    print(f"  EMILY CORPUS STATS — {corpus_path.name}")
    print(f"{'='*55}")
    print(f"  File:         {corpus_path}  ({size_kb:.1f} KB)")
    print(f"  Total records:{len(records):>8}")
    print(f"    LM records: {len(lm_records):>8}  (text-only)")
    print(f"    Instruct:   {len(instruct_records):>8}  (prompt+completion pairs)")
    if other_records:
        print(f"    Other:      {len(other_records):>8}  (unknown format)")
    print()
    print(f"  Total chars:  {total_chars:>8,}")
    print(f"  Est. tokens:  {total_tokens:>8,}  (~4 chars/token)")
    print(f"  GPT-2 context: {total_tokens/512:.0f}x 512-token sequences (Colab MAX_LENGTH)")
    print()
    print(f"  Text length stats:")
    print(f"    Min:          {min_len:>6} chars")
    print(f"    Median:       {median_len:>6} chars")
    print(f"    Max:          {max_len:>6} chars")
    print(f"    Short (<100): {short_count:>6} records")
    print()
    print(f"  Quality:")
    dupe_pct = dupe_count / max(len(all_texts), 1) * 100
    dupe_flag = " ⚠" if dupe_pct > 5 else " ✓"
    short_pct = short_count / max(len(records), 1) * 100
    short_flag = " ⚠" if short_pct > 10 else " ✓"
    print(f"    Duplicates:   {dupe_count:>4} ({dupe_pct:.1f}%){dupe_flag}")
    print(f"    Short records:{short_count:>4} ({short_pct:.1f}%){short_flag}")

    # Colab training estimate
    seqs_per_epoch = max(1, total_tokens // 512)
    steps_per_epoch = max(1, seqs_per_epoch // 4)  # batch_size=4, grad_accum=4 → effective batch 16
    total_steps = steps_per_epoch * 3  # 3 epochs
    seconds_t4 = total_steps * 0.6  # ~0.6s/step on T4 GPU (GPT-2 small, batch=4)
    minutes_t4 = seconds_t4 / 60
    print()
    print(f"  Colab estimate (T4 GPU, batch=4, 3 epochs):")
    print(f"    ~{seqs_per_epoch} sequences/epoch  →  ~{total_steps} total steps")
    print(f"    ~{minutes_t4:.1f} min training time  (~{total_steps * 0.6:.0f}s @ ~0.6s/step)")

    if args.verbose and source_counter:
        print()
        print(f"  Source breakdown:")
        for src, count in source_counter.most_common():
            pct = count / len(records) * 100
            bar = "█" * int(pct / 2)
            print(f"    {src:<30} {count:>4} ({pct:4.1f}%) {bar}")

    print(f"{'='*55}")

    # Return non-zero if quality checks fail
    if dupe_pct > 5 or short_pct > 10:
        print("\n  WARN: Quality flags raised. Investigate before uploading.")
        sys.exit(1)
    else:
        print("\n  ✓ Corpus quality OK. Ready to upload via drive_sync.py.")


if __name__ == "__main__":
    main()
