#!/usr/bin/env python3
"""
Build JSONL training corpus from Emily golden docs + prime directive.

Output format: one JSON object per line, {"text": "<content>"} for language
modeling or {"prompt": "<p>", "completion": "<c>"} for instruction fine-tuning.

Usage:
    python3 scripts/prime_directive_dataset.py \
        --emily-root /home/fatbaby/EMILY \
        --output /tmp/emily-corpus.jsonl \
        [--mode lm|instruct] \
        [--verbose]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

GOLDEN_INDEX_PATH = "context/golden-docs-index.md"
PRIME_DIRECTIVE_PATH = "docs/emily-prime-directive-data-collection.md"
TRAINING_DATA_GLOB = "var/training-data/*.jsonl"
FULL_CONTEXT_PATH = "context/full-system-context.md"
BACKLOG_PATH = "BACKLOG.md"

CHUNK_SIZE = 1500


def parse_golden_index(emily_root: Path) -> list[dict]:
    """Parse golden-docs-index.md and return list of {name, path, tier, description} dicts."""
    index_path = emily_root / GOLDEN_INDEX_PATH
    if not index_path.exists():
        print(f"WARNING: golden-docs-index.md not found at {index_path}", file=sys.stderr)
        return []

    entries = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Name") or line.startswith("| ---"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 5:
            continue
        name, path, tier, budget, desc = cols[0], cols[1], cols[2], cols[3], cols[4]
        if not name or not path:
            continue
        try:
            tier_int = int(tier)
        except ValueError:
            tier_int = 99
        entries.append({
            "name": name,
            "path": path.strip("`"),
            "tier": tier_int,
            "description": desc,
        })
    return entries


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > size and current:
            chunks.append("\n\n".join(current))
            # 20% overlap: keep last paragraph
            current = current[-1:]
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def read_doc(base: Path, rel_path: str) -> str | None:
    """Resolve a path that may be relative to base or to /home/fatbaby."""
    candidates = [
        base.parent / rel_path,          # relative to EMILY parent (repo root)
        base / rel_path,                  # relative to EMILY dir
        Path("/home/fatbaby") / rel_path, # absolute from home
        Path(rel_path),                   # as given
    ]
    for p in candidates:
        try:
            if p.exists():
                return p.read_text()
        except Exception:
            continue
    return None


def make_lm_records(text: str, source: str) -> list[dict]:
    """Language-modeling records: {"text": chunk}."""
    records = []
    for chunk in chunk_text(text):
        if len(chunk.strip()) < 50:
            continue
        records.append({"text": chunk, "_source": source})
    return records


def backlog_to_instruct(backlog_text: str) -> list[dict]:
    """Extract instruction pairs from BACKLOG.md done items."""
    records = []
    for line in backlog_text.splitlines():
        line = line.strip()
        if not line.startswith("- [x]"):
            continue
        item = line[5:].strip()
        if len(item) < 20:
            continue
        prompt = (
            "You are Emily Prime, chief of staff for EINHORN_INDUSTRIAL. "
            "The following backlog item was completed. Write a concise CHANGELOG entry "
            f"and Apple body for it.\n\nBacklog item: {item}"
        )
        completion = (
            f"CHANGELOG entry: feat: {item}\n\n"
            f"Apple body: Completed: {item}. "
            "Filed per The Emily Way principle: Apple-before-done."
        )
        records.append({"prompt": prompt, "completion": completion})
    return records


def build_corpus(emily_root: Path, mode: str, verbose: bool) -> list[dict]:
    records = []

    # 1. Full context doc (Tier 1 — highest signal density)
    full_ctx = read_doc(emily_root, FULL_CONTEXT_PATH)
    if full_ctx:
        r = make_lm_records(full_ctx, "full-system-context")
        records.extend(r)
        if verbose:
            print(f"  full-system-context: {len(r)} chunks")
    else:
        print("  WARNING: full-system-context.md not found", file=sys.stderr)

    # 2. Prime directive (high-signal identity doc)
    prime = read_doc(emily_root, PRIME_DIRECTIVE_PATH)
    if prime:
        r = make_lm_records(prime, "prime-directive")
        records.extend(r)
        if verbose:
            print(f"  prime-directive: {len(r)} chunks")
    else:
        print("  WARNING: prime directive not found", file=sys.stderr)

    # 3. All Tier 1 + Tier 2 golden docs
    golden_entries = parse_golden_index(emily_root)
    tier12 = [e for e in golden_entries if e["tier"] <= 2]
    if verbose:
        print(f"  golden-index: {len(tier12)} Tier 1+2 docs")

    for entry in tier12:
        text = read_doc(emily_root, entry["path"])
        if not text:
            if verbose:
                print(f"    SKIP (not found): {entry['name']} @ {entry['path']}")
            continue
        r = make_lm_records(text, entry["name"])
        records.extend(r)
        if verbose:
            print(f"    {entry['name']}: {len(r)} chunks")

    # 4. Backlog — instruction pairs (done items)
    if mode == "instruct":
        backlog_text = read_doc(emily_root, BACKLOG_PATH)
        if backlog_text:
            r = backlog_to_instruct(backlog_text)
            records.extend(r)
            if verbose:
                print(f"  backlog (instruct pairs): {len(r)} pairs")

    # 5. Existing training-data JSONL files
    training_dir = emily_root / "var" / "training-data"
    if training_dir.exists():
        for jsonl_path in sorted(training_dir.glob("*.jsonl")):
            count = 0
            with jsonl_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "text" in obj or ("prompt" in obj and "completion" in obj):
                            records.append(obj)
                            count += 1
                    except json.JSONDecodeError:
                        continue
            if verbose:
                print(f"  training-data/{jsonl_path.name}: {count} records")

    return records


def main():
    parser = argparse.ArgumentParser(description="Build GPT-2 training corpus from Emily golden docs")
    parser.add_argument("--emily-root", default="/home/fatbaby/EMILY",
                        help="Path to the EMILY repo root")
    parser.add_argument("--output", default="/tmp/emily-corpus.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--mode", choices=["lm", "instruct"], default="lm",
                        help="lm=language modeling (text), instruct=instruction pairs (prompt+completion)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    emily_root = Path(args.emily_root)
    if not emily_root.exists():
        print(f"ERROR: EMILY root not found: {emily_root}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Building corpus from {emily_root} (mode={args.mode})")

    records = build_corpus(emily_root, args.mode, args.verbose)

    # Strip internal _source field from output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            out_rec = {k: v for k, v in rec.items() if not k.startswith("_")}
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
