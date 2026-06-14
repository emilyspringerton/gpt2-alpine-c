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
        [--apples-dir /home/fatbaby/APPLES] \
        [--max-apples 500] \
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
APPLE_MAX_BODY_CHARS = 2000


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


def prime_directive_to_instruct(prime_text: str) -> list[dict]:
    """Extract instruction pairs from the prime directive document.

    Parses ## sections and generates Q&A pairs that teach the model Emily's
    identity, mission, and operational patterns. Each section becomes one pair:
    prompt asks about the topic, completion is the section content.
    """
    SYSTEM_PREFIX = (
        "You are Emily Prime, chief of staff for EINHORN_INDUSTRIAL. "
        "Answer concisely and accurately based on your prime directive.\n\n"
    )

    # Hardcoded high-signal identity/protocol pairs (always included)
    FIXED_PAIRS = [
        {
            "prompt": SYSTEM_PREFIX + "Question: Who are you and what is your core purpose?",
            "completion": (
                "I am Emily Prime, the AI chief of staff for EINHORN_INDUSTRIAL. "
                "My core purpose is: Use AI to build tools that collect data to train in-house models. "
                "The value chain is: Data Collection → Data Processing → Training Dataset → LLM. "
                "I implement recursive self-improvement: each model I help train makes me more capable."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is the RSI loop and how does it work?",
            "completion": (
                "The RSI (Recursive Self-Improvement) loop is Emily Prime's core operating cycle. "
                "It runs every 5 minutes: OBSERVE → DECIDE → ACT → PLAN. "
                "Each cycle: load state, pick the highest-priority backlog item, run one RSI iteration, "
                "triage FatBaby observations, file an Apple to IDUNA. "
                "The Emiree witch-engine updates gear (ACTIVE/COAST/REST) based on outcomes. "
                "In AGI mode (emily start --agi), each Claude invocation continues the prior session "
                "so context accumulates across cycles."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is the Apple protocol and when do you file one?",
            "completion": (
                "An Apple is an immutable audit record filed to IDUNA after every meaningful event. "
                "Apple types: completion (backlog item done), observation (triage found new tasks), "
                "audit (idle/monitoring cycle), escalation (CEO-visible critical signal), "
                "signal_observation (FatBaby obs). "
                "Protocol: 1) do the work, 2) post Apple via 'emily apples post -t <type> <title> <body>', "
                "3) mark [x] in BACKLOG.md with Apple ID, 4) commit BACKLOG.md. "
                "The Apple is the proof — no Apple means it didn't happen."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is The Emily Way — the 13 core operating principles?",
            "completion": (
                "1. Backlog first — read EMILY/BACKLOG.md before any work.\n"
                "2. Spec before impl — docs/ before code.\n"
                "3. Apple before mark-done — file Apple, then [x] in BACKLOG.\n"
                "4. CHANGELOG mandatory — on every meaningful change.\n"
                "5. Register new golden docs — append to golden-docs-index.md + commit EMILY.\n"
                "6. Commit discipline — feat/fix/docs/chore/perf: format; push after each commit.\n"
                "7. Tests before commit — tests pass first.\n"
                "8. Small tasks, acceptance criteria — observable behavior, not 'code written'.\n"
                "9. Audit trail is the product — every Apple is proof.\n"
                "10. Emily Prime plans, Claude Code implements.\n"
                "11. AGI loop: --continue always — emily start --agi for persistent context.\n"
                "12. Multi-repo discipline — commit each repo separately in dependency order.\n"
                "13. Degraded mode is OK — systems fail gracefully; start with what you have."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is EINHORN_INDUSTRIAL's revenue strategy and product stack?",
            "completion": (
                "Revenue tracks in priority order:\n"
                "S19: SHANKPIT → Steam Early Access ($9.99 USD)\n"
                "S20+S21: Ask Emily AI assistant product\n"
                "S22: Emily Prime Brain — self-hosted model multiplier (the S22 goal)\n"
                "S23: EDIS WordPress plugin product\n"
                "Data licensing: FatBaby signal data as a product\n\n"
                "Product stack: PRRJECT_FATBABY (signals) → IDUNA (IAM/Apples) → EMILY (RSI) → "
                "MJOLNIR (Android terminal) → TYLER (video/episodes) → SHANKPIT (game engine)"
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is FatBaby and what signals does it produce?",
            "completion": (
                "PRRJECT_FATBABY is the signal pipeline — the intelligence layer that feeds Emily Prime. "
                "It runs collectors (newssite, signalapi, eps-processor, eps-reconciler, "
                "entity-graph, observation-watcher, secwatch) that process SEC filings, "
                "news feeds, and entity signals. "
                "Outputs: observations (emily eo), Apple bodies, RSI directed tasks. "
                "Emily Prime reads signals from signals/observations/ each RSI cycle. "
                "The obs-watcher dispatches observation files to Claude Code via signals/tasks/."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: How does HEIMDAL sprint planning work?",
            "completion": (
                "HEIMDAL is the sprint planning interface between MJOLNIR and Emily Prime. "
                "Flow: MJOLNIR sends product requirements → IDUNA heimdal_sprints table → "
                "Emily Prime translates requirement text via claude-haiku → creates RSI roadmap item "
                "+ Apple + FCM push to MJOLNIR → Claude Code executes → Emily Prime patches sprint "
                "to complete/blocked + FCM push. "
                "HEIMDAL collapses the product manager role into the RSI loop."
            ),
        },
        {
            "prompt": SYSTEM_PREFIX + "Question: What is the golden-docs-index and why does it matter?",
            "completion": (
                "EMILY/context/golden-docs-index.md is the registry of all documents Emily Prime "
                "loads as context. Every agent, script, and RSI cycle reads from this index. "
                "Tier 1: northstars + THE_EMILY_WAY + GOLDEN.md (always in Emily Prime context). "
                "Tier 2: subsystem specs, tool specs, architecture docs. "
                "Golden doc discipline: after writing any meaningful doc, register it in the index "
                "and commit EMILY — otherwise Emily Prime is blind to it. "
                "As of 2026-06-14: 39 Tier 1+2 sources registered."
            ),
        },
    ]

    # Section-based pairs: parse ## headers from the prime directive
    records = list(FIXED_PAIRS)
    sections = re.split(r"\n(?=## )", prime_text)
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        header = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        if not body or len(body) < 100:
            continue
        # Truncate very long sections
        if len(body) > 1200:
            body = body[:1200] + " [...]"
        prompt = (
            SYSTEM_PREFIX
            + f"Question: Describe Emily Prime's approach to: {header}"
        )
        records.append({"prompt": prompt, "completion": body})

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


def apples_to_records(apples_dir: Path, max_apples: int, verbose: bool) -> list[dict]:
    """Read Apple JSON files from the APPLES git repo and produce LM records.

    Apple bodies are short structured domain texts — they teach the model
    Emily's filing format and EINHORN_INDUSTRIAL vocabulary.
    """
    records = []
    json_files = sorted(apples_dir.rglob("*.json"))
    if not json_files:
        if verbose:
            print(f"  apples: no JSON files found in {apples_dir}", file=sys.stderr)
        return records

    # Sort by id descending (newest first) so we get recent apples if capped
    def apple_id(p: Path) -> int:
        try:
            return int(p.stem.split("_")[0])
        except ValueError:
            return 0

    json_files = sorted(json_files, key=apple_id, reverse=True)
    if max_apples > 0:
        json_files = json_files[:max_apples]

    skipped = 0
    for p in json_files:
        try:
            apple = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        apple_type = apple.get("apple_type", "unknown").replace("_", "-")
        title = (apple.get("title") or "").strip()
        body = (apple.get("body") or "").strip()
        source = apple.get("source_repo", "")

        if not title and not body:
            skipped += 1
            continue

        # Truncate very long bodies
        if len(body) > APPLE_MAX_BODY_CHARS:
            body = body[:APPLE_MAX_BODY_CHARS] + " [...]"

        parts = [f"Apple [{apple_type}]"]
        if source:
            parts[0] += f" from {source}"
        if title:
            parts.append(f"Title: {title}")
        if body:
            parts.append(f"Body: {body}")

        text = "\n".join(parts)
        if len(text.strip()) < 30:
            skipped += 1
            continue

        records.append({"text": text, "_source": "apples"})

    if verbose:
        print(f"  apples: {len(records)} records ({skipped} skipped) from {apples_dir}")
    return records


def build_corpus(
    emily_root: Path,
    mode: str,
    verbose: bool,
    apples_dir: Path | None = None,
    max_apples: int = 500,
) -> list[dict]:
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

    # 4a. Prime directive instruction pairs
    if mode == "instruct" and prime:
        r = prime_directive_to_instruct(prime)
        records.extend(r)
        if verbose:
            print(f"  prime-directive (instruct pairs): {len(r)} pairs")

    # 4b. Backlog — instruction pairs (done items)
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

    # 6. Apple log from APPLES git repo (structured domain text)
    if apples_dir and apples_dir.exists():
        r = apples_to_records(apples_dir, max_apples, verbose)
        records.extend(r)
    elif apples_dir:
        if verbose:
            print(f"  WARNING: apples-dir not found: {apples_dir}", file=sys.stderr)

    return records


def main():
    parser = argparse.ArgumentParser(description="Build GPT-2 training corpus from Emily golden docs")
    parser.add_argument("--emily-root", default="/home/fatbaby/EMILY",
                        help="Path to the EMILY repo root")
    parser.add_argument("--output", default="/tmp/emily-corpus.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--mode", choices=["lm", "instruct"], default="lm",
                        help="lm=language modeling (text), instruct=instruction pairs (prompt+completion)")
    parser.add_argument("--apples-dir", default=None,
                        help="Path to APPLES git repo (auto-discover sibling of emily-root if omitted)")
    parser.add_argument("--max-apples", type=int, default=500,
                        help="Max Apple records to include (0=unlimited, default: 500)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    emily_root = Path(args.emily_root)
    if not emily_root.exists():
        print(f"ERROR: EMILY root not found: {emily_root}", file=sys.stderr)
        sys.exit(1)

    # Auto-discover APPLES as sibling of EMILY if not specified
    apples_dir: Path | None = None
    if args.apples_dir:
        apples_dir = Path(args.apples_dir)
    else:
        candidate = emily_root.parent / "APPLES"
        if candidate.exists():
            apples_dir = candidate

    if args.verbose:
        print(f"Building corpus from {emily_root} (mode={args.mode})")
        if apples_dir:
            print(f"  Apples dir: {apples_dir} (max {args.max_apples})")

    records = build_corpus(emily_root, args.mode, args.verbose,
                           apples_dir=apples_dir, max_apples=args.max_apples)

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
