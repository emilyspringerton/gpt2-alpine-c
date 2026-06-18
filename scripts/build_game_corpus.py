#!/usr/bin/env python3
"""
build_game_corpus.py — aggregate SHANKPIT replay NDJSON files into training JSONL.

Reads all *.ndjson files from a replay directory (written by emily-bot -replay-dir),
converts each {tick, state, action} record to an instruction pair
{prompt: state, completion: action}, deduplicates, and writes a JSONL corpus
ready for --game-replays ingestion by prime_directive_dataset.py.

Usage:
    python3 scripts/build_game_corpus.py \
        --replay-dir /home/fatbaby/SHANKPIT/var/replays \
        --output /tmp/game-corpus.jsonl \
        [--max-records 5000] \
        [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path


def load_replays(replay_dir: Path, max_records: int, verbose: bool) -> list:
    files = sorted(replay_dir.glob("*.ndjson"))
    if not files:
        print(f"WARNING: no *.ndjson files found in {replay_dir}", file=sys.stderr)
        return []
    records = []
    for fpath in files:
        n_before = len(records)
        with fpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    state = rec.get("state", "").strip()
                    action = rec.get("action", "").strip()
                    if state and action:
                        records.append({"prompt": state, "completion": action, "_source": "game_replay"})
                except (json.JSONDecodeError, KeyError):
                    continue
        if verbose:
            print(f"  {fpath.name}: {len(records) - n_before} records")
        if max_records > 0 and len(records) >= max_records:
            break
    return records[:max_records] if max_records > 0 else records


def deduplicate(records: list) -> tuple[list, int]:
    seen = set()
    out = []
    for r in records:
        key = r["prompt"] + "\x00" + r["completion"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out, len(records) - len(out)


def main():
    parser = argparse.ArgumentParser(description="Aggregate SHANKPIT replay NDJSON → training JSONL")
    parser.add_argument("--replay-dir", default="/home/fatbaby/SHANKPIT/var/replays",
                        help="Directory containing *.ndjson replay files")
    parser.add_argument("--output", default="/tmp/game-corpus.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Cap total records (0=unlimited)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Skip deduplication")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        print(f"ERROR: replay dir not found: {replay_dir}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loading replays from {replay_dir}")

    records = load_replays(replay_dir, args.max_records, args.verbose)

    if not args.no_dedupe:
        records, removed = deduplicate(records)
        if args.verbose and removed:
            print(f"  Deduplication: removed {removed} → {len(records)} records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    if records:
        print(f"Sample state:  {records[0]['prompt'][:80]}")
        print(f"Sample action: {records[0]['completion']}")


if __name__ == "__main__":
    main()
