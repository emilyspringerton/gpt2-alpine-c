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
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
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


def sec_filings_to_records(fatbaby_root: Path, max_docs: int, verbose: bool) -> list[dict]:
    """Extract SEC filing text from secwatch source_document_persisted events."""
    records = []
    events_dir = fatbaby_root / "var" / "secwatch" / "events"
    if not events_dir.exists():
        if verbose:
            print(f"  sec-filings: events dir not found: {events_dir}", file=sys.stderr)
        return records

    skipped = 0
    total = 0
    for ndjson_path in sorted(events_dir.glob("*.ndjson")):
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    e = d.get("event", {})
                    if e.get("type") != "source_document_persisted":
                        continue
                    data = e.get("data", {})
                    text = (data.get("cleaned_text") or "").strip()
                    if not text or len(text) < 100:
                        skipped += 1
                        continue
                    ticker = data.get("ticker", "")
                    form = data.get("form", "") or data.get("source_type", "")
                    header = f"SEC Filing [{form}] - {ticker}\n" if (form or ticker) else "SEC Filing\n"
                    combined = header + text
                    for chunk in chunk_text(combined):
                        if len(chunk.strip()) >= 50:
                            records.append({"text": chunk, "_source": "sec-filings"})
                    total += 1
                    if max_docs > 0 and total >= max_docs:
                        break
                except (json.JSONDecodeError, OSError):
                    skipped += 1
        if max_docs > 0 and total >= max_docs:
            break

    if verbose:
        print(f"  sec-filings: {len(records)} chunks from {total} docs ({skipped} skipped)")
    return records


def press_releases_to_records(fatbaby_root: Path, max_docs: int, verbose: bool) -> list[dict]:
    """Extract press release text from prwatch-body pr_body_fetched events."""
    records = []
    events_dir = fatbaby_root / "var" / "prwatch-body" / "events"
    if not events_dir.exists():
        if verbose:
            print(f"  press-releases: events dir not found: {events_dir}", file=sys.stderr)
        return records

    skipped = 0
    total = 0
    for ndjson_path in sorted(events_dir.glob("*.ndjson")):
        with open(ndjson_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    e = d.get("event", {})
                    if e.get("type") != "pr_body_fetched":
                        continue
                    data = e.get("data", {})
                    body = (data.get("body") or "").strip()
                    headline = (data.get("headline") or "").strip()
                    if not body or len(body) < 100:
                        skipped += 1
                        continue
                    combined = (f"Press Release: {headline}\n\n" if headline else "") + body
                    for chunk in chunk_text(combined):
                        if len(chunk.strip()) >= 50:
                            records.append({"text": chunk, "_source": "press-releases"})
                    total += 1
                    if max_docs > 0 and total >= max_docs:
                        break
                except (json.JSONDecodeError, OSError):
                    skipped += 1
        if max_docs > 0 and total >= max_docs:
            break

    if verbose:
        print(f"  press-releases: {len(records)} chunks from {total} docs ({skipped} skipped)")
    return records


def eps_headlines_to_records(fatbaby_root: Path, verbose: bool) -> list[dict]:
    """S146-02: EPS headline records for FABLE E0 (HQ-SPEC-AI-103 §3's flagship task).

    Reads var/eps/articles.ndjson (raw PR-derived EPS extractions) joined against
    var/eps/oracle.ndjson (eps-reconciler's confirm/contradict verdicts) by
    source_identity. One record per article, never chunked -- record identity is
    what provenance hangs on (chunking, as sec_filings_to_records/
    press_releases_to_records do, would sever the article from its oracle verdict).

    Verdict gating (Löbian rule 1, HQ-SPEC-AI-103 §5.1): only "confirmed" cases are
    FABLE-eligible -- pending/contradicted cases are excluded and counted, never
    silently dropped without a number attached.
    """
    records: list[dict] = []
    articles_path = fatbaby_root / "var" / "eps" / "articles.ndjson"
    oracle_path = fatbaby_root / "var" / "eps" / "oracle.ndjson"
    if not articles_path.exists() or not oracle_path.exists():
        if verbose:
            print(f"  eps-headlines: articles/oracle store not found under {fatbaby_root / 'var' / 'eps'}",
                  file=sys.stderr)
        return records

    # oracle.ndjson: last verdict per source_identity wins (a case can be
    # re-reconciled; the store is append-only, so later lines supersede).
    oracle_by_identity: dict[str, dict] = {}
    with oracle_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = case.get("source_identity")
            if sid:
                oracle_by_identity[sid] = case

    confirmed = 0
    excluded_pending = 0
    excluded_contradicted = 0
    skipped_no_oracle = 0
    with articles_path.open() as f:
        for raw_line in f:
            raw_line_stripped = raw_line.strip()
            if not raw_line_stripped:
                continue
            try:
                article = json.loads(raw_line_stripped)
            except json.JSONDecodeError:
                continue

            sid = article.get("source_identity")
            case = oracle_by_identity.get(sid) if sid else None
            if case is None:
                skipped_no_oracle += 1
                continue

            verdict = case.get("verdict", "pending")
            if verdict == "pending":
                excluded_pending += 1
                continue
            if verdict == "contradicted":
                excluded_contradicted += 1
                continue
            if verdict != "confirmed":
                # Unknown verdict value -- fail closed (not FABLE-eligible), don't
                # silently include it either.
                excluded_pending += 1
                continue

            headline = (article.get("headline") or "").strip()
            dek = (article.get("dek") or "").strip()
            body = (article.get("body") or "").strip()
            text = "\n".join(p for p in (headline, dek, body) if p)
            if not text:
                continue

            source_event_hash = hashlib.sha256(raw_line_stripped.encode("utf-8")).hexdigest()
            period = article.get("period") or {}

            records.append({
                "text": text,
                "_source": "eps-headlines",
                "_provenance": {
                    "source_event_hash": source_event_hash,
                    "source_identity": sid,
                    "oracle_case_id": case.get("case_id"),
                    "verdict": verdict,
                    "extracted_eps": case.get("extracted_eps"),
                    "filed_eps": case.get("filed_eps"),
                    "ticker": article.get("ticker", ""),
                    "period": period,
                    "label_date": case.get("recorded_at") or article.get("publish_at"),
                    "license_class": "own-exhaust",
                },
            })
            confirmed += 1

    if verbose:
        print(f"  eps-headlines: {confirmed} confirmed records "
              f"({excluded_pending} pending, {excluded_contradicted} contradicted, "
              f"{skipped_no_oracle} no oracle case -- all excluded, not silently dropped)")
    return records


def load_eval_tombstones(gpt2_root: Path, verbose: bool) -> tuple[set[str], str]:
    """S146-04: load the eval contamination tombstone list.

    Returns (set of tombstoned source_event_hash values, hash of the tombstone
    list itself for the snapshot manifest). Missing file is not an error --
    S145-02 (fableeval freeze) hasn't run yet in a fresh checkout, so an empty
    tombstone set with a stable empty-list hash is the correct default, not a
    warning-worthy gap.
    """
    path = gpt2_root / "var" / "eval-tombstones.json"
    if not path.exists():
        empty_hash = hashlib.sha256(b"[]").hexdigest()
        return set(), empty_hash
    raw = path.read_text()
    entries = json.loads(raw)
    hashes = {e["sha256"] for e in entries if "sha256" in e}
    list_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if verbose:
        print(f"  eval-tombstones: {len(hashes)} frozen hashes loaded from {path}")
    return hashes, list_hash


def apply_tombstones(records: list[dict], tombstones: set[str], verbose: bool) -> list[dict]:
    """Drop any record whose source_event_hash is in the tombstone set.

    Applied after generation, before dedupe/write, per HQ-SPEC-AI-103 §8 step 2:
    the eval suite must be excluded from training data, mechanically, by hash --
    not by hoping the eval and training splits never overlap by construction.
    """
    if not tombstones:
        return records
    kept = []
    removed = 0
    for rec in records:
        h = (rec.get("_provenance") or {}).get("source_event_hash")
        if h and h in tombstones:
            removed += 1
            continue
        kept.append(rec)
    if verbose and removed:
        print(f"  eval-tombstones: removed {removed} record(s) present in the frozen eval suite")
    return kept


def _git_rev(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_snapshot(records: list[dict], gpt2_root: Path, cli_args: list[str], tombstone_list_hash: str,
                    verbose: bool) -> Path:
    """S146-01: write an immutable, content-addressed snapshot pair.

    gpt2-alpine-c/var/snapshots/eps-<date>-<shorthash>.{jsonl,manifest.json}.
    snapshot_id = sha256 over the sorted list of per-record source_event_hash
    values -- this is the contract Go `fabledata` inherits (HQ-SPEC-AI-103 §4a)
    once it's built; this Python writer defines the format first.
    """
    snapshots_dir = gpt2_root / "var" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    record_hashes = sorted(
        h for h in ((r.get("_provenance") or {}).get("source_event_hash") for r in records) if h
    )
    if not record_hashes:
        # No provenance-bearing records (e.g. an empty EPS oracle) -- hash the
        # empty set rather than refuse to snapshot. An empty snapshot is a
        # legitimate, honest state, not an error.
        pass
    snapshot_id = hashlib.sha256("".join(record_hashes).encode("utf-8")).hexdigest()
    short_hash = snapshot_id[:12]
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    base_name = f"eps-{date_str}-{short_hash}"
    jsonl_path = snapshots_dir / f"{base_name}.jsonl"
    manifest_path = snapshots_dir / f"{base_name}.manifest.json"

    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "builder_git_rev": _git_rev(gpt2_root),
        "builder_args": cli_args,
        "tombstone_list_hash": tombstone_list_hash,
        "jsonl_file": jsonl_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if verbose:
        print(f"  snapshot: {jsonl_path} ({len(records)} records, snapshot_id={short_hash})")
    return jsonl_path


TYLER_DIRS = ["episodes", "lore", "engine", "characters", "manuscripts", "memos", "outlines"]
TYLER_ROOT_FILES = ["README.md", "universe_engine.md", "THE_FIELD.md", "CITY_OF_LIGHT.md"]


def tyler_to_records(tyler_root: Path, verbose: bool) -> list[dict]:
    """Chunk all TYLER episode scripts, lore, and docs into LM records."""
    records = []
    paths: list[Path] = []

    for fname in TYLER_ROOT_FILES:
        p = tyler_root / fname
        if p.exists():
            paths.append(p)

    for dirname in TYLER_DIRS:
        d = tyler_root / dirname
        if d.exists():
            paths.extend(sorted(d.rglob("*.md")))

    skipped = 0
    for p in paths:
        try:
            text = p.read_text().strip()
            if len(text) < 50:
                skipped += 1
                continue
            rel = str(p.relative_to(tyler_root))
            for chunk in chunk_text(text):
                if len(chunk.strip()) >= 50:
                    records.append({"text": chunk, "_source": f"tyler/{rel}"})
        except OSError:
            skipped += 1

    if verbose:
        print(f"  tyler: {len(records)} chunks from {len(paths) - skipped} files ({skipped} skipped)")
    return records


def build_corpus(
    emily_root: Path,
    mode: str,
    verbose: bool,
    apples_dir: Path | None = None,
    max_apples: int = 500,
    fatbaby_root: Path | None = None,
    max_sec_docs: int = 2000,
    max_pr_docs: int = 1000,
    tyler_root: Path | None = None,
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

    # 7. SEC filing text (cleaned_text from source_document_persisted events)
    if fatbaby_root:
        r = sec_filings_to_records(fatbaby_root, max_sec_docs, verbose)
        records.extend(r)

    # 8. Press release bodies (pr_body_fetched events from prwatch-body)
    if fatbaby_root:
        r = press_releases_to_records(fatbaby_root, max_pr_docs, verbose)
        records.extend(r)

    # 9. TYLER episode scripts, lore, and universe docs
    if tyler_root and tyler_root.exists():
        r = tyler_to_records(tyler_root, verbose)
        records.extend(r)
    elif tyler_root:
        if verbose:
            print(f"  WARNING: tyler-root not found: {tyler_root}", file=sys.stderr)

    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicate text content by hash. Returns (deduped, n_removed)."""
    seen: set[int] = set()
    out = []
    for rec in records:
        key = rec.get("text") or (rec.get("prompt", "") + rec.get("completion", ""))
        h = hash(key)
        if h not in seen:
            seen.add(h)
            out.append(rec)
    return out, len(records) - len(out)


def stratified_sample(records: list[dict], max_records: int) -> list[dict]:
    """Downsample to max_records, preserving source distribution."""
    if max_records <= 0 or len(records) <= max_records:
        return records
    # Group by source
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        src = rec.get("_source", "unknown")
        by_source.setdefault(src, []).append(rec)
    # Sample proportionally from each source
    ratio = max_records / len(records)
    sampled = []
    for src, recs in by_source.items():
        n = max(1, round(len(recs) * ratio))
        step = max(1, len(recs) // n)
        sampled.extend(recs[::step][:n])
    # Trim or pad to exact max_records
    if len(sampled) > max_records:
        sampled = sampled[:max_records]
    return sampled


def load_game_replays(replay_dir: "Path", max_records: int = 0, verbose: bool = False) -> list:
    """Load SHANKPIT replay NDJSON files as instruction pairs {prompt, completion, _source}."""
    from pathlib import Path as _Path
    records = []
    ndjson_files = sorted(_Path(replay_dir).glob("*.ndjson"))
    for fpath in ndjson_files:
        with fpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    state = rec.get("state", "")
                    action = rec.get("action", "")
                    if state and action:
                        records.append({
                            "prompt": state,
                            "completion": action,
                            "_source": "game_replay",
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
        if max_records > 0 and len(records) >= max_records:
            break
    if max_records > 0:
        records = records[:max_records]
    if verbose:
        print(f"  [game_replay] loaded {len(records)} pairs from {len(ndjson_files)} file(s)")
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
    parser.add_argument("--fatbaby-root", default=None,
                        help="Path to PRRJECT_FATBABY repo (explicit only — no auto-discover)")
    parser.add_argument("--max-sec-docs", type=int, default=2000,
                        help="Max SEC filing docs to include (0=unlimited, default: 2000)")
    parser.add_argument("--max-pr-docs", type=int, default=1000,
                        help="Max press release docs to include (0=unlimited, default: 1000)")
    parser.add_argument("--tyler-root", default=None,
                        help="Path to TYLER repo (explicit only — no auto-discover)")
    # Exclusion flags
    parser.add_argument("--no-sec", action="store_true",
                        help="Exclude SEC filing text (overrides --fatbaby-root sec ingestion)")
    parser.add_argument("--no-press-releases", action="store_true",
                        help="Exclude press release text")
    parser.add_argument("--no-tyler", action="store_true",
                        help="Exclude TYLER episode/lore text")
    # Quality flags
    parser.add_argument("--dedupe", action="store_true", default=True,
                        help="Remove duplicate records by content hash (default: on)")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                        help="Disable deduplication")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Cap total output records via stratified sampling (0=unlimited)")
    # Preset
    parser.add_argument("--colab", action="store_true",
                        help="Colab-safe preset: Emily operational text only, deduped, max 1500 records."
                             " Implies: --no-sec --no-press-releases --no-tyler --max-apples 200 --max-records 1500")
    parser.add_argument("--fable-eps", action="store_true",
                        help="S146-03: EPS-headline-only preset for FABLE E0 (HQ-SPEC-AI-103). "
                             "Track B -- confirmed-verdict EPS records only, no golden docs / TYLER / "
                             "Apples / chunked SEC text. Implies --snapshot. Requires --fatbaby-root.")
    parser.add_argument("--snapshot", action="store_true",
                        help="S146-01: write an immutable, content-addressed snapshot pair "
                             "(var/snapshots/eps-<date>-<hash>.jsonl + .manifest.json) instead of "
                             "(in addition to, if --output is also given) the plain --output write.")
    parser.add_argument("--gpt2-root", default=None,
                        help="Path to gpt2-alpine-c repo root (default: this script's repo). "
                             "Used for --snapshot output and --fable-eps tombstone loading.")
    parser.add_argument("--game-replays", default=None,
                        help="Directory of SHANKPIT replay NDJSON files to include as game instruction pairs")
    parser.add_argument("--max-game-records", type=int, default=0,
                        help="Cap game replay records (0=unlimited, default 0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Apply --colab preset before anything else
    if args.colab:
        args.no_sec = True
        args.no_press_releases = True
        args.no_tyler = True
        args.dedupe = True
        if args.max_apples == 500:   # only override if still at default
            args.max_apples = 200
        if args.max_records == 0:
            args.max_records = 1500

    if args.fable_eps:
        args.snapshot = True
        if not args.fatbaby_root:
            print("ERROR: --fable-eps requires --fatbaby-root (var/eps/ lives there)", file=sys.stderr)
            sys.exit(1)

    gpt2_root = Path(args.gpt2_root) if args.gpt2_root else Path(__file__).resolve().parent.parent

    emily_root = Path(args.emily_root)
    if not emily_root.exists():
        print(f"ERROR: EMILY root not found: {emily_root}", file=sys.stderr)
        sys.exit(1)

    repo_parent = emily_root.parent

    # Auto-discover APPLES as sibling of EMILY if not specified
    apples_dir: Path | None = None
    if args.apples_dir:
        apples_dir = Path(args.apples_dir)
    else:
        candidate = repo_parent / "APPLES"
        if candidate.exists():
            apples_dir = candidate

    # PRRJECT_FATBABY — explicit only (no auto-discover to avoid corpus explosion)
    fatbaby_root: Path | None = None
    if args.fatbaby_root:
        fatbaby_root = Path(args.fatbaby_root)

    # TYLER — explicit only
    tyler_root: Path | None = None
    if args.tyler_root:
        tyler_root = Path(args.tyler_root)

    # Apply exclusion flags after auto-discovery
    if args.no_sec or args.no_press_releases:
        fatbaby_root = None  # nothing to read from fatbaby if both excluded
    if args.no_tyler:
        tyler_root = None

    if args.verbose and not args.fable_eps:
        preset = " [--colab preset]" if args.colab else ""
        print(f"Building corpus from {emily_root} (mode={args.mode}){preset}")
        if apples_dir:
            print(f"  Apples dir:   {apples_dir} (max {args.max_apples})")
        if fatbaby_root:
            print(f"  FatBaby root: {fatbaby_root} (sec max {args.max_sec_docs}, pr max {args.max_pr_docs})")
        else:
            print(f"  SEC/press:    excluded")
        if tyler_root:
            print(f"  TYLER root:   {tyler_root}")
        else:
            print(f"  TYLER:        excluded")
        if args.dedupe:
            print(f"  Deduplication: enabled")
        if args.max_records > 0:
            print(f"  Max records:  {args.max_records} (stratified)")

    if args.fable_eps:
        # Track B (S146-03): EPS-headline records only. Bypasses build_corpus
        # entirely -- golden docs / TYLER / Apples / chunked SEC text are all
        # out of scope for this preset by design, not by exclusion flags.
        if args.verbose:
            print(f"Building --fable-eps snapshot from {fatbaby_root}")
        records = eps_headlines_to_records(fatbaby_root, args.verbose)
    else:
        records = build_corpus(
            emily_root, args.mode, args.verbose,
            apples_dir=apples_dir, max_apples=args.max_apples,
            fatbaby_root=fatbaby_root,
            max_sec_docs=args.max_sec_docs,
            max_pr_docs=args.max_pr_docs,
            tyler_root=tyler_root,
        )

    # Game replay pairs (30% of total when mixed corpus)
    if args.game_replays and not args.fable_eps:
        game_records = load_game_replays(Path(args.game_replays), args.max_game_records, args.verbose)
        records.extend(game_records)
        if args.verbose:
            print(f"  Game replays: {len(game_records)} instruction pairs from {args.game_replays}")

    # Deduplication
    if args.dedupe:
        records, n_removed = deduplicate(records)
        if args.verbose and n_removed:
            print(f"  Deduplication: removed {n_removed} duplicates → {len(records)} records")

    # S146-04: eval contamination tombstoning — after generation/dedupe, before write.
    tombstones, tombstone_list_hash = load_eval_tombstones(gpt2_root, args.verbose)
    records = apply_tombstones(records, tombstones, args.verbose)

    # Stratified cap
    if args.max_records > 0 and len(records) > args.max_records:
        before = len(records)
        records = stratified_sample(records, args.max_records)
        if args.verbose:
            print(f"  Capped: {before} → {len(records)} records (stratified)")

    if args.snapshot:
        # S146-01: immutable, content-addressed snapshot — the contract Go
        # fabledata inherits once it exists (HQ-SPEC-AI-103 §4a).
        write_snapshot(records, gpt2_root, sys.argv[1:], tombstone_list_hash, args.verbose)

    # Write output — keep _source metadata (ignored by HuggingFace Trainer, used by corpus_stats.py)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
