#!/usr/bin/env python3
"""
Unit tests for scripts/corpus_stats.py's provenance audit mode (S146-05).

Run with: python3 -m pytest tests/test_corpus_stats.py -v
     or:  python3 tests/test_corpus_stats.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from corpus_stats import (
    check_contamination,
    load_provenance_records,
    load_tombstone_hashes,
    provenance_breakdown,
    run_provenance_audit,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestLoadProvenanceRecords(unittest.TestCase):
    def test_keeps_only_records_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            write_jsonl(path, [
                {"text": "has provenance", "_provenance": {"verdict": "confirmed"}},
                {"text": "no provenance"},
                {"text": "empty provenance", "_provenance": {}},
            ])
            records = load_provenance_records(path)
            # empty dict is falsy -- correctly excluded, same as missing entirely
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["text"], "has provenance")

    def test_empty_corpus_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text("")
            self.assertEqual(load_provenance_records(path), [])

    def test_skips_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text('{"_provenance": {"verdict": "confirmed"}}\nnot json\n')
            records = load_provenance_records(path)
            self.assertEqual(len(records), 1)


class TestProvenanceBreakdown(unittest.TestCase):
    def test_counts_by_verdict_source_license(self):
        records = [
            {"_source": "eps-headlines", "_provenance": {"verdict": "confirmed", "license_class": "own-exhaust"}},
            {"_source": "eps-headlines", "_provenance": {"verdict": "confirmed", "license_class": "own-exhaust"}},
            {"_source": "eps-headlines", "_provenance": {"verdict": "contradicted", "license_class": "own-exhaust"}},
        ]
        breakdown = provenance_breakdown(records)
        self.assertEqual(breakdown["verdicts"], {"confirmed": 2, "contradicted": 1})
        self.assertEqual(breakdown["sources"], {"eps-headlines": 3})
        self.assertEqual(breakdown["license_classes"], {"own-exhaust": 3})

    def test_missing_fields_bucket_as_unknown(self):
        records = [{"_provenance": {}}]
        breakdown = provenance_breakdown(records)
        self.assertEqual(breakdown["verdicts"], {"unknown": 1})
        self.assertEqual(breakdown["sources"], {"unknown": 1})
        self.assertEqual(breakdown["license_classes"], {"unknown": 1})

    def test_empty_records_all_empty(self):
        breakdown = provenance_breakdown([])
        self.assertEqual(breakdown["verdicts"], {})
        self.assertEqual(breakdown["sources"], {})
        self.assertEqual(breakdown["license_classes"], {})


class TestLoadTombstoneHashes(unittest.TestCase):
    def test_missing_file_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_tombstone_hashes(Path(tmp)), set())

    def test_loads_hashes_from_real_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            (gpt2_root / "var").mkdir()
            entries = [
                {"sha256": "hash-a", "suite": "eps-v1", "frozen_at": "2026-08-15"},
                {"sha256": "hash-b", "suite": "eps-v1", "frozen_at": "2026-08-15"},
            ]
            (gpt2_root / "var" / "eval-tombstones.json").write_text(json.dumps(entries))
            self.assertEqual(load_tombstone_hashes(gpt2_root), {"hash-a", "hash-b"})


class TestCheckContamination(unittest.TestCase):
    def test_no_tombstones_no_findings(self):
        records = [{"_provenance": {"source_event_hash": "h1"}}]
        self.assertEqual(check_contamination(records, set()), [])

    def test_clean_snapshot_no_overlap(self):
        records = [{"_provenance": {"source_event_hash": "h1"}}]
        self.assertEqual(check_contamination(records, {"some-other-hash"}), [])

    def test_detects_contaminated_record(self):
        records = [
            {"_provenance": {"source_event_hash": "clean-hash"}},
            {"_provenance": {"source_event_hash": "tombstoned-hash"}},
        ]
        found = check_contamination(records, {"tombstoned-hash"})
        self.assertEqual(found, ["tombstoned-hash"])

    def test_detects_multiple_contaminated_records(self):
        records = [
            {"_provenance": {"source_event_hash": "bad-1"}},
            {"_provenance": {"source_event_hash": "bad-2"}},
        ]
        found = check_contamination(records, {"bad-1", "bad-2"})
        self.assertEqual(sorted(found), ["bad-1", "bad-2"])


class TestRunProvenanceAudit(unittest.TestCase):
    def test_clean_snapshot_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            corpus_path = gpt2_root / "snapshot.jsonl"
            write_jsonl(corpus_path, [
                {"_source": "eps-headlines",
                 "_provenance": {"verdict": "confirmed", "license_class": "own-exhaust",
                                  "source_event_hash": "clean-hash"}},
            ])
            code = run_provenance_audit(corpus_path, gpt2_root, verbose=False)
            self.assertEqual(code, 0)

    def test_contaminated_snapshot_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            (gpt2_root / "var").mkdir()
            (gpt2_root / "var" / "eval-tombstones.json").write_text(
                json.dumps([{"sha256": "tombstoned-hash", "suite": "eps-v1", "frozen_at": "2026-08-15"}])
            )
            corpus_path = gpt2_root / "snapshot.jsonl"
            write_jsonl(corpus_path, [
                {"_source": "eps-headlines",
                 "_provenance": {"verdict": "confirmed", "license_class": "own-exhaust",
                                  "source_event_hash": "tombstoned-hash"}},
            ])
            code = run_provenance_audit(corpus_path, gpt2_root, verbose=False)
            self.assertEqual(code, 1)

    def test_no_provenance_records_is_clean_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            corpus_path = gpt2_root / "general.jsonl"
            write_jsonl(corpus_path, [{"text": "no provenance here"}])
            code = run_provenance_audit(corpus_path, gpt2_root, verbose=False)
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
