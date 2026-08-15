#!/usr/bin/env python3
"""
Unit tests for scripts/prime_directive_dataset.py.

Run with: python3 -m pytest tests/test_dataset.py -v
     or:  python3 tests/test_dataset.py
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Resolve script location regardless of working dir
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prime_directive_dataset import (
    apples_to_records,
    apply_tombstones,
    backlog_to_instruct,
    build_corpus,
    chunk_text,
    eps_headlines_to_records,
    load_eval_tombstones,
    make_lm_records,
    parse_golden_index,
    prime_directive_to_instruct,
    towerprint_augmented_records,
    write_snapshot,
    _stable_sample_keep,
)


class TestChunkText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text(""), [])

    def test_single_paragraph_under_limit(self):
        chunks = chunk_text("hello world", size=200)
        self.assertEqual(len(chunks), 1)
        self.assertIn("hello world", chunks[0])

    def test_splits_at_paragraph_boundary(self):
        para_a = "A " * 100
        para_b = "B " * 100
        text = para_a + "\n\n" + para_b
        chunks = chunk_text(text, size=150)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("A" in c for c in chunks))
        self.assertTrue(any("B" in c for c in chunks))

    def test_overlap_keeps_last_paragraph(self):
        paras = [f"Para{i} " * 50 for i in range(5)]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, size=200)
        # Each chunk except possibly the last should contain at least 1 paragraph
        for c in chunks:
            self.assertGreater(len(c.strip()), 0)

    def test_strips_whitespace_only_paragraphs(self):
        text = "  \n\nreal content here\n\n   \n\nmore content"
        chunks = chunk_text(text, size=500)
        for c in chunks:
            self.assertNotEqual(c.strip(), "")


class TestMakeLmRecords(unittest.TestCase):
    def test_basic(self):
        records = make_lm_records("Hello world.\n\nThis is a test.", source="test")
        self.assertIsInstance(records, list)
        for r in records:
            self.assertIn("text", r)
            self.assertIn("_source", r)
            self.assertEqual(r["_source"], "test")

    def test_skips_short_chunks(self):
        # Chunks under 50 chars should be skipped
        records = make_lm_records("Hi.", source="test")
        self.assertEqual(records, [])


class TestBacklogToInstruct(unittest.TestCase):
    def test_extracts_done_items(self):
        backlog = "- [x] implement Drive upload API\n- [ ] pending task\n- [x] write unit tests for corpus builder"
        pairs = backlog_to_instruct(backlog)
        self.assertEqual(len(pairs), 2)
        for p in pairs:
            self.assertIn("prompt", p)
            self.assertIn("completion", p)

    def test_skips_short_items(self):
        backlog = "- [x] foo"  # too short (< 20 chars)
        pairs = backlog_to_instruct(backlog)
        self.assertEqual(pairs, [])

    def test_prompt_contains_item(self):
        backlog = "- [x] implement Emily Prime corpus dataset builder"
        pairs = backlog_to_instruct(backlog)
        self.assertEqual(len(pairs), 1)
        self.assertIn("implement Emily Prime corpus dataset builder", pairs[0]["prompt"])

    def test_completion_contains_changelog_entry(self):
        backlog = "- [x] implement Emily Prime corpus dataset builder"
        pairs = backlog_to_instruct(backlog)
        self.assertIn("CHANGELOG entry", pairs[0]["completion"])


class TestParseGoldenIndex(unittest.TestCase):
    def test_parses_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            emily_root = Path(tmp)
            ctx_dir = emily_root / "context"
            ctx_dir.mkdir()
            index = ctx_dir / "golden-docs-index.md"
            index.write_text(
                "| Name | Path | Tier | Budget | Description |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| FATBABY | `PRRJECT_FATBABY/NORTHSTAR.md` | 1 | 0 | FatBaby northstar |\n"
                "| IDUNA | `IDUNA/NORTHSTAR.md` | 2 | 0 | IDUNA northstar |\n"
                "| OLD | `old/doc.md` | 3 | 0 | Old doc |\n"
            )
            entries = parse_golden_index(emily_root)
        self.assertEqual(len(entries), 3)
        tier1 = [e for e in entries if e["tier"] == 1]
        self.assertEqual(len(tier1), 1)
        self.assertEqual(tier1[0]["name"], "FATBABY")

    def test_missing_index_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = parse_golden_index(Path(tmp))
        self.assertEqual(entries, [])


class TestPrimeDirectiveInstruct(unittest.TestCase):
    SAMPLE_PRIME = """# Emily's Prime Directive
## Executive Alignment

**Emily's Core Purpose:** Use AI to build tools that collect data to train in-house models.

Value Chain: Data Collection → Data Processing → Training Dataset → LLM

## 1. DATA COLLECTION MISSION

Emily collects high-quality data from Reddit and Wikipedia at scale.
Quality filters ensure only substantive content reaches the training pipeline.
Daily target: 10,000 posts from 50+ subreddits.

## 2. ITERATIVE RSI LOOPS

The RSI (Recursive Self-Improvement) loop runs every 5 minutes:
OBSERVE → DECIDE → ACT → PLAN → file Apple → repeat.
"""

    def test_returns_list_of_pairs(self):
        pairs = prime_directive_to_instruct(self.SAMPLE_PRIME)
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)

    def test_all_pairs_have_prompt_and_completion(self):
        pairs = prime_directive_to_instruct(self.SAMPLE_PRIME)
        for p in pairs:
            self.assertIn("prompt", p, f"Missing prompt: {p}")
            self.assertIn("completion", p, f"Missing completion: {p}")

    def test_fixed_identity_pair_present(self):
        pairs = prime_directive_to_instruct(self.SAMPLE_PRIME)
        prompts = [p["prompt"] for p in pairs]
        self.assertTrue(any("Who are you" in pr or "core purpose" in pr for pr in prompts))

    def test_section_pairs_extracted(self):
        pairs = prime_directive_to_instruct(self.SAMPLE_PRIME)
        # Should have at least the fixed pairs + one section pair
        self.assertGreater(len(pairs), 8)

    def test_short_sections_skipped(self):
        short_prime = "## Empty Section\n\nShort.\n"
        pairs = prime_directive_to_instruct(short_prime)
        # Only fixed pairs, no section pairs for short content
        section_pairs = [p for p in pairs if "Describe Emily Prime" in p["prompt"]]
        self.assertEqual(len(section_pairs), 0)

    def test_long_sections_truncated(self):
        long_body = "Detail. " * 300
        long_prime = f"## Long Section\n\n{long_body}\n"
        pairs = prime_directive_to_instruct(long_prime)
        section_pairs = [p for p in pairs if "Long Section" in p["prompt"]]
        if section_pairs:
            self.assertLessEqual(len(section_pairs[0]["completion"]), 1500)


class TestApplesRecords(unittest.TestCase):
    def _make_apple(self, tmp: Path, apple_id: int, apple_type: str,
                    title: str, body: str, source_repo: str = "EMILY") -> None:
        day_dir = tmp / "20260614"
        day_dir.mkdir(exist_ok=True)
        data = {
            "id": apple_id,
            "apple_type": apple_type,
            "title": title,
            "body": body,
            "source_repo": source_repo,
            "archived_at": "2026-06-14T01:00:00Z",
        }
        (day_dir / f"{apple_id}_{apple_type}.json").write_text(json.dumps(data))

    def test_reads_apple_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            apples_dir = Path(tmp)
            self._make_apple(apples_dir, 1, "completion",
                             "S26-01 IDUNA Drive API done", "Implemented Drive upload endpoint.")
            self._make_apple(apples_dir, 2, "signal_observation",
                             "FatBaby obs: RSI loop latency spike", "severity: warn\nbody: 2s lag")
            records = apples_to_records(apples_dir, max_apples=0, verbose=False)
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertIn("text", r)
            self.assertIn("Apple [", r["text"])
            self.assertEqual(r["_source"], "apples")

    def test_skips_empty_apples(self):
        with tempfile.TemporaryDirectory() as tmp:
            apples_dir = Path(tmp)
            day_dir = apples_dir / "20260614"
            day_dir.mkdir()
            (day_dir / "99_empty.json").write_text(json.dumps({"id": 99, "apple_type": "x"}))
            records = apples_to_records(apples_dir, max_apples=0, verbose=False)
        self.assertEqual(len(records), 0)

    def test_respects_max_apples(self):
        with tempfile.TemporaryDirectory() as tmp:
            apples_dir = Path(tmp)
            for i in range(10):
                self._make_apple(apples_dir, i, "completion", f"Title {i}", f"Body {i}")
            records = apples_to_records(apples_dir, max_apples=3, verbose=False)
        self.assertEqual(len(records), 3)

    def test_truncates_long_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            apples_dir = Path(tmp)
            long_body = "x" * 5000
            self._make_apple(apples_dir, 1, "completion", "Long Apple", long_body)
            records = apples_to_records(apples_dir, max_apples=0, verbose=False)
        self.assertEqual(len(records), 1)
        self.assertIn("[...]", records[0]["text"])

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = apples_to_records(Path(tmp), max_apples=0, verbose=False)
        self.assertEqual(records, [])


class TestBuildCorpus(unittest.TestCase):
    def _setup_emily(self, tmp: Path) -> Path:
        emily_root = tmp / "EMILY"
        emily_root.mkdir()
        ctx = emily_root / "context"
        ctx.mkdir()

        # Minimal golden-docs-index with one Tier 1 doc
        (ctx / "golden-docs-index.md").write_text(
            "| Name | Path | Tier | Budget | Description |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| TEST-DOC | EMILY/docs/test.md | 1 | 0 | Test doc |\n"
        )

        docs = emily_root / "docs"
        docs.mkdir()
        (docs / "test.md").write_text(
            "# Test Doc\n\nThis is the test golden doc. " * 20
        )
        (docs / "emily-prime-directive-data-collection.md").write_text(
            "# Emily Prime Directive\n\nCore principles for Emily Prime operation. " * 10
        )

        return emily_root

    def test_build_lm_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            emily_root = self._setup_emily(Path(tmp))
            records = build_corpus(emily_root, mode="lm", verbose=False)
        self.assertGreater(len(records), 0)
        for r in records:
            self.assertIn("text", r)

    def test_build_instruct_mode_adds_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            emily_root = self._setup_emily(Path(tmp))
            # Add a BACKLOG with done items
            (emily_root / "BACKLOG.md").write_text(
                "- [x] implement corpus dataset builder for Emily Prime\n"
                "- [ ] pending task not done\n"
            )
            records = build_corpus(emily_root, mode="instruct", verbose=False)
        instruct = [r for r in records if "prompt" in r]
        self.assertGreater(len(instruct), 0)

    def test_with_apples_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            emily_root = self._setup_emily(Path(tmp))
            apples_dir = Path(tmp) / "APPLES"
            apples_dir.mkdir()
            day_dir = apples_dir / "20260614"
            day_dir.mkdir()
            apple = {
                "id": 1, "apple_type": "completion",
                "title": "GPT-2 fine-tune complete", "body": "Checkpoint saved to Drive.",
                "source_repo": "gpt2-alpine-c", "archived_at": "2026-06-14T01:00:00Z",
            }
            (day_dir / "1_completion.json").write_text(json.dumps(apple))

            records = build_corpus(emily_root, mode="lm", verbose=False,
                                   apples_dir=apples_dir)
        apple_records = [r for r in records if r.get("_source") == "apples"]
        self.assertEqual(len(apple_records), 1)

    def test_output_is_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            emily_root = self._setup_emily(Path(tmp))
            records = build_corpus(emily_root, mode="lm", verbose=False)
            out_path = Path(tmp) / "out.jsonl"
            with out_path.open("w") as f:
                for r in records:
                    out_rec = {k: v for k, v in r.items() if not k.startswith("_")}
                    f.write(json.dumps(out_rec) + "\n")

            # Re-parse every line
            with out_path.open() as f:
                for line in f:
                    obj = json.loads(line)
                    self.assertIn("text", obj)


class TestEpsHeadlinesToRecords(unittest.TestCase):
    def _write_ndjson(self, path: Path, lines: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for obj in lines:
                f.write(json.dumps(obj) + "\n")

    def test_missing_store_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = eps_headlines_to_records(Path(tmp), verbose=False)
            self.assertEqual(records, [])

    def test_confirmed_case_included_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            fatbaby_root = Path(tmp)
            article = {
                "source_identity": "pr:1", "ticker": "AAPL",
                "headline": "Apple reports Q3 EPS of $1.50", "dek": "GAAP diluted EPS: $1.50",
                "body": "Period: Q3 2026", "period": {"fiscal_quarter": "Q3", "fiscal_year": 2026},
                "eps_value": 1.5, "publish_at": "2026-07-01T00:00:00Z",
            }
            self._write_ndjson(fatbaby_root / "var" / "eps" / "articles.ndjson", [article])
            self._write_ndjson(fatbaby_root / "var" / "eps" / "oracle.ndjson", [{
                "case_id": "eps:abc", "source_identity": "pr:1", "ticker": "AAPL",
                "extracted_eps": 1.5, "filed_eps": 1.5, "verdict": "confirmed",
                "recorded_at": "2026-07-02",
            }])

            records = eps_headlines_to_records(fatbaby_root, verbose=False)
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["_source"], "eps-headlines")
            self.assertIn("Apple reports Q3 EPS", rec["text"])
            prov = rec["_provenance"]
            self.assertEqual(prov["oracle_case_id"], "eps:abc")
            self.assertEqual(prov["verdict"], "confirmed")
            self.assertEqual(prov["license_class"], "own-exhaust")
            self.assertTrue(len(prov["source_event_hash"]) == 64)  # sha256 hex

    def test_pending_and_contradicted_excluded_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            fatbaby_root = Path(tmp)
            articles = [
                {"source_identity": "pr:1", "headline": "H1", "ticker": "A"},
                {"source_identity": "pr:2", "headline": "H2", "ticker": "B"},
                {"source_identity": "pr:3", "headline": "H3", "ticker": "C"},
            ]
            self._write_ndjson(fatbaby_root / "var" / "eps" / "articles.ndjson", articles)
            self._write_ndjson(fatbaby_root / "var" / "eps" / "oracle.ndjson", [
                {"case_id": "1", "source_identity": "pr:1", "verdict": "pending"},
                {"case_id": "2", "source_identity": "pr:2", "verdict": "contradicted"},
                {"case_id": "3", "source_identity": "pr:3", "verdict": "confirmed"},
            ])
            records = eps_headlines_to_records(fatbaby_root, verbose=False)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["_provenance"]["oracle_case_id"], "3")

    def test_article_without_oracle_case_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            fatbaby_root = Path(tmp)
            self._write_ndjson(fatbaby_root / "var" / "eps" / "articles.ndjson",
                                [{"source_identity": "pr:orphan", "headline": "H"}])
            self._write_ndjson(fatbaby_root / "var" / "eps" / "oracle.ndjson", [])
            records = eps_headlines_to_records(fatbaby_root, verbose=False)
            self.assertEqual(records, [])

    def test_never_chunked_even_if_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            fatbaby_root = Path(tmp)
            long_body = "word " * 2000  # far exceeds CHUNK_SIZE
            self._write_ndjson(fatbaby_root / "var" / "eps" / "articles.ndjson",
                                [{"source_identity": "pr:1", "headline": "H", "body": long_body}])
            self._write_ndjson(fatbaby_root / "var" / "eps" / "oracle.ndjson",
                                [{"case_id": "1", "source_identity": "pr:1", "verdict": "confirmed"}])
            records = eps_headlines_to_records(fatbaby_root, verbose=False)
            self.assertEqual(len(records), 1)  # one record, not chunked into many


class TestEvalTombstones(unittest.TestCase):
    def test_missing_file_returns_empty_set_stable_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            hashes, list_hash = load_eval_tombstones(Path(tmp), verbose=False)
            self.assertEqual(hashes, set())
            self.assertEqual(list_hash, hashlib.sha256(b"[]").hexdigest())

    def test_loads_hashes_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            (gpt2_root / "var").mkdir()
            entries = [{"sha256": "abc123", "suite": "eps-v1", "frozen_at": "2026-07-16"}]
            (gpt2_root / "var" / "eval-tombstones.json").write_text(json.dumps(entries))
            hashes, list_hash = load_eval_tombstones(gpt2_root, verbose=False)
            self.assertEqual(hashes, {"abc123"})
            self.assertNotEqual(list_hash, hashlib.sha256(b"[]").hexdigest())

    def test_apply_tombstones_removes_matching_records(self):
        records = [
            {"text": "keep", "_provenance": {"source_event_hash": "keep-hash"}},
            {"text": "drop", "_provenance": {"source_event_hash": "drop-hash"}},
            {"text": "no-provenance"},
        ]
        result = apply_tombstones(records, {"drop-hash"}, verbose=False)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["text"] for r in result], ["keep", "no-provenance"])

    def test_apply_tombstones_noop_on_empty_set(self):
        records = [{"text": "a"}, {"text": "b"}]
        result = apply_tombstones(records, set(), verbose=False)
        self.assertEqual(result, records)


class TestWriteSnapshot(unittest.TestCase):
    def test_writes_jsonl_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpt2_root = Path(tmp)
            records = [
                {"text": "a", "_provenance": {"source_event_hash": "h1"}},
                {"text": "b", "_provenance": {"source_event_hash": "h2"}},
            ]
            jsonl_path = write_snapshot(records, gpt2_root, ["--fable-eps"], "tombstone-hash", verbose=False)
            self.assertTrue(jsonl_path.exists())
            manifest_path = jsonl_path.with_suffix("").with_suffix(".manifest.json")
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["record_count"], 2)
            self.assertEqual(manifest["tombstone_list_hash"], "tombstone-hash")
            self.assertEqual(manifest["builder_args"], ["--fable-eps"])
            # snapshot_id is deterministic: sha256 over sorted record hashes
            expected_id = hashlib.sha256("h1h2".encode()).hexdigest()
            self.assertEqual(manifest["snapshot_id"], expected_id)

    def test_snapshot_id_is_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            records_a = [{"_provenance": {"source_event_hash": "h1"}}, {"_provenance": {"source_event_hash": "h2"}}]
            records_b = [{"_provenance": {"source_event_hash": "h2"}}, {"_provenance": {"source_event_hash": "h1"}}]
            p1 = write_snapshot(records_a, Path(tmp), [], "x", verbose=False)
            m1 = json.loads(p1.with_suffix("").with_suffix(".manifest.json").read_text())
            p2 = write_snapshot(records_b, Path(tmp), [], "x", verbose=False)
            m2 = json.loads(p2.with_suffix("").with_suffix(".manifest.json").read_text())
            self.assertEqual(m1["snapshot_id"], m2["snapshot_id"])

    def test_empty_records_snapshot_is_honest_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = write_snapshot([], Path(tmp), [], "x", verbose=False)
            self.assertTrue(jsonl_path.exists())
            manifest = json.loads(jsonl_path.with_suffix("").with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["record_count"], 0)
            self.assertEqual(manifest["snapshot_id"], hashlib.sha256(b"").hexdigest())


class TestStableSampleKeep(unittest.TestCase):
    def test_deterministic_across_calls(self):
        rec = {"text": "a stable sampling test record"}
        first = _stable_sample_keep(rec, 0.5)
        second = _stable_sample_keep(rec, 0.5)
        self.assertEqual(first, second)

    def test_zero_fraction_keeps_nothing(self):
        for i in range(20):
            rec = {"text": f"record {i}"}
            self.assertFalse(_stable_sample_keep(rec, 0.0))

    def test_full_fraction_keeps_everything(self):
        for i in range(20):
            rec = {"text": f"record {i}"}
            self.assertTrue(_stable_sample_keep(rec, 1.0))

    def test_roughly_matches_requested_fraction(self):
        # Not a tight statistical test -- just confirms the sampler isn't
        # wildly miscalibrated (e.g. off by a factor of 2, or inverted).
        kept = sum(
            1 for i in range(2000)
            if _stable_sample_keep({"text": f"record number {i}"}, 0.1)
        )
        self.assertGreater(kept, 100)   # want ~200, allow wide margin
        self.assertLess(kept, 400)


class TestTowerprintAugmentedRecords(unittest.TestCase):
    def _fake_cli(self, tmp: Path, behavior: str = "ok") -> Path:
        """A fake towerprint-cli standing in for the real Go binary, so
        these tests don't depend on `make towerprint-cli` having been run
        first -- same reasoning the rest of this file's tests use fixtures
        instead of hitting real external state.
        """
        script = tmp / "fake-towerprint-cli.py"
        if behavior == "ok":
            script.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, json\n"
                "text = sys.stdin.read()\n"
                "if not any(c.isalpha() for c in text):\n"
                "    print(json.dumps({'error': 'no letters'})); sys.exit(1)\n"
                "print(json.dumps({'tower': ['ROW1', 'ROW2']}))\n"
            )
        elif behavior == "always_fail":
            script.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, json\n"
                "print(json.dumps({'error': 'always fails'})); sys.exit(1)\n"
            )
        script.chmod(0o755)
        return script

    def test_zero_fraction_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._fake_cli(Path(tmp))
            records = [{"text": "some record text"}]
            result = towerprint_augmented_records(records, 0.0, cli, verbose=False)
            self.assertEqual(result, [])

    def test_missing_cli_binary_returns_empty_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            records = [{"text": "some record text"}]
            result = towerprint_augmented_records(records, 1.0, missing, verbose=False)
            self.assertEqual(result, [])

    def test_full_fraction_produces_pairs_for_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._fake_cli(Path(tmp))
            records = [
                {"text": "Emily Prime record one"},
                {"text": "Emily Prime record two"},
            ]
            result = towerprint_augmented_records(records, 1.0, cli, verbose=False)
            self.assertEqual(len(result), 2)
            for rec in result:
                self.assertIn("prompt", rec)
                self.assertIn("completion", rec)
                self.assertEqual(rec["_source"], "towerprint-augmented")
                self.assertEqual(rec["completion"], "ROW1\nROW2")

    def test_cli_failure_skips_record_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._fake_cli(Path(tmp), behavior="always_fail")
            records = [{"text": "some record text"}]
            result = towerprint_augmented_records(records, 1.0, cli, verbose=False)
            self.assertEqual(result, [])

    def test_input_text_capped_before_hashing_to_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._fake_cli(Path(tmp))
            records = [{"text": "x" * 5000}]
            result = towerprint_augmented_records(records, 1.0, cli, verbose=False)
            self.assertEqual(len(result), 1)
            self.assertLessEqual(len(result[0]["prompt"]), 500)

    def test_empty_records_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cli = self._fake_cli(Path(tmp))
            self.assertEqual(towerprint_augmented_records([], 1.0, cli, verbose=False), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
