#!/usr/bin/env python3
"""
Unit tests for scripts/prime_directive_dataset.py.

Run with: python3 -m pytest tests/test_dataset.py -v
     or:  python3 tests/test_dataset.py
"""

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
    backlog_to_instruct,
    build_corpus,
    chunk_text,
    make_lm_records,
    parse_golden_index,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
