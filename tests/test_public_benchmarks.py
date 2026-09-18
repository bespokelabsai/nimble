"""Discovery, label-blind selection, and manifest writing of the public-benchmark core."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nimble.datasets.public_benchmarks import build, jsonl_line, load_sources, select, select_ids, validate_source
from nimble.datasets.public_records import read_jsonl
from nimble.datasets.public_sources import boolq

EXPECTED_SOURCES = {"aegis2", "boolq", "civil_comments", "helpsteer2", "massive", "multinli", "paws",
                    "pubmedqa", "squad2", "summeval", "vitaminc"}


def rec(identifier, family, domain="d", target="x"):
    return {"id": identifier, "family": family, "domain": domain, "reference": {"target": target}}


def boolq_raws(count):
    rows = [{"question": f"q{i}", "passage": f"passage {i // 2}", "answer": i % 2 == 0} for i in range(count)]
    rows.append({"question": "bad", "passage": "p", "answer": "yes"})  # skipped by the converter
    return rows


class DiscoveryTests(unittest.TestCase):
    def test_every_source_module_is_discovered(self):
        self.assertEqual(set(load_sources()), EXPECTED_SOURCES)

    def test_contract_violations_name_the_missing_piece(self):
        module = SimpleNamespace(__name__="fake", NAME="fake", SOURCE_URL="u", LICENSE="l", NOTE="n",
                                 SUBSETS=(), rows=lambda p, s="": [], record=lambda r, s="": None)
        with self.assertRaises(ValueError) as caught:
            validate_source(module)
        self.assertIn("missing RELEASE_YEAR", str(caught.exception))
        module.RELEASE_YEAR = 2020
        with self.assertRaises(ValueError) as caught:
            validate_source(module, expected_name="other")
        self.assertIn("named 'other'", str(caught.exception))


class SelectionTests(unittest.TestCase):
    def test_selection_is_label_blind_and_keeps_families_whole(self):
        records = [rec(f"r{i}", f"f{i // 3}", domain="a" if i < 15 else "b", target=str(i)) for i in range(30)]
        chosen = [r["id"] for r in select(records, seed=7, limit=12)]
        relabeled = [dict(r, reference={"target": "changed"}) for r in records]
        self.assertEqual([r["id"] for r in select(relabeled, seed=7, limit=12)], chosen)
        self.assertEqual(len(chosen), 12)
        families = {r["family"] for r in records if r["id"] in chosen}
        self.assertEqual({r["id"] for r in records if r["family"] in families}, set(chosen))
        self.assertNotEqual([r["id"] for r in select(records, seed=8, limit=12)], chosen)

    def test_select_ids_requires_whole_families(self):
        records = [rec("a", "f1"), rec("b", "f1"), rec("c", "f2")]
        self.assertEqual([r["id"] for r in select_ids(records, ["c", "a", "b"])], ["a", "b", "c"])
        with self.assertRaises(ValueError):
            select_ids(records, ["a"])
        with self.assertRaises(ValueError):
            select_ids(records, ["a", "b", "zzz"])


class BuildTests(unittest.TestCase):
    def test_build_writes_payload_and_manifest_and_guards_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "boolq"
            manifest = build("boolq", boolq_raws(8), out, seed=1, limit=4)
            self.assertEqual((manifest["count"], manifest["skipped"], manifest["types"]), (4, 1, {"noul": 4}))
            payload = (out / "all.jsonl").read_bytes()
            self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(payload).hexdigest())
            text = (out / "manifest.json").read_text()
            self.assertEqual(json.loads(text)["ids"], manifest["ids"])
            self.assertEqual(sum(line.startswith('  "ids": [') for line in text.splitlines()), 1)
            build("boolq", boolq_raws(8), out, seed=1, limit=4)  # identical rebuild is accepted
            with self.assertRaises(ValueError):
                build("boolq", boolq_raws(6), out, seed=1, limit=4)
            with self.assertRaises(ValueError):  # no whole family fits, so nothing may be written silently
                build("boolq", boolq_raws(8), Path(tmp) / "none", seed=1, limit=1)

    def test_ids_from_reproduces_another_build_and_excludes_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = build("boolq", boolq_raws(8), Path(tmp) / "one", seed=3, limit=4)
            second = build("boolq", boolq_raws(8), Path(tmp) / "two", ids_from=Path(tmp) / "one" / "manifest.json")
            self.assertEqual(second["ids"], first["ids"])
            self.assertEqual(second["ids_from"]["dataset_sha256"], first["dataset_sha256"])
            with self.assertRaises(ValueError):
                build("boolq", boolq_raws(8), Path(tmp) / "three", limit=2,
                      ids_from=Path(tmp) / "one" / "manifest.json")

    def test_subset_rules_and_unknown_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build("boolq", boolq_raws(2), Path(tmp) / "a", subset="x")
            with self.assertRaises(ValueError):
                build("summeval", [], Path(tmp) / "b")
            with self.assertRaises(ValueError):
                build("nope", [], Path(tmp) / "c")


class LineHandlingTests(unittest.TestCase):
    def test_readers_split_on_newline_only_and_writers_escape_separators(self):
        rows = [{"text": "para graph   linefeed"}, {"text": "plain"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(read_jsonl(path), rows)
            self.assertEqual(boolq.rows(path), rows)
        line = jsonl_line(rows[0])
        self.assertEqual(len(line.splitlines()), 1)
        self.assertEqual(json.loads(line), rows[0])


if __name__ == "__main__":
    unittest.main()
