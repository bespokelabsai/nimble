"""Interrupted Jev runs must retain the model condition for saved responses."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from nimble.evaluation.evaluate_public_jev import run
from tests.test_evaluate_public import record, response, write_jsonl


def interrupt(done, total):
    raise KeyboardInterrupt


class ResumeManifestTests(unittest.TestCase):
    def test_interruption_keeps_provenance_and_refuses_a_different_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write_jsonl(Path(tmp) / "data.jsonl", [record("a", "entailment")])
            output = Path(tmp) / "run"
            with self.assertRaises(KeyboardInterrupt):
                run(data, output, transport=lambda payload: response(), progress=interrupt)
            self.assertEqual(len((output / "rows.jsonl").read_text().splitlines()), 1)
            transport = Mock(side_effect=AssertionError("No request should be sent"))
            with self.assertRaisesRegex(ValueError, "different data or model"):
                run(data, output, model="another-model", transport=transport)
            transport.assert_not_called()

    def test_same_model_resumes_without_repeating_completed_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write_jsonl(Path(tmp) / "data.jsonl", [record("a", "entailment")])
            output = Path(tmp) / "run"
            with self.assertRaises(KeyboardInterrupt):
                run(data, output, transport=lambda payload: response(), progress=interrupt)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["model"], "jev-1.13.0")
            transport = Mock(side_effect=AssertionError("Completed request was repeated"))
            report = run(data, output, transport=transport)
            self.assertEqual((report["count"], report["resumed"]), (1, True))
            transport.assert_not_called()

    def test_existing_responses_without_provenance_are_not_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = write_jsonl(Path(tmp) / "data.jsonl", [record("a", "entailment")])
            output = Path(tmp) / "run"
            run(data, output, transport=lambda payload: response())
            (output / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "provenance"):
                run(data, output, model="another-model", transport=Mock())
