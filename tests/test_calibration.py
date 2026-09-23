"""Fitted temperatures apply only to the exact checkpoint they were fitted on."""

import json
import tempfile
import unittest
from pathlib import Path

from nimble.scoring.calibration import (ADAPTER_REVISIONS, fitted_temperature,
    resolve_temperature, served_temperature, V2_MODEL, V2_ADAPTER_SHA256, V2_TEMPERATURE)

NIMBLE = "bespokelabs/Bespoke-Nimble-9B"
REVISION = "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"


class FittedTemperatureTests(unittest.TestCase):
    def test_only_the_exact_checkpoint_has_a_temperature(self):
        self.assertEqual(fitted_temperature(NIMBLE, REVISION), 2.179078721266035)
        self.assertIsNone(fitted_temperature(NIMBLE, "main"))
        self.assertIsNone(fitted_temperature("Qwen/Qwen3.5-9B", REVISION))

    def test_every_adapter_hash_names_a_fitted_revision(self):
        for revision in ADAPTER_REVISIONS.values():
            self.assertIsNotNone(fitted_temperature(NIMBLE, revision))


class V2DefaultTests(unittest.TestCase):
    def test_release_default_survives_documentation_revisions(self):
        for revision in ("main", "84c365a5b0366ba008df817faeffa631002e7d34", "new-card-revision"):
            self.assertEqual(resolve_temperature(V2_MODEL, revision), V2_TEMPERATURE)
            self.assertIsNone(fitted_temperature(V2_MODEL, revision))

    def test_local_merge_uses_adapter_identity(self):
        with tempfile.TemporaryDirectory() as path:
            ready = Path(path) / "READY.json"
            ready.write_text(json.dumps({"adapter_sha256": V2_ADAPTER_SHA256,
                                         "base_revision": "unrelated-base"}))
            self.assertEqual(resolve_temperature("local-model", "unknown", model_path=path), V2_TEMPERATURE)
            with self.assertRaisesRegex(ValueError, "allow_uncalibrated"):
                resolve_temperature("local-model", "unknown", 1.0, model_path=path)
            ready.write_text(json.dumps({"base_revision": V2_ADAPTER_SHA256}))
            self.assertEqual(resolve_temperature("local-model", "unknown", model_path=path), 1.0)

    def test_raw_probabilities_require_deliberate_opt_in(self):
        with self.assertRaisesRegex(ValueError, "allow_uncalibrated"):
            resolve_temperature(V2_MODEL, "main", 1.0)
        self.assertEqual(resolve_temperature(V2_MODEL, "main", 1.0, allow_uncalibrated=True), 1.0)
        self.assertEqual(resolve_temperature(V2_MODEL, "main", 2.5), 2.5)

    def test_other_checkpoints_keep_their_defaults(self):
        self.assertEqual(resolve_temperature(NIMBLE, REVISION), 2.179078721266035)
        self.assertEqual(resolve_temperature("Qwen/Qwen3.5-9B", REVISION), 1.0)
        self.assertEqual(resolve_temperature(NIMBLE, "unknown"), 1.0)

    def test_served_ready_layouts(self):
        self.assertEqual(served_temperature({"model": V2_MODEL, "revision": "main"}), V2_TEMPERATURE)
        self.assertEqual(served_temperature({"adapter_sha256": V2_ADAPTER_SHA256}), V2_TEMPERATURE)
        self.assertEqual(served_temperature({"model": NIMBLE, "revision": REVISION}), 2.179078721266035)
        self.assertEqual(served_temperature({"adapter_sha256": next(iter(ADAPTER_REVISIONS))}), 2.179078721266035)
        self.assertEqual(served_temperature({"base_revision": V2_ADAPTER_SHA256}), 1.0)

    def test_invalid_overrides(self):
        for value in [True, "2.1", 0, -1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_temperature(V2_MODEL, "main", value)


if __name__ == "__main__":
    unittest.main()
