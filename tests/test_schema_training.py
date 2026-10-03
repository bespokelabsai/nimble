"""Candidate masking, semantic labels, and held-out separation."""

import unittest
from types import SimpleNamespace

from nimble.training.schema_data import as_scoring, validate_separation, resolve_checkpoint, prepare_data, read_rows


class SchemaDataTests(unittest.TestCase):
    def test_read_rows_preserves_unicode_separators(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            for separator in ("\u0085", "\u2028", "\u2029"):
                with self.subTest(separator=repr(separator)):
                    rows = [
                        {"id": "one", "input": {"state": f"First{separator}Second"}},
                        {"id": "two", "input": {"state": "Another record"}},
                    ]
                    contents = "\r\n\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                    path.write_bytes((contents + "\r\n").encode("utf-8"))
                    self.assertEqual(read_rows(path), rows)

    def test_explicit_model_revision_and_export_identity(self):
        import json
        import tempfile
        from pathlib import Path
        from nimble.scoring.parallel_schema import MODEL_ID, REVISION
        self.assertEqual(resolve_checkpoint(MODEL_ID, None), (MODEL_ID, REVISION))
        with self.assertRaisesRegex(ValueError, 'explicit --revision'):
            resolve_checkpoint('Qwen/Qwen3.5-9B', None)
        model, revision = resolve_checkpoint('Qwen/Qwen3.5-9B', 'pinned-9b')
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ('train', 'train_scoring', 'train_tokens'):
                (p / (name + '.jsonl')).write_text('{"id":"one"}\n')
            (p / 'manifest.json').write_text('{}')
            (p / 'train_tokens.manifest.json').write_text(json.dumps({'model': MODEL_ID, 'revision': REVISION}))
            with self.assertRaisesRegex(ValueError, 'model/revision differs'):
                prepare_data(p, p / 'eval.jsonl', None, 2048, model, revision)

    def test_gold_labels_do_not_enter_context_or_schema(self):
        row = {"id": "r", "family": "f", "source_family": "f", "input": {
            "state": "A fact", "questions": {"decision": {
                "type": "score", "instructions": "Degree", "criteria": ["low", "high"]}}},
            "reference": {"target": 1, "reason": "HIDDEN_REASON"}}
        result = as_scoring(row, True)
        self.assertEqual(result["target"], "1")
        self.assertEqual(result["schema"]["decision"]["choices"], ["0", "1"])
        self.assertNotIn("HIDDEN_REASON", str(result))
        self.assertEqual(result["context"], "A fact")

    def test_reject_source_family_leakage_and_wrong_splits(self):
        training = [{"id": "t", "split": "train", "source_family": "family", "provenance": {"source_id": "s"}, "input": {"state": "one"}}]
        validation = [{"id": "v", "split": "validation", "family": "other", "input": {"state": "two"}}]
        validate_separation(training, validation)
        for update in ({"family": "family"}, {"id": "s"}, {"split": "eval"}, {"input": {"state": "one"}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_separation(training, [{**validation[0], **update}])

    def test_curated_holdout_uses_original_source_not_pair_id(self):
        training = [{"id": "t", "split": "train", "source_family": "shared",
                     "provenance": {"source_id": "seed"}, "input": {"state": "one"}}]
        heldout = {"id": "v", "family": "different-pair", "source_family": "other",
                   "split": "validation", "provenance": {"source_id": "other-seed"},
                   "input": {"state": "two"}}
        validate_separation(training, [heldout])
        with self.assertRaisesRegex(ValueError, "Source family"):
            validate_separation(training, [{**heldout, "source_family": "shared"}])
        with self.assertRaisesRegex(ValueError, "validation source"):
            validate_separation(training, [{**heldout, "provenance": {"source_id": "seed"}}])


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from nimble.training.schema_train import CandidateCollator, CandidateTrainer, decision_result
        cls.torch = torch
        cls.collator = CandidateCollator
        cls.trainer = CandidateTrainer
        cls.decision = staticmethod(decision_result)

    def test_mixed_candidate_counts_mask_padding_and_gradients(self):
        torch = self.torch
        rows = [{"input_ids": [1, 2], "candidate_ids": [2, 4], "labels": 1},
                {"input_ids": [1], "candidate_ids": [1, 3, 5], "labels": 2}]
        batch = self.collator(0)(rows)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [0, 1]])
        logits = torch.tensor([[[1000., 0., 1., 0., 2., 0.]], [[1000., 1., 0., 2., 0., 3.]]], requires_grad=True)
        def model(**kwargs):
            self.assertEqual(kwargs["logits_to_keep"], 1)
            self.assertNotIn("labels", kwargs)
            return SimpleNamespace(logits=logits)
        loss, out = self.trainer.compute_loss(None, model, batch, return_outputs=True)
        expected = (torch.nn.functional.cross_entropy(torch.tensor([[1., 2.]]), torch.tensor([1])) +
                    torch.nn.functional.cross_entropy(torch.tensor([[1., 2., 3.]]), torch.tensor([2]))) / 2
        self.assertAlmostEqual(loss.item(), expected.item())
        self.assertEqual(out.logits.softmax(-1)[0, 2:].sum().item(), 0.)
        loss.backward()
        self.assertEqual(logits.grad[:, :, 0].abs().sum().item(), 0.)
        self.assertGreater(logits.grad.abs().sum().item(), 0.)

    def test_reject_gold_index_in_padded_candidates(self):
        with self.assertRaises(ValueError):
            self.collator(0)([{"input_ids": [1], "candidate_ids": [2, 3], "labels": 2}])

    def test_shuffled_boolean_and_ordered_score_values(self):
        torch = self.torch
        result = self.decision({"choices": [True, False], "kind": "noul", "labels": 1}, torch.tensor([0., 2.]))
        self.assertIs(result["prediction"], False)
        self.assertAlmostEqual(result["probability_true"], 1 / (1 + __import__('math').exp(2)))
        result = self.decision({"choices": ["2", "0", "1"], "kind": "score", "labels": 2}, torch.zeros(3))
        self.assertEqual(result["expected_score"], 1.)
        self.assertEqual(result["prediction"], 2)
        self.assertFalse(result["correct"])

    def test_extreme_log_loss_is_not_clipped(self):
        result = self.decision({"choices": ["a", "b"], "labels": 1}, self.torch.tensor([100., 0.]))
        self.assertAlmostEqual(result["nll"], 100.)


if __name__ == "__main__":
    unittest.main()
