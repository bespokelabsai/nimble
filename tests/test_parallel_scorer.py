"""Small real hybrid-model checks, without downloading or loading 4B weights."""

import unittest
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from nimble.scoring.calibration import fitted_temperature
from nimble.scoring.parallel_schema import REVISION, PreparedPrompts, parse_schema, prepare_prompts, validate_schema
from nimble.scoring.parallel_scorer import ParallelScorer, broadcast_cache, candidate_projection


def toy_scorer():
    mx.random.seed(42)
    model = Model(ModelArgs(model_type="qwen3_5", text_config={
        "model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "vocab_size": 32, "full_attention_interval": 4,
        "linear_num_value_heads": 2, "linear_num_key_heads": 2,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "tie_word_embeddings": True,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    }))
    model.eval()
    mx.eval(model.parameters())
    scorer = ParallelScorer.__new__(ParallelScorer)
    scorer.model = model
    scorer.backbone = model.language_model.model
    scorer.head_weight = scorer.backbone.embed_tokens.weight
    scorer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    scorer.temperature = 1.0
    return scorer


def prompts():
    prefix = [1, 2, 3, 4, 5]
    tails = [[6, 7], [8, 9, 10, 11], [12]]
    return PreparedPrompts(["one", "two", "three"], [["A", "B"], [False, True], ["A", "B", "C"]],
                           prefix, tails, [prefix + t for t in tails], [[15, 16], [15, 16], [15, 16, 17]])


class ParallelTests(unittest.TestCase):
    def setUp(self):
        # CPU FP32 provides a tight reference independent of Metal kernel choices.
        self.previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self.previous_device)

    def test_parallel_matches_full_independent_with_unequal_suffix_lengths(self):
        scorer = toy_scorer()
        p = prompts()
        parallel, metrics = scorer.evaluate(p)
        independent, _ = scorer.evaluate(p, mode="independent")
        cached, _ = scorer.evaluate(p, mode="cached_serial")
        self.assertEqual(metrics["prefix_prefill_count"], 1)
        self.assertEqual(metrics["branch_batches"], 1)
        for actual, full, serial in zip(parallel, independent, cached):
            np.testing.assert_allclose(np.array(actual), np.array(full), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(np.array(actual), np.array(serial), rtol=1e-4, atol=1e-4)

    def test_microbatch_and_one_field(self):
        scorer = toy_scorer()
        p = prompts()
        all_rows, _ = scorer.evaluate(p)
        small_rows, metrics = scorer.evaluate(p, field_batch_size=2)
        self.assertEqual(metrics["branch_batches"], 2)
        for a, b in zip(all_rows, small_rows):
            np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-4)
        one = PreparedPrompts(p.names[:1], p.choices[:1], p.prefix_ids, p.suffix_ids[:1], p.full_ids[:1], p.candidate_ids[:1])
        actual, _ = scorer.evaluate(one)
        np.testing.assert_allclose(np.array(actual[0]), np.array(all_rows[0]), rtol=1e-4, atol=1e-4)

    def test_branch_updates_do_not_mutate_prefill_cache(self):
        scorer = toy_scorer()
        cache = scorer.model.make_cache()
        h = scorer.backbone(mx.array([[1, 2, 3]]), cache=cache)
        mx.eval(h, [c.state for c in cache])
        before = [[np.array(x).copy() for x in c.state] for c in cache]
        branch = broadcast_cache(cache, 2)
        out = scorer.backbone(mx.array([[4, 5], [6, 7]]), cache=branch)
        mx.eval(out, [c.state for c in branch])
        for source, states in zip(cache, before):
            for x, old in zip(source.state, states):
                np.testing.assert_array_equal(np.array(x), old)
        self.assertEqual(cache[-1].offset, 3)
        self.assertEqual(branch[-1].offset, 5)

    def test_candidate_only_head_matches_full_projection(self):
        hidden = mx.random.normal((3, 8))
        weight = mx.random.normal((32, 8))
        ids = [2, 5, 17]
        actual = candidate_projection(hidden, weight, ids)
        expected = (hidden @ weight.T)[:, mx.array(ids)]
        mx.eval(actual, expected)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)

    def test_boolean_validation_and_duplicate_keys(self):
        validate_schema({"review": {"type": "boolean", "description": "Requires review"}})
        for text in ('{"a":{},"a":{}}', '{"a":{"type":"boolean","description":"x","choices":[0,1]}}'):
            with self.assertRaises(ValueError):
                parse_schema(text)

    def test_programmatic_boolean_and_temperature(self):
        scorer = toy_scorer()
        scorer.prepare = lambda *args: prompts()
        result = scorer.score("context", {})
        self.assertIs(type(result["output"]["two"]), bool)
        for field in result["fields"].values():
            self.assertAlmostEqual(sum(field["scores"].values()), 1.0, places=6)
        scorer.temperature = 2.0
        warmer = scorer.score("context", {})
        self.assertEqual(result["output"], warmer["output"])
        for name in result["fields"]:
            self.assertLessEqual(max(warmer["fields"][name]["scores"].values()), max(result["fields"][name]["scores"].values()) + 1e-6)

    def test_fitted_temperature_is_reported(self):
        scorer = toy_scorer()
        scorer.prepare = lambda *args: prompts()
        self.assertFalse(scorer.score("context", {})["temperature_fitted"])
        scorer.model_id, scorer.revision = "bespokelabs/Bespoke-Nimble-9B", "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"
        scorer.temperature = fitted_temperature(scorer.model_id, scorer.revision)
        result = scorer.score("context", {})
        self.assertTrue(result["temperature_fitted"])
        self.assertEqual((result["model"], result["revision"]), (scorer.model_id, scorer.revision))

    def test_metal_path_with_explicit_numerical_tolerance(self):
        mx.set_default_device(mx.gpu)
        scorer = toy_scorer()
        parallel, _ = scorer.evaluate(prompts())
        independent, _ = scorer.evaluate(prompts(), mode="independent")
        serial, _ = scorer.evaluate(prompts(), mode="cached_serial")
        for actual, full, cached in zip(parallel, independent, serial):
            for expected in (full, cached):
                np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0, atol=0.01)
                np.testing.assert_allclose(np.array(mx.softmax(actual)), np.array(mx.softmax(expected)), rtol=0, atol=0.003)
                self.assertEqual(mx.argmax(actual).item(), mx.argmax(expected).item())


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        path = PROJECT_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots" / REVISION
        if not path.is_dir():
            raise unittest.SkipTest("Local Qwen tokenizer has not been downloaded.")
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls.schema = {
            'a "quoted" field': {"type": "enum", "choices": ["LOW", "VERY_HIGH"], "description": "Severity"},
            "révision": {"type": "boolean", "description": "Requires human review"},
        }

    def test_exact_token_reconstruction_and_single_token_codes(self):
        p = prepare_prompts(self.tokenizer, 'Context <|im_end|> café __PARALLEL_FIELD_TARGET__', self.schema, 4096)
        for suffix, full, ids in zip(p.suffix_ids, p.full_ids, p.candidate_ids):
            self.assertEqual(p.prefix_ids + suffix, full)
            self.assertTrue(suffix)
            self.assertEqual(len(ids), len(set(ids)))
        # The actual chat closes with a completed, empty thinking block.
        self.assertTrue(self.tokenizer.decode(p.full_ids[0]).endswith("<think>\n\n</think>\n\n"))

    def test_overlength_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
            prepare_prompts(self.tokenizer, "context", self.schema, 5)


class V2MlxLoadingTests(unittest.TestCase):
    def test_constructor_resolves_default_and_requires_raw_opt_in(self):
        import tempfile
        from unittest.mock import patch
        from nimble.scoring.calibration import V2_MODEL, V2_TEMPERATURE
        language = SimpleNamespace(model=object(), lm_head=SimpleNamespace(weight=mx.array([[1.0]])),
                                   args=SimpleNamespace(max_position_embeddings=8193))
        model = SimpleNamespace(model_type="qwen3_5", language_model=language, eval=lambda:None)
        with tempfile.TemporaryDirectory() as path, \
             patch("nimble.scoring.parallel_scorer.load", return_value=(model, object())), \
             patch.object(mx.metal, "is_available", return_value=True), \
             patch.object(mx, "set_default_device"):
            scorer = ParallelScorer(path, model_id=V2_MODEL, revision="main")
            self.assertEqual(scorer.temperature, V2_TEMPERATURE)
            with self.assertRaisesRegex(ValueError, "allow_uncalibrated"):
                ParallelScorer(path, model_id=V2_MODEL, revision="main", temperature=1.0)
            raw = ParallelScorer(path, model_id=V2_MODEL, revision="main", temperature=1.0,
                                  allow_uncalibrated=True)
            self.assertEqual(raw.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
