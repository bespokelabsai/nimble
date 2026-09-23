"""CPU numerical tests of the CUDA runner's backbone and candidate head path."""

import unittest
from types import SimpleNamespace

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from nimble.scoring.cuda_scorer import candidate_projection, physical_weight


class CudaProjectionTests(unittest.TestCase):
    def check_model(self, model):
        torch.manual_seed(17)
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        candidates = [18, 12, 25]
        with torch.inference_mode():
            hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1]
            actual = candidate_projection(hidden, model.get_output_embeddings().weight, candidates)
            expected = model(input_ids=tokens, use_cache=False).logits[:, -1, candidates]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_gemma_sliding_attention(self):
        config = Gemma3TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                 vocab_size=64, sliding_window=8, max_position_embeddings=64,
                                 layer_types=["sliding_attention"] * 5 + ["full_attention"])
        self.check_model(Gemma3ForCausalLM(config))

    def test_qwen_hybrid_untied_head(self):
        config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                 vocab_size=64, max_position_embeddings=64,
                                 linear_num_value_heads=2, linear_num_key_heads=2,
                                 linear_key_head_dim=16, linear_value_head_dim=16,
                                 tie_word_embeddings=False,
                                 layer_types=["linear_attention"] * 3 + ["full_attention"])
        self.check_model(Qwen3_5ForCausalLM(config))

    def test_bf16_projection_accumulates_in_float32(self):
        hidden = torch.tensor([[1.125, .375]], dtype=torch.bfloat16)
        weight = torch.tensor([[.25, .5], [2., 3.], [-.5, 1.]], dtype=torch.bfloat16)
        actual = candidate_projection(hidden, weight, [2, 0])
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, (hidden.float() @ weight.float().T)[:, [2, 0]])

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_gemma_eager_at_sliding_window_boundary(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.manual_seed(17)
        config = Gemma3TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                 vocab_size=64, sliding_window=512, max_position_embeddings=1024,
                                 layer_types=["sliding_attention"])
        model = Gemma3ForCausalLM(config).eval().to('cuda')
        ids = torch.arange(513, device='cuda').remainder(64).unsqueeze(0)
        with torch.inference_mode():
            model.set_attn_implementation('eager')
            eager = model(ids, use_cache=False).logits[:, -1]
            # In one sliding layer, the last position cannot see token zero.
            changed = ids.clone()
            changed[0, 0] = 63
            excluded = model(changed, use_cache=False).logits[:, -1]
            model.set_attn_implementation('sdpa')
            with sdpa_kernel(SDPBackend.MATH):
                reference = model(ids, use_cache=False).logits[:, -1]
        torch.testing.assert_close(eager, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(eager, excluded, atol=1e-5, rtol=1e-5)


class OffloadedHeadTests(unittest.TestCase):
    """accelerate offload leaves a meta parameter whose values live in the hook's weights map."""

    def offload(self, head):
        real = head.weight.detach().clone()
        head.weight = torch.nn.Parameter(torch.empty(head.weight.shape, device="meta"))
        head._hf_hook = SimpleNamespace(weights_map={"weight": real}, execution_device="cpu")

    def test_projection_from_an_offloaded_head_matches_the_resident_head(self):
        torch.manual_seed(17)
        head, hidden = torch.nn.Linear(4, 6, bias=False), torch.randn(1, 4)
        with torch.inference_mode():
            expected = head(hidden)[:, [5, 2]]
            self.offload(head)
            actual = candidate_projection(hidden, physical_weight(head), [5, 2])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_meta_head_without_a_weights_map_is_rejected(self):
        with self.assertRaises(RuntimeError):
            physical_weight(torch.nn.Linear(4, 6, bias=False, device="meta"))


class V2CudaLoadingTests(unittest.TestCase):
    def test_constructor_resolves_default_and_requires_raw_opt_in(self):
        from unittest.mock import patch
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration
        from nimble.scoring.cuda_scorer import CudaCandidateScorer
        from nimble.scoring.calibration import V2_MODEL, V2_TEMPERATURE
        embed = torch.nn.Embedding(8, 4)
        model = SimpleNamespace(model=object(), get_output_embeddings=lambda:embed,
                                get_input_embeddings=lambda:embed)
        model.eval = lambda:model
        model.to = lambda _:model
        config = SimpleNamespace(model_type="qwen3_5", max_position_embeddings=8193)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=0), \
             patch.object(AutoConfig, "from_pretrained", return_value=config), \
             patch.object(AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=(model, {})):
            scorer = CudaCandidateScorer("unused", V2_MODEL, "main")
            self.assertEqual(scorer.temperature, V2_TEMPERATURE)
            with self.assertRaisesRegex(ValueError, "allow_uncalibrated"):
                CudaCandidateScorer("unused", V2_MODEL, "main", temperature=1.0)
            raw = CudaCandidateScorer("unused", V2_MODEL, "main", temperature=1.0,
                                      allow_uncalibrated=True)
            self.assertEqual(raw.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
