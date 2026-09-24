"""Release compatibility: original calibration and the new wide-choice contract."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from nimble.scoring.calibration import (ADAPTER_REVISIONS, LATEST_ADAPTER_SHA256,
    V2_MODEL, V2_TEMPERATURE, resolve_temperature)
from nimble.scoring.release_contract import prompt_builder
from nimble.scoring import parallel_schema


class ReleaseDefaultsTests(unittest.TestCase):
    def test_new_weights_do_not_inherit_old_calibration(self):
        for model, revision in [(V2_MODEL, 'main'),
                               ('bespokelabs/Bespoke-Nimble-9B', next(iter(ADAPTER_REVISIONS.values())))]:
            with self.subTest(model=model):
                self.assertEqual(resolve_temperature(model, revision,
                    adapter_sha256=LATEST_ADAPTER_SHA256), 1.0)

    def test_original_tag_keeps_original_temperature_after_local_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'READY.json').write_text(json.dumps({
                'adapter_sha256': next(iter(ADAPTER_REVISIONS))}))
            self.assertEqual(resolve_temperature('bespokelabs/Bespoke-Nimble-9B',
                'original-2676', model_path=directory), V2_TEMPERATURE)

    def test_unknown_model_keeps_legacy_prompt_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIs(prompt_builder(directory, object()), parallel_schema.prepare_prompts)


@unittest.skipUnless(os.environ.get('NIMBLE_RELEASE_PATH'), 'Set NIMBLE_RELEASE_PATH for tokenizer integration')
class ReleaseTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.directory = Path(os.environ['NIMBLE_RELEASE_PATH'])
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.directory, local_files_only=True)
        cls.prepare = staticmethod(prompt_builder(cls.directory, cls.tokenizer))

    def test_short_prompts_remain_identical(self):
        schema = {'route': {'type': 'enum', 'description': 'Choose a route.',
                            'choices': ['sales', 'support']}}
        original = parallel_schema.prepare_prompts(self.tokenizer, 'Need help.', schema, 8192)
        actual = self.prepare(self.tokenizer, 'Need help.', schema, 8192)
        self.assertEqual(original.full_ids, actual.full_ids)
        self.assertEqual(original.candidate_ids, actual.candidate_ids)

    def test_wide_choices_keep_all_candidates_and_enforce_context(self):
        schema = {'route': {'type': 'enum', 'description': 'Choose a route.',
                            'choices': [f'route_{i}' for i in range(255)]}}
        actual = self.prepare(self.tokenizer, 'Use route_254.', schema, 8192)
        self.assertEqual(len(set(actual.candidate_ids[0])), 255)
        self.assertEqual(actual.choices[0][-1], 'route_254')
        with self.assertRaisesRegex(ValueError, 'Nothing was truncated'):
            self.prepare(self.tokenizer, 'Use route_254.', schema, 10)
        schema['route']['choices'].append('route_255')
        with self.assertRaisesRegex(ValueError, '1–255'):
            self.prepare(self.tokenizer, 'Use route_254.', schema, 8192)

    def test_modified_codebook_is_rejected(self):
        contract = json.loads((self.directory / 'schema_config.json').read_text())
        contract['candidate_token_ids'][-1] = contract['candidate_token_ids'][0]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'schema_config.json').write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, 'encoding'):
                prompt_builder(directory, self.tokenizer)


if __name__ == '__main__':
    unittest.main()
