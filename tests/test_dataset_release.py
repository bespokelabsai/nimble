import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.training.evidence_data import validate_evidence_data
from nimble.training.schema_data import fingerprint, read_rows
from nimble.training.verify_dataset import verify


class CanonicalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = PROJECT_ROOT / 'data'
        cls.rows = read_rows(cls.directory / 'train.jsonl')
        cls.manifest = json.loads((cls.directory / 'manifest.json').read_text())

    def test_release_is_self_contained_and_holdout_is_unchanged(self):
        result = verify(self.directory)
        self.assertEqual((result['training_rows'], result['holdout_rows']), (2676, 324))
        self.assertEqual(result['sha256']['train.jsonl'],
                         'beadbb9b81837f7c339e090cd210ce91f65a55f2a3ada1ce9b623765b8d6fe2e')
        self.assertEqual(result['sha256']['eval.jsonl'],
                         '8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896')
        self.assertEqual({p.name for p in self.directory.iterdir()},
                         {'train.jsonl', 'eval.jsonl', 'manifest.json'})

    def test_changed_holdout_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train.jsonl', 'eval.jsonl', 'manifest.json'):
                shutil.copyfile(self.directory / name, root / name)
            with (root / 'eval.jsonl').open('a') as file:
                file.write('\n')
            with self.assertRaisesRegex(ValueError, 'Frozen dataset bytes changed: eval.jsonl'):
                verify(root)

    def test_certificate_reconstruction_rejects_tampering_even_with_new_training_hash(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['reference']['reason'] = 'Changed after verification'
        manifest = {**self.manifest, 'train_sha256': fingerprint(rows)}
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            validate_evidence_data(rows, manifest)


class TextEncodingTests(unittest.TestCase):
    """The release stores UTF-8 text, so every reader must decode it as UTF-8.

    The manifest embeds non-ASCII case text, so a reader that omits the encoding
    depends on the host locale: it succeeds where that locale is UTF-8 and raises
    UnicodeDecodeError where it is not, as on Windows/cp1252 or LANG=C. The
    documented verification command must work on either.
    """

    @staticmethod
    def _locale_default_reads(directory):
        """Read the release with Path.read_text patched to reject implicit encodings."""
        original = Path.read_text
        offending = []

        def strict(self, *args, **kwargs):
            if not args and kwargs.get("encoding") is None:
                offending.append(self.name)
            return original(self, *args, **kwargs)

        from unittest import mock
        with mock.patch.object(Path, "read_text", strict):
            result = verify(directory)
        return result, offending

    def test_verification_decodes_every_file_as_utf8(self):
        result, offending = self._locale_default_reads(PROJECT_ROOT / 'data')
        self.assertEqual(offending, [])
        self.assertEqual((result['training_rows'], result['holdout_rows']), (2676, 324))

    def test_manifest_text_is_not_ascii(self):
        raw = (PROJECT_ROOT / 'data' / 'manifest.json').read_bytes()
        self.assertTrue(any(byte > 0x7F for byte in raw),
                        'fixture assumption: the manifest carries non-ASCII text')
        with self.assertRaises(UnicodeDecodeError):
            raw.decode('ascii')


if __name__ == '__main__':
    unittest.main()
