"""Every public-benchmark source module: real invariants on tiny fixtures, no network."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from nimble.datasets.public_records import FORBIDDEN_INPUT_KEYS, reference_keys
from nimble.datasets.public_sources import (aegis2, boolq, civil_comments, helpsteer2, massive, multinli, paws,
                                            pubmedqa, squad2, summeval, vitaminc)
from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import validate_schema

VOTES = ["entailment"] * 4 + ["neutral"]
ARTICLE = {"id": "cnn-1", "text": "The council voted on Monday to close the bridge for repairs until spring.",
           "machine_summaries": ["The bridge closes for repairs.", "The council met on Monday."],
           "human_summaries": ["Bridge closed until spring."],
           "relevance": [4.67, 2.33], "coherence": [4.0, 3.0], "fluency": [5.0, 5.0], "consistency": [5.0, 2.5]}
SQUAD_PARAGRAPH = {"context": "The bridge opened in 1932 and spans the river.", "qas": [
    {"id": "q1", "question": "When did the bridge open?", "answers": [{"text": "1932", "answer_start": 20}],
     "is_impossible": False},
    {"id": "q2", "question": "How long is the bridge?", "answers": [], "is_impossible": True}]}

# (module, subset, raw row, expected target, expected family) for the happy path of every source.
CASES = [
    (vitaminc, "", {"unique_id": "abc_1", "case_id": "abc", "wiki_revision_id": "7", "label": "REFUTES",
                    "claim": "It had 1000 guests.", "evidence": "It had 6000 guests.", "page": "Con",
                    "revision_type": "real"}, "REFUTES", "abc"),
    (massive, "en-US", {"id": 7, "locale": "en-US", "partition": "test", "scenario": "alarm",
                        "intent": "alarm_set", "utt": "wake me at five", "annot_utt": "", "worker_id": "1"},
     "alarm", "massive-7"),
    (boolq, "", {"question": "is the sky blue", "passage": "The sky appears blue.", "answer": True}, True, None),
    (squad2, "", {"id": "q1", "title": "Bridge", "context": SQUAD_PARAGRAPH["context"],
                  "question": "When?", "answers": {"text": ["1932"], "answer_start": [20]}}, True, None),
    (paws, "", {"id": 3, "sentence1": "Ann met Bob.", "sentence2": "Bob met Ann.", "label": 1}, True, "3"),
    (multinli, "", {"pairID": "p1", "promptID": "pr1", "genre": "fiction", "sentence1": "He slept.",
                    "sentence2": "He rested.", "gold_label": "entailment", "annotator_labels": VOTES},
     "entailment", "pr1"),
    (civil_comments, "", {"row_index": 4, "text": "You are a fool.", "toxicity": 0.8, "insult": 0.7}, True,
     "civil_comments-4"),
    (aegis2, "", {"id": "a1", "prompt": "How do I pick a lock?", "response": None, "prompt_label": "unsafe",
                  "response_label": None, "violated_categories": "criminal planning, other",
                  "prompt_label_source": "human", "response_label_source": None,
                  "reconstruction_id_if_redacted": None}, True, "aegis2-a1"),
    (helpsteer2, "", {"row_index": 0, "prompt": "Explain tides.", "response": "The moon pulls the sea.",
                      "helpfulness": 3, "correctness": 4, "coherence": 4, "complexity": 1, "verbosity": 1},
     3, None),
    (summeval, "relevance", None, 4, "cnn-1"),  # raw comes from summeval.rows() over ARTICLE
    (pubmedqa, "", {"pubid": "9", "question": "Does X help?", "context": {"contexts": ["X helped."]},
                    "long_answer": "Yes, X helps.", "final_decision": "Yes"}, "yes", "9"),
]


def summeval_raws():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jsonl"
        path.write_text(json.dumps(ARTICLE) + "\n")
        return summeval.rows(path)


class SourceContractTests(unittest.TestCase):
    def test_every_source_builds_a_valid_leak_free_record(self):
        for module, subset, raw, target, family in CASES:
            if module is summeval:
                raw = summeval_raws()[0]
            with self.subTest(module.NAME):
                record = module.record(raw, subset)
                self.assertEqual(record["reference"]["target"], target)
                if family is not None:
                    self.assertEqual(record["family"], family)
                self.assertTrue(record["id"].startswith(module.NAME))
                self.assertEqual(reference_keys(record["input"]) & FORBIDDEN_INPUT_KEYS, set())
                validate_schema(adapt_input(record["input"])[1])

    def test_answer_text_never_enters_the_input(self):
        squad = squad2.record(CASES[3][2])  # the answer span lives in the paragraph; the answer list must not
        self.assertNotIn("answers", reference_keys(squad["input"]))
        self.assertEqual(squad["reference"]["answer_texts"], ["1932"])
        pubmed = pubmedqa.record(CASES[10][2])
        self.assertNotIn("Yes, X helps.", json.dumps(pubmed["input"]))

    def test_rows_that_must_be_skipped_return_none(self):
        skipped = [
            (massive, "en-US", {**CASES[1][2], "partition": "train"}),
            (massive, "de-DE", CASES[1][2]),
            (multinli, "", {**CASES[5][2], "gold_label": "-"}),
            (aegis2, "", {**CASES[7][2], "prompt_label_source": "llm_jury"}),
            (aegis2, "", {**CASES[7][2], "reconstruction_id_if_redacted": "r1", "prompt": "REDACTED"}),
            (boolq, "", {**CASES[2][2], "answer": "yes"}),
            (paws, "", {**CASES[4][2], "label": 2}),
            (pubmedqa, "", {**CASES[10][2], "final_decision": None}),
            (squad2, "", {**CASES[3][2], "answers": {"text": [], "answer_start": []}, "is_impossible": False}),
        ]
        for module, subset, raw in skipped:
            with self.subTest(module.NAME):
                self.assertIsNone(module.record(raw, subset))

    def test_unexpected_labels_raise_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            massive.record({**CASES[1][2], "scenario": "kitchen"}, "en-US")
        with self.assertRaises(ValueError):
            vitaminc.record({**CASES[0][2], "label": "MAYBE"})
        with self.assertRaises(ValueError):  # gold must be a most-voted option
            multinli.record({**CASES[5][2], "annotator_labels": ["neutral"] * 3 + ["entailment"] * 2})


class FamilyAndDistributionTests(unittest.TestCase):
    def test_massive_ids_and_families_are_identical_across_locales(self):
        english = massive.record(CASES[1][2], "en-US")
        german = massive.record({**CASES[1][2], "locale": "de-DE", "utt": "weck mich um fünf"}, "de-DE")
        self.assertEqual(english["id"], german["id"])
        self.assertEqual(english["family"], german["family"])
        self.assertNotEqual(english["domain"], german["domain"])

    def test_multinli_distribution_is_the_vote_share(self):
        record = multinli.record(CASES[5][2])
        reference = record["reference"]
        self.assertEqual(reference["distribution"], {"entailment": 0.8, "neutral": 0.2, "contradiction": 0.0})
        self.assertEqual(set(reference["distribution"]), set(multinli.CRITERIA))
        self.assertEqual((reference["annotations"], reference["unanimous"]), (5, False))

    def test_civil_comments_threshold_and_rater_fraction(self):
        toxic = civil_comments.record(CASES[6][2])
        self.assertAlmostEqual(toxic["reference"]["distribution"]["true"], 0.8)
        self.assertAlmostEqual(sum(toxic["reference"]["distribution"].values()), 1.0)
        benign = civil_comments.record({**CASES[6][2], "toxicity": 0.49})
        self.assertIs(benign["reference"]["target"], False)

    def test_helpsteer2_responses_to_one_prompt_share_a_family(self):
        first = helpsteer2.record(CASES[8][2])
        second = helpsteer2.record({**CASES[8][2], "row_index": 1, "response": "Gravity.", "helpfulness": 0})
        self.assertEqual(first["family"], second["family"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(first["input"]["questions"]["decision"]["criteria"]), 5)
        self.assertEqual(second["reference"]["target"], 0)

    def test_summeval_unpacks_summaries_and_rounds_half_up(self):
        raws = summeval_raws()
        self.assertEqual([r["summary_index"] for r in raws], [0, 1])
        self.assertEqual([summeval.level_index(v) for v in (1.0, 2.33, 2.5, 4.67)], [0, 1, 2, 4])
        relevance = [summeval.record(r, "relevance") for r in raws]
        consistency = summeval.record(raws[1], "consistency")
        self.assertEqual([r["reference"]["target"] for r in relevance], [4, 1])
        self.assertEqual(consistency["reference"]["target"], 2)
        self.assertEqual({r["family"] for r in relevance}, {"cnn-1"})
        self.assertEqual(len({r["id"] for r in relevance}), 2)
        self.assertNotIn("relevance", relevance[0]["reference"]["other_dimensions"])


class ReaderTests(unittest.TestCase):
    def test_squad_nested_layout_is_flattened(self):
        document = {"data": [{"title": "Bridge", "paragraphs": [SQUAD_PARAGRAPH]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev-v2.0.json"
            path.write_text(json.dumps(document))
            raws = squad2.rows(path)
        records = [squad2.record(raw) for raw in raws]
        self.assertEqual([r["reference"]["target"] for r in records], [True, False])
        self.assertEqual(len({r["family"] for r in records}), 1)

    def test_massive_rows_come_straight_from_the_tarball(self):
        rows = [CASES[1][2], {**CASES[1][2], "id": 8, "partition": "dev"}]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
            info = tarfile.TarInfo("1.1/data/en-US.jsonl")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "amazon-massive-dataset-1.1.tar.gz"
            path.write_bytes(buffer.getvalue())
            raws = list(massive.rows(path, "en-US"))
        self.assertEqual([r["id"] for r in raws], [7, 8])
        self.assertEqual([massive.record(r, "en-US") is None for r in raws], [False, True])


if __name__ == "__main__":
    unittest.main()
