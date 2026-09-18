"""Teacher-free runner, Jev client, run comparison, and suite summary on synthetic fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

from nimble.evaluation import compare_public, evaluate_public_jev, summarize_public_suite
from nimble.evaluation.evaluate_banking77 import summarize as banking77_summarize
from nimble.evaluation.evaluate_public import (distribution_metrics, entropy, evaluate_records,
                                               expected_calibration_error, jensen_shannon, model_input, run,
                                               summarize_rows, total_variation)
from nimble.evaluation.evaluate_public_jev import ApiError, request_with_retry, score_record
from nimble.scoring.parallel_schema import choice_key

TABLE = {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1, "true": 0.9, "false": 0.1,
         "0": 0.1, "1": 0.3, "2": 0.6}
NLI = {"entailment": "It follows.", "neutral": "It may hold.", "contradiction": "It cannot hold."}


def record(row_id, target, kind="choice", **reference):
    if kind == "choice":
        question = {"type": "choice", "instructions": "Relation?", "criteria": NLI}
    elif kind == "noul":
        question = {"type": "noul", "instructions": "Supported?", "criteria": {"false": "No.", "true": "Yes."}}
    else:
        question = {"type": "score", "instructions": "Helpful?", "criteria": ["Not.", "Some.", "Very."]}
    return {"id": row_id, "split": "validation", "domain": kind, "family": f"fam-{row_id}",
            "source_family": f"fam-{row_id}",
            "input": {"state": {"premise": "A man naps.", "hypothesis": "A person rests."},
                      "questions": {"decision": question}},
            "reference": {"target": target, "human_reviewed": True, **reference}}


class FakeScorer:
    """Answers by option key, never by position, and records what it was shown."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def score(self, context, schema, mode="independent"):
        if self.fail_at == len(self.calls) + 1:
            raise RuntimeError("interrupted")
        self.calls.append({"context": context, "schema": json.loads(json.dumps(schema))})
        choices = schema["decision"]["choices"]
        keys = [choice_key(value) for value in choices]
        total = sum(TABLE[key] for key in keys)
        scores = {key: TABLE[key] / total for key in keys}
        best = max(scores, key=scores.get)
        return {"fields": {"decision": {"scores": scores, "value": choices[keys.index(best)],
                                        "code_to_choice": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices))}},
                "metrics": {}}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class MetricTests(unittest.TestCase):
    def test_divergences_and_entropy_hand_values(self):
        p, q, half = {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, {"a": 0.5, "b": 0.5}
        self.assertEqual((jensen_shannon(p, dict(p)), total_variation(p, dict(p))), (0.0, 0.0))
        self.assertAlmostEqual(jensen_shannon(p, q), 1.0, places=12)
        self.assertAlmostEqual(total_variation(p, q), 1.0, places=12)
        self.assertAlmostEqual(jensen_shannon(p, half), 0.311278124459133, places=12)
        self.assertAlmostEqual(total_variation(p, half), 0.5, places=12)
        self.assertAlmostEqual(entropy(half), 1.0, places=12)
        with self.assertRaises(ValueError):
            distribution_metrics({"a": 1.0}, {"b": 1.0})

    def test_ece_matches_the_banking77_binning(self):
        cases = [(0.75, True), (0.25, False), (0.99, True), (0.4, False), (0.42, True), (1.0, True)]
        reference = banking77_summarize([{"correct": c, "confidence": p, "latency_s": 1.0} for p, c in cases])
        self.assertAlmostEqual(expected_calibration_error(cases), reference["ece10"], places=12)
        self.assertAlmostEqual(expected_calibration_error([(0.75, True), (0.25, False)]), 0.25, places=12)


class RunnerTests(unittest.TestCase):
    def test_records_evaluate_end_to_end_without_a_teacher(self):
        human = {"entailment": 0.5, "neutral": 0.5, "contradiction": 0.0}
        records = [record("c1", "entailment"), record("c2", "neutral"), record("n1", True, "noul"),
                   record("n2", False, "noul"), record("s1", 2, "score"), record("s2", 0, "score"),
                   record("d1", "entailment", distribution=human)]
        rows = evaluate_records(records, FakeScorer())
        self.assertEqual([r["student"]["correct"] for r in rows], [True, False, True, False, True, False, True])
        self.assertAlmostEqual(rows[4]["student"]["expected_score"], 1.5, places=12)
        self.assertAlmostEqual(rows[6]["distribution"]["human_entropy_bits"], 1.0, places=12)
        report = summarize_rows(rows)
        self.assertEqual((report["summary"]["all"]["correct"], report["distribution_agreement"]["count"]), (4, 1))
        self.assertAlmostEqual(report["summary"]["score"]["mean_absolute_score_error"], 1.0, places=12)

    def test_reference_content_never_reaches_the_scorer_and_noul_keys_are_enforced(self):
        scorer = FakeScorer()
        evaluate_records([record("c1", "entailment", rationale="SENTINEL")], scorer)
        self.assertNotIn("SENTINEL", json.dumps(scorer.calls))
        self.assertNotIn("reference", json.dumps(scorer.calls))
        bad = record("n1", True, "noul")
        bad["input"]["questions"]["decision"]["criteria"] = {"no": "No.", "yes": "Yes."}
        with self.assertRaises(ValueError):
            model_input(bad)

    def test_shuffle_keeps_probabilities_and_moves_letter_codes(self):
        records = [record(f"c{i}", "entailment") for i in range(12)]
        plain = evaluate_records(records, FakeScorer())
        shuffled = evaluate_records(records, FakeScorer(), shuffle_seed=17)
        self.assertEqual([r["student"]["probabilities"] for r in plain],
                         [r["student"]["probabilities"] for r in shuffled])
        self.assertNotEqual(summarize_rows(plain)["selected_code_counts"],
                            summarize_rows(shuffled)["selected_code_counts"])

    def test_run_writes_outputs_guards_overwrites_and_resumes_after_interruption(self):
        records = [record(f"c{i}", "entailment") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            tainted = write_jsonl(Path(tmp) / "t.jsonl", [dict(records[0], teacher={})])
            with self.assertRaises(ValueError):
                run(tainted, Path(tmp) / "t", model_id="m", revision="r", scorer_factory=FakeScorer)
            data, out = write_jsonl(Path(tmp) / "data.jsonl", records), Path(tmp) / "out"
            with self.assertRaises(RuntimeError):
                run(data, out, model_id="m", revision="r", model_path="models/x",
                    scorer_factory=lambda: FakeScorer(fail_at=3))
            self.assertEqual(len((out / "rows.jsonl").read_text().splitlines()), 2)
            self.assertFalse((out / "summary.json").exists())
            with self.assertRaises(ValueError):  # an existing directory needs resume=True
                run(data, out, model_id="m", revision="r", scorer_factory=FakeScorer)
            with self.assertRaises(ValueError) as caught:  # changed settings are refused
                run(data, out, model_id="m", revision="r", model_path="models/x", resume=True, shuffle_seed=1,
                    scorer_factory=FakeScorer)
            self.assertEqual(str(caught.exception), "Cannot resume: shuffle_seed differ from the saved run")
            scorer = FakeScorer()  # the same checkpoint given as an absolute path resumes
            report = run(data, out, model_id="m", revision="r", model_path=Path("models/x").resolve(),
                         resume=True, scorer_factory=lambda: scorer)
            self.assertEqual((len(scorer.calls), report["count"], report["resumed_rows"]), (3, 5, 2))
            with self.assertRaises(ValueError):  # a completed run cannot be resumed
                run(data, out, model_id="m", revision="r", resume=True, scorer_factory=FakeScorer)


def response(choice="entailment"):
    return {"model": "jev-1.13.0", "answers": {"decision": {
        "type": "choice", "choice": choice, "confidence": 0.7,
        "probabilities": {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}}},
        "usage": {"input_tokens": 50, "output_tokens": 3}}


class FakeTransport:
    """Scripted responses per call: a dict to return or an exception to raise."""

    def __init__(self, script):
        self.script, self.payloads = list(script), []

    def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class JevTests(unittest.TestCase):
    def test_request_carries_only_the_model_input(self):
        row = dict(record("a", "entailment"), teacher={"answers": {}})
        payload = evaluate_public_jev.request_payload(row, "jev-1.13.0")
        self.assertEqual(set(payload), {"model", "state", "questions"})
        self.assertNotIn("reference", json.dumps(payload))
        row["input"]["state"]["reference"] = {"target": "entailment"}
        with self.assertRaises(ValueError):
            evaluate_public_jev.request_payload(row, "jev-1.13.0")

    def test_retry_credential_and_validation_paths(self):
        delays = []
        transport = FakeTransport([ApiError(429), ApiError(503), response()])
        result = request_with_retry(transport, {"model": "m"}, attempts=5, sleep=delays.append)
        self.assertEqual((result["answers"]["decision"]["choice"], delays), ("entailment", [1.0, 2.0]))
        with self.assertRaises(RuntimeError):  # rejected credentials abort the run
            score_record(record("a", "entailment"), "jev-1.13.0", FakeTransport([ApiError(401)]), sleep=lambda s: None)
        with tempfile.TemporaryDirectory() as tmp:  # a malformed reference aborts before any request
            bad = write_jsonl(Path(tmp) / "bad.jsonl", [record("s", "2", kind="score")])
            untouched = FakeTransport([])
            with self.assertRaises(ValueError):
                evaluate_public_jev.run(bad, Path(tmp) / "run", transport=untouched)
            self.assertEqual(untouched.payloads, [])
        invalid = score_record(record("a", "entailment"), "jev-1.13.0",
                               FakeTransport([response(choice="neutral")]), sleep=lambda s: None)
        self.assertIn("Invalid response", invalid["error"])
        self.assertFalse(invalid["student"]["correct"])

    def test_run_counts_errors_incorrect_and_resumes_by_id(self):
        records = [record("a", "entailment"), record("b", "neutral"), record("c", "entailment")]
        with tempfile.TemporaryDirectory() as tmp:
            data, out = write_jsonl(Path(tmp) / "all.jsonl", records), Path(tmp) / "run"
            transport = FakeTransport([response(), ApiError(400), response()])
            report = evaluate_public_jev.run(data, out, transport=transport, concurrency=1, sleep=lambda s: None)
            self.assertEqual((report["count"], report["valid"], report["errors"],
                              report["summary"]["all"]["correct"]), (3, 2, 1, 2))
            self.assertTrue(all("reference" not in json.dumps(p) for p in transport.payloads))
            again = FakeTransport([response()])  # only the failed record is asked again, and its row replaced
            report = evaluate_public_jev.run(data, out, transport=again, sleep=lambda s: None)
            self.assertEqual((len(again.payloads), report["resumed"], report["retried_errors"], report["errors"],
                              report["summary"]["all"]["correct"]), (1, True, 1, 0, 2))
            with self.assertRaises(ValueError):  # different settings in the same directory
                evaluate_public_jev.run(data, out, transport=again, limit=1)


def scored(row_id, correct, family, error=None):
    row = {"id": row_id, "type": "choice", "domain": "d", "family": family, "split": "validation",
           "reference": {"target": "x"}, "elapsed_seconds": 0.1,
           "student": {"prediction": "x" if correct else "y", "correct": correct, "top_probability": 0.8,
                       "negative_log_likelihood": 0.5, "multiclass_brier": 0.2, "probabilities": {"x": 0.8, "y": 0.2}}}
    if error:
        row.update(error=error, student={"prediction": None, "correct": False})
    return row


class CompareTests(unittest.TestCase):
    def test_statistics_hand_values(self):
        low, high = compare_public.wilson_interval(2, 4)
        self.assertAlmostEqual(low, 0.15004, places=4)
        self.assertAlmostEqual(high, 0.84996, places=4)
        self.assertAlmostEqual(compare_public.mcnemar_exact(3, 0), 0.25)
        self.assertEqual(compare_public.mcnemar_exact(0, 0), 1.0)

    def test_compare_joins_by_id_and_reports_families_and_discordants(self):
        nimble = {r["id"]: r for r in [scored("1", True, "f1"), scored("2", True, "f1"),
                                        scored("3", False, "f2"), scored("4", True, "f2")]}
        jev = {r["id"]: r for r in [scored("1", True, "f1"), scored("2", False, "f1"),
                                     scored("3", False, "f2"), scored("4", False, "f2", error="HTTP 503")]}
        result = compare_public.compare({"nimble": nimble, "jev": jev})
        self.assertEqual((result["runs"]["nimble"]["accuracy"], result["runs"]["jev"]["errors"]), (0.75, 1))
        self.assertEqual((result["runs"]["nimble"]["family_complete"]["complete"],
                          result["runs"]["jev"]["family_complete"]["complete"]), (1, 0))
        pair = result["pairs"]["nimble vs jev"]
        self.assertEqual((pair["both_correct"], pair["left_only_correct"], pair["right_only_correct"]), (1, 2, 0))
        self.assertAlmostEqual(pair["mcnemar_exact_p"], 0.5)
        del jev["4"]
        with self.assertRaises(ValueError):
            compare_public.join_runs({"nimble": nimble, "jev": jev})


class SuiteTests(unittest.TestCase):
    def test_suite_aggregates_complete_subsets_and_pairs_locales(self):
        runs = ("nimble-9b", "jev-1.13.0")
        with tempfile.TemporaryDirectory() as tmp:
            root, data = Path(tmp) / "evaluations", Path(tmp) / "data"

            def write_run(subset, run, rows, complete=True):
                write_jsonl(root / subset / run / "rows.jsonl", rows)
                if complete:
                    (root / subset / run / "summary.json").write_text("{}")

            write_run("choice-a", runs[0], [scored("1", True, "f1"), scored("2", True, "f1"),
                                            scored("3", False, "f2"), scored("4", True, "f3")])
            write_run("choice-a", runs[1], [scored("1", True, "f1"), scored("2", False, "f1"),
                                            scored("3", False, "f2"), scored("4", False, "f3")])
            write_jsonl(data / "choice-a" / "manifest.json", [{"dataset": "a", "license": "MIT"}])
            for locale, flags in (("massive-en-US", [True] * 4), ("massive-de-DE", [True, False, True, False])):
                write_run(locale, runs[0], [scored(f"m{i}", ok, f"m{i}") for i, ok in enumerate(flags)])
                write_run(locale, runs[1], [scored(f"m{i}", True, f"m{i}") for i in range(4)])
            write_run("pending", runs[0], [scored("p", True, "p")])
            write_run("pending", runs[1], [scored("p", True, "p")], complete=False)
            result = summarize_public_suite.summarize(root, runs, data)
            write_run("massive-de-DE", runs[0], [scored("other", True, "other")])  # locales no longer join
            self.assertIn("different record IDs", summarize_public_suite.summarize(root, runs, data)["skipped"]["multilingual"])
        choice = {s["subset"]: s for s in result["subsets"]}["choice-a"]
        self.assertEqual((choice["license"], choice["n"], choice["families"]), ("MIT", 4, 3))
        self.assertEqual((choice["runs"][runs[0]]["accuracy"], choice["runs"][runs[1]]["accuracy"]), (0.75, 0.25))
        self.assertEqual((choice["runs"][runs[0]]["family_complete"], choice["runs"][runs[1]]["family_complete"]), (2, 0))
        averages = result["averages"]["all"]["runs"]
        self.assertAlmostEqual(averages[runs[0]]["macro_accuracy"], (0.75 + 1.0 + 0.5) / 3)
        self.assertAlmostEqual(averages[runs[1]]["micro_accuracy"], 9 / 12)
        paired = result["multilingual"][runs[0]]["paired"]
        self.assertEqual((paired["both_correct"], paired["left_only_correct"], paired["right_only_correct"]), (2, 2, 0))
        self.assertAlmostEqual(paired["mcnemar_exact_p"], 0.5)
        self.assertIn("still running", result["skipped"]["pending"])
        self.assertIn("| choice-a | choice | 4 | 3 |", summarize_public_suite.report_markdown(result))


if __name__ == "__main__":
    unittest.main()
