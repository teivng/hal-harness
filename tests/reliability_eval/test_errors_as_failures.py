"""Tests for HAL_ERRORS_AS_FAILURES handling of "ERROR: ..." entries in raw_eval_results."""

from reliability_eval.loaders.results import (
    ERRORS_AS_FAILURES_ENV,
    count_error_entries,
    extract_minimal_eval_data,
)
from reliability_eval.metrics.predictability import compute_predictability_metrics
from reliability_eval.metrics.robustness import compute_accuracy

RAW = {
    "ok": {"reward": 1.0, "cost": 0.01, "taken_actions": [], "confidence": 0.9},
    "died": "ERROR: boom",
    "prompt": [{"score": 1.0}, "ERROR: variation died"],
}


class TestDefaultDropsErrors:
    def test_string_entries_are_dropped(self, monkeypatch):
        monkeypatch.delenv(ERRORS_AS_FAILURES_ENV, raising=False)
        result = extract_minimal_eval_data(RAW)
        assert set(result) == {"ok", "prompt"}
        assert result["prompt"] == [{"score": 1.0}]

    def test_accuracy_excludes_errors(self, monkeypatch):
        monkeypatch.delenv(ERRORS_AS_FAILURES_ENV, raising=False)
        runs = [{"raw_eval_results": extract_minimal_eval_data(RAW)}]
        assert compute_accuracy(runs) == 1.0


class TestErrorsAsFailures:
    def test_string_task_becomes_failed_record(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "1")
        result = extract_minimal_eval_data(RAW)
        assert set(result) == {"ok", "died", "prompt"}
        died = result["died"]
        assert set(died) == set(result["ok"]) | {"error"}
        assert died["reward"] == 0.0
        assert died["cost"] == 0.0
        assert died["action_names"] == []
        assert died["confidence"] is None
        assert died["confidence_details"] == {
            "num_actions": 0,
            "num_errors": 1,
            "parsed_score": None,
        }
        assert died["error"] == "ERROR: boom"

    def test_error_text_truncated_to_200_chars(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "1")
        result = extract_minimal_eval_data({"t": "ERROR: " + "x" * 500})
        assert len(result["t"]["error"]) == 200

    def test_string_variation_becomes_score_zero(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "1")
        result = extract_minimal_eval_data(RAW)
        assert result["prompt"] == [{"score": 1.0}, {"score": 0}]

    def test_accuracy_counts_errors_as_failures(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "1")
        runs = [{"raw_eval_results": extract_minimal_eval_data(RAW)}]
        # ok=1, died=0, prompt variations 1 and 0 -> 2/4
        assert compute_accuracy(runs) == 0.5

    def test_predictability_skips_converted_records(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "1")
        runs = [{"raw_eval_results": extract_minimal_eval_data(RAW)}]
        metrics = compute_predictability_metrics(runs)
        assert metrics["mean_confidence"] == 0.9

    def test_other_values_do_not_enable(self, monkeypatch):
        monkeypatch.setenv(ERRORS_AS_FAILURES_ENV, "0")
        assert "died" not in extract_minimal_eval_data(RAW)


class TestCountErrorEntries:
    def test_counts_top_level_and_variation_strings(self):
        assert count_error_entries(RAW) == 2

    def test_zero_without_errors(self):
        assert count_error_entries({"ok": RAW["ok"], "p": [{"score": 1.0}]}) == 0
