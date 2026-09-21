"""Tests for the dual-formula outcome consistency (primary vs pre-PR#179 legacy)."""

import math

import pytest

from reliability_eval.constants import EPSILON
from reliability_eval.metrics.consistency import (
    compute_consistency_metrics,
    compute_outcome_consistency,
    compute_outcome_consistency_legacy,
)


class TestLegacyVsPrimary:
    def test_partial_agreement(self):
        successes = [1, 1, 0, 1, 0]  # p_hat = 0.6, sample var (ddof=1) = 0.3
        assert compute_outcome_consistency(successes) == pytest.approx(0.04)
        expected_legacy = max(0.0, 1 - 0.3 / (0.6 * 0.4 + EPSILON))
        assert expected_legacy == 0.0
        assert compute_outcome_consistency_legacy(successes) == pytest.approx(0.0)

    def test_unanimous_success(self):
        assert compute_outcome_consistency([1, 1, 1, 1, 1]) == pytest.approx(1.0)
        assert compute_outcome_consistency_legacy([1, 1, 1, 1, 1]) == pytest.approx(
            1.0
        )

    def test_unanimous_failure(self):
        assert compute_outcome_consistency_legacy([0, 0, 0, 0, 0]) == pytest.approx(
            1.0
        )

    def test_single_run_returns_nan(self):
        assert math.isnan(compute_outcome_consistency_legacy([1]))


class TestLegacySurfacedInMetrics:
    def _runs(self, rewards_by_task):
        """Build minimal baseline runs from {task_id: [reward per run]}."""
        n_runs = len(next(iter(rewards_by_task.values())))
        return [
            {
                "raw_eval_results": {
                    tid: {"reward": rs[i]} for tid, rs in rewards_by_task.items()
                }
            }
            for i in range(n_runs)
        ]

    def test_per_task_and_aggregate(self):
        runs = self._runs({"a": [1, 1, 0, 1, 0], "b": [1, 1, 1, 1, 1]})
        result = compute_consistency_metrics(runs)

        task_df = result["task_df"].set_index("task_id")
        assert task_df.loc["a", "consistency_outcome"] == pytest.approx(0.04)
        assert task_df.loc["a", "consistency_outcome_legacy"] == pytest.approx(0.0)
        assert task_df.loc["b", "consistency_outcome_legacy"] == pytest.approx(1.0)

        # Primary metric is untouched; legacy is an extra value alongside it.
        assert result["consistency_outcome"] == pytest.approx((0.04 + 1.0) / 2)
        assert result["consistency_outcome_legacy"] == pytest.approx(0.5)
        assert "consistency_outcome_legacy_se" in result

    def test_too_few_runs_yields_nan(self):
        result = compute_consistency_metrics(self._runs({"a": [1]}))
        assert math.isnan(result["consistency_outcome_legacy"])


def test_legacy_column_in_dataframe():
    from reliability_eval.metrics.agent import metrics_to_dataframe
    from reliability_eval.types import ReliabilityMetrics

    m = ReliabilityMetrics(agent_name="x")
    m.consistency_outcome = 0.52
    m.extra["consistency_outcome_legacy"] = 0.5
    df = metrics_to_dataframe([m])

    cols = list(df.columns)
    assert cols.index("consistency_outcome_legacy") == cols.index("consistency_outcome") + 1
    assert df.loc[0, "consistency_outcome"] == pytest.approx(0.52)
    assert df.loc[0, "consistency_outcome_legacy"] == pytest.approx(0.5)
    assert "consistency_outcome_legacy_se" in cols
