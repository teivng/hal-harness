"""Tests for hal.utils.local_trace — the on-disk replacement for Weave traces."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import litellm
import pytest

from hal.utils.local_trace import (
    clear_task_traces,
    install_litellm_trace,
    load_local_traces,
)
from hal.utils.weave_utils import (
    CACHED_PRICE_OVERRIDES,
    MODEL_PRICES_DICT,
    cost_from_token_usage,
)
from reliability_eval.loaders.results import extract_minimal_logging_data

AGENT = "Qwen/Qwen3-4B-Instruct-2507"
AUX = "Qwen/Qwen3-32B"
T0 = datetime(2026, 9, 11, 12, 0, 0)


def _record(task_id, model, offset_s, duration_s, prompt, completion):
    start = T0 + timedelta(seconds=offset_s)
    return {
        "task_id": task_id,
        "model": model,
        "started_at": start.isoformat(),
        "ended_at": (start + timedelta(seconds=duration_s)).isoformat(),
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _write_fixture(trace_dir):
    """Two tasks, three calls. Task 0: agent, aux, agent (written out of order);
    task 1: one agent call."""
    files = {
        "0": [
            _record("0", AGENT, 10, 2.0, 300, 30),
            _record("0", AUX, 5, 1.0, 100, 10),
            _record("0", AGENT, 0, 1.5, 200, 20),
        ],
        "1": [_record("1", AGENT, 0, 0.5, 50, 5)],
    }
    for task_id, records in files.items():
        with open(os.path.join(trace_dir, f"{task_id}.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


@pytest.fixture
def trace_dir():
    with tempfile.TemporaryDirectory() as d:
        _write_fixture(d)
        yield d


class TestLoadLocalTraces:
    def test_missing_dir_is_empty(self):
        assert load_local_traces("/nonexistent/local_traces") == (0.0, {}, {}, {})

    def test_token_usage_per_model(self, trace_dir):
        _, usage, _, _ = load_local_traces(trace_dir)
        assert usage == {
            AGENT: {
                "prompt_tokens": 550,
                "completion_tokens": 55,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            AUX: {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        # Rationale: keyed by served name, same four fields as get_total_cost,
        # so the aux (user simulator) model is billed alongside the agent.

    def test_raw_logging_shape(self, trace_dir):
        _, _, raw_logging, _ = load_local_traces(trace_dir)
        assert set(raw_logging) == {"0", "1"}
        entry = raw_logging["0"]
        assert entry["messages"] == []
        assert entry["model"] == AGENT
        assert entry["call_count"] == 3
        assert [c["started_at"] for c in entry["call_metadata"]] == sorted(
            c["started_at"] for c in entry["call_metadata"]
        )
        assert entry["call_metadata"][0]["usage"]["prompt_tokens"] == 200
        assert entry["call_metadata"][1]["ended_at"] == (
            T0 + timedelta(seconds=6)
        ).isoformat()
        # Rationale: mirrors get_weave_calls — call_metadata sorted by
        # started_at, model of the task is the one with the most calls.

    def test_latency_dict(self, trace_dir):
        _, _, _, latency = load_local_traces(trace_dir)
        assert latency["0"] == {
            "first_call_timestamp": T0.isoformat(),
            "last_call_timestamp": (T0 + timedelta(seconds=10)).isoformat(),
            "total_time": 10.0,
        }
        assert latency["1"]["total_time"] == 0.0
        # Rationale: as in get_weave_calls, total_time spans first to last
        # *start*; a single call has zero span.

    def test_total_cost_matches_helper(self, trace_dir):
        cost, usage, _, _ = load_local_traces(trace_dir)
        assert cost > 0
        assert cost == cost_from_token_usage(usage)

    def test_feeds_reliability_loader(self, trace_dir):
        _, _, raw_logging, _ = load_local_traces(trace_dir)
        entries = extract_minimal_logging_data(raw_logging)
        task0 = [e for e in entries if e["weave_task_id"] == "0"]
        assert len(task0) == 3
        assert [e["usage_count"] for e in task0] == [1, 1, 1]
        assert [e["latency_ms"] for e in task0] == [1500.0, 1000.0, 2000.0]
        assert [e["prompt_tokens"] for e in task0] == [200, 100, 300]
        assert sum(e["completion_tokens"] for e in entries) == 65
        # Rationale: the API-call count, per-call latency and token channels
        # of resource consistency come out exactly as they did from Weave.


class TestCostFromTokenUsage:
    def test_formula(self):
        usage = {
            AUX: {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 200,
            },
            "unpriced-model": {
                "prompt_tokens": 10**9,
                "completion_tokens": 10**9,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        p = MODEL_PRICES_DICT[AUX]
        cached = CACHED_PRICE_OVERRIDES.get(AUX, p["prompt_tokens"])
        expected = (
            (1000 - 200) * p["prompt_tokens"]
            + 10 * cached
            + 200 * cached
            + 100 * p["completion_tokens"]
        )
        assert cost_from_token_usage(usage) == pytest.approx(expected)
        # Rationale: the loop lifted out of get_total_cost — fresh input,
        # cache writes/reads at the cached price, completions; unpriced skipped.


class TestInstallLitellmTrace:
    @pytest.fixture
    def callbacks(self, monkeypatch):
        monkeypatch.setattr(litellm, "success_callback", [])
        monkeypatch.setattr(litellm, "_async_success_callback", [])
        with tempfile.TemporaryDirectory() as d:
            install_litellm_trace("7", d)
            yield d, litellm.success_callback[0], litellm._async_success_callback[0]

    def test_sync_callback_writes_record(self, callbacks):
        d, sync_cb, _ = callbacks
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
        )
        sync_cb({"model": AUX}, response, T0, T0 + timedelta(seconds=2))
        with open(os.path.join(d, "7.jsonl")) as f:
            lines = f.read().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {
            "task_id": "7",
            "model": AUX,
            "started_at": T0.isoformat(),
            "ended_at": (T0 + timedelta(seconds=2)).isoformat(),
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

    def test_async_callback_and_missing_usage(self, callbacks):
        d, sync_cb, async_cb = callbacks
        sync_cb({"model": AUX}, SimpleNamespace(usage=None), T0, T0)
        assert not os.path.exists(os.path.join(d, "7.jsonl"))
        response = SimpleNamespace(usage={"prompt_tokens": 1, "completion_tokens": 1})
        asyncio.run(async_cb({"model": AGENT}, response, T0, T0))
        with open(os.path.join(d, "7.jsonl")) as f:
            (line,) = f.read().splitlines()
        assert json.loads(line)["usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 0,
        }
        # Rationale: no usage -> no record (nothing to bill); acompletion goes
        # through the async list and dict-shaped usage is accepted.


def test_clear_task_traces(trace_dir):
    clear_task_traces(trace_dir, ["0", "missing"])
    assert sorted(os.listdir(trace_dir)) == ["1.jsonl"]
    # Rationale: --continue_run re-runs a task from scratch, so its earlier
    # calls must not be counted twice.
