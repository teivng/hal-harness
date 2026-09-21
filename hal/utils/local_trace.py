"""Local, on-disk LLM call tracing, used when Weave is disabled (WEAVE_DISABLED=1).

Every task subprocess appends one JSON line per completed litellm call to
<trace_dir>/<task_id>.jsonl (see install_litellm_trace). process_results then
folds those files into the same cost / usage / logging / latency structures
that weave_utils.get_total_cost and get_weave_calls derive from Weave traces,
so the UPLOAD json carries the resource channels without anything leaving the
machine.
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple

from .weave_utils import cost_from_token_usage

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _field(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def trace_record(task_id, kwargs, completion_response, start_time, end_time):
    """Build the JSON record for one litellm success callback; None if the
    response carries no token usage (nothing to bill or count)."""
    usage = _field(completion_response, "usage")
    if usage is None:
        return None
    counts = {k: _field(usage, k) for k in USAGE_FIELDS}
    if counts["prompt_tokens"] is None and counts["completion_tokens"] is None:
        return None
    return {
        "task_id": task_id,
        "model": kwargs.get("model"),
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "usage": {k: v or 0 for k, v in counts.items()},
    }


def install_litellm_trace(task_id: str, trace_dir: str) -> None:
    """Register litellm success callbacks (sync and async) that append one
    record per completed call to <trace_dir>/<task_id>.jsonl."""
    import litellm

    path = os.path.join(trace_dir, f"{task_id}.jsonl")

    def record(kwargs, completion_response, start_time, end_time):
        rec = trace_record(task_id, kwargs, completion_response, start_time, end_time)
        if rec is not None:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    async def record_async(kwargs, completion_response, start_time, end_time):
        record(kwargs, completion_response, start_time, end_time)

    # litellm runs plain callables in success_callback only for sync calls and
    # awaits the ones in _async_success_callback for acompletion, so each call
    # is written exactly once.
    litellm.success_callback.append(record)
    litellm._async_success_callback.append(record_async)


def clear_task_traces(trace_dir: str, task_ids: Iterable[str]) -> None:
    """Drop the trace files of tasks about to be re-run (--continue_run)."""
    for task_id in task_ids:
        path = os.path.join(trace_dir, f"{task_id}.jsonl")
        if os.path.exists(path):
            os.remove(path)


def load_local_traces(trace_dir: str) -> Tuple[float, Dict, Dict, Dict]:
    """Return (total_cost, token_usage, raw_logging, latency_dict) in the shapes
    of weave_utils.get_total_cost / get_weave_calls. Missing dir -> empty."""
    token_usage: Dict[str, Dict[str, int]] = {}
    raw_logging: Dict[str, Dict[str, Any]] = {}
    latency_dict: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(trace_dir):
        return 0.0, token_usage, raw_logging, latency_dict

    for name in sorted(os.listdir(trace_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(trace_dir, name)) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            continue
        task_id = name[: -len(".jsonl")]
        records.sort(key=lambda r: r["started_at"])

        for r in records:
            usage = token_usage.setdefault(
                r["model"],
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )
            usage["prompt_tokens"] += r["usage"].get("prompt_tokens", 0)
            usage["completion_tokens"] += r["usage"].get("completion_tokens", 0)

        raw_logging[task_id] = {
            "messages": [],
            "call_metadata": [
                {
                    "usage": r["usage"],
                    "started_at": r["started_at"],
                    "ended_at": r["ended_at"],
                }
                for r in records
            ],
            # Agent and user-simulator calls share the file; the agent makes
            # the most calls, as the fullest Weave call did.
            "model": Counter(r["model"] for r in records).most_common(1)[0][0],
            "call_count": len(records),
        }
        first, last = records[0]["started_at"], records[-1]["started_at"]
        latency_dict[task_id] = {
            "first_call_timestamp": first,
            "last_call_timestamp": last,
            "total_time": (
                datetime.fromisoformat(last) - datetime.fromisoformat(first)
            ).total_seconds(),
        }

    return cost_from_token_usage(token_usage), token_usage, raw_logging, latency_dict
