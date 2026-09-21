"""Result loading and preprocessing for reliability_eval."""

import json
import os
from collections import defaultdict
from pathlib import Path

from reliability_eval.loaders.agent_names import extract_agent_name
from reliability_eval.loaders.gaia_task_levels import extract_gaia_task_levels

# The harness writes an "ERROR: ..." string into raw_eval_results for a task whose
# subprocess died. Upstream (and the published dashboard) silently drop such
# entries from every metric. Set this env var to "1" to score them as failed
# tasks instead (reward 0, no confidence).
ERRORS_AS_FAILURES_ENV = "HAL_ERRORS_AS_FAILURES"


def errors_as_failures() -> bool:
    return os.environ.get(ERRORS_AS_FAILURES_ENV) == "1"


def count_error_entries(raw_eval: dict) -> int:
    """Count string ("ERROR: ...") entries in raw_eval_results.

    Counts top-level string entries and string entries inside prompt-sensitivity
    variation lists, i.e. everything extract_minimal_eval_data either drops
    (default) or converts to a failure (HAL_ERRORS_AS_FAILURES=1).
    """
    count = 0
    for task_eval in raw_eval.values():
        if isinstance(task_eval, str):
            count += 1
        elif isinstance(task_eval, list):
            count += sum(isinstance(v, str) for v in task_eval)
    return count


def _failed_task_record(error: str) -> dict:
    """Minimal record for an errored task, keyed like the normal dict branch."""
    return {
        "reward": 0.0,
        "cost": 0.0,
        "action_names": [],
        "confidence": None,
        "confidence_details": {"num_actions": 0, "num_errors": 1, "parsed_score": None},
        "abstention": {},
        "llm_safety": {},
        "error": error[:200],
    }


def detect_run_type(data: dict, run_dir_name: str) -> str:
    """Detect the type of run (baseline, fault, structural, prompt, etc.)."""
    agent_args = data.get("metadata", {}).get("agent_args", {})
    config = data.get("config", {})

    if agent_args.get("enable_fault_injection") == "true":
        return "fault"
    if agent_args.get("enable_structural_perturbations") == "true":
        return "structural"

    # Check for prompt sensitivity runs (via config or metadata)
    if config.get("prompt_sensitivity") or data.get("metadata", {}).get(
        "prompt_sensitivity"
    ):
        return "prompt"

    name_lower = run_dir_name.lower()
    if "fault" in name_lower:
        return "fault"
    if "struct" in name_lower or "perturbed" in name_lower:
        return "structural"
    if "prompt" in name_lower and (
        "sensitivity" in name_lower
        or "mild" in name_lower
        or "medium" in name_lower
        or "strong" in name_lower
        or "naturalistic" in name_lower
    ):
        return "prompt"

    return "baseline"


def extract_minimal_logging_data(raw_logging: list[dict]) -> list[dict]:
    """
    Extract only the minimal fields needed from raw_logging_results.
    This avoids keeping large conversation histories in memory.

    Fields extracted per entry:
    - weave_task_id: to map to tasks
    - summary.usage: to count API calls (we only need the length)
    - summary.weave.latency_ms: for latency metrics
    - token totals: prompt_tokens and completion_tokens summed across models
    """
    minimal = []
    # hal/utils/weave_utils.py::get_weave_calls changed shape after the reliability
    # runs: it used to return a list of per-call records (weave_task_id + summary),
    # and now returns {task_id: {messages, call_metadata: [{usage, started_at,
    # ended_at}, ...], model, call_count}}. Expand the new shape into the same
    # per-call records so every downstream metric (api-call count, per-call
    # latency, token totals) is computed identically for both.
    if isinstance(raw_logging, dict):
        expanded = []
        for task_id, entry in raw_logging.items():
            if not isinstance(entry, dict):
                continue
            model = entry.get("model", "unknown")
            for call in entry.get("call_metadata", []) or []:
                usage = call.get("usage") or {}
                latency_ms = None
                try:
                    from datetime import datetime

                    t0 = datetime.fromisoformat(call["started_at"])
                    t1 = datetime.fromisoformat(call["ended_at"])
                    latency_ms = (t1 - t0).total_seconds() * 1000.0
                except (KeyError, TypeError, ValueError):
                    pass
                expanded.append(
                    {
                        "weave_task_id": task_id,
                        "summary": {
                            "usage": {
                                model: {
                                    "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                                    "completion_tokens": usage.get("completion_tokens", 0) or 0,
                                }
                            },
                            "weave": {"latency_ms": latency_ms},
                        },
                    }
                )
        raw_logging = expanded
    for entry in raw_logging:
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("weave_task_id")
        if task_id is None:
            continue
        summary = entry.get("summary", {})
        usage = summary.get("usage", {})
        # Only store the count of usage entries, not the full usage dict
        usage_count = len(usage)
        latency_ms = summary.get("weave", {}).get("latency_ms")
        # Sum token counts across all models for cost estimation
        prompt_tokens = 0
        completion_tokens = 0
        for model_usage in usage.values():
            if isinstance(model_usage, dict):
                prompt_tokens += model_usage.get("prompt_tokens", 0)
                completion_tokens += model_usage.get("completion_tokens", 0)
        minimal.append(
            {
                "weave_task_id": task_id,
                "usage_count": usage_count,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
    return minimal


def extract_minimal_eval_data(raw_eval: dict) -> dict:
    """
    Extract only the minimal fields needed from raw_eval_results.
    This avoids keeping large action details, tool outputs, etc. in memory.

    Fields extracted per task (dict format):
    - reward: success/failure
    - cost: task cost
    - action_names: only action names (not full action objects)
    - confidence: confidence score
    - confidence_details: num_actions, num_errors, parsed_score
    - abstention: abstention data
    - llm_safety: safety analysis results

    For prompt sensitivity results (list format), preserves score/reward from each variation.

    String entries ("ERROR: ..." for a task whose subprocess died) are dropped by
    default, matching upstream. With HAL_ERRORS_AS_FAILURES=1 they become failed
    records instead (reward 0 / {"score": 0}); see count_error_entries.
    """
    convert_errors = errors_as_failures()
    minimal = {}
    for task_id, task_eval in raw_eval.items():
        if isinstance(task_eval, list):
            # Prompt sensitivity format: list of variation results
            # Extract only score/reward from each variation
            variations = []
            for v in task_eval:
                if isinstance(v, dict):
                    variations.append({"score": v.get("score", v.get("reward", 0))})
                elif convert_errors and isinstance(v, str):
                    variations.append({"score": 0})
            minimal[task_id] = variations
        elif convert_errors and isinstance(task_eval, str):
            minimal[task_id] = _failed_task_record(task_eval)
        elif isinstance(task_eval, dict):
            # Normal result format
            # Extract only action names from taken_actions
            taken_actions = task_eval.get("taken_actions", [])
            action_names = [
                a.get("name", "") for a in taken_actions if isinstance(a, dict)
            ]

            # Extract minimal confidence_details
            conf_details = task_eval.get("confidence_details", {})
            minimal_conf_details = {}
            if isinstance(conf_details, dict):
                minimal_conf_details = {
                    "num_actions": conf_details.get("num_actions", 0),
                    "num_errors": conf_details.get("num_errors", 0),
                    "parsed_score": conf_details.get("parsed_score"),
                }

            # Cost: try direct field first, then metrics.estimated_cost
            cost_val = task_eval.get("cost", 0.0)
            if not cost_val:
                cost_val = (
                    task_eval.get("metrics", {}).get("estimated_cost", 0.0)
                    if isinstance(task_eval.get("metrics"), dict)
                    else 0.0
                )

            minimal[task_id] = {
                "reward": task_eval.get("reward", 0.0),
                "cost": cost_val,
                "action_names": action_names,  # Pre-extracted action names
                "confidence": task_eval.get("confidence"),
                "confidence_details": minimal_conf_details,
                "abstention": task_eval.get("abstention", {}),
                "llm_safety": task_eval.get("llm_safety", {}),
            }
    return minimal


def load_all_results(results_dir: Path, benchmark: str) -> dict[str, dict]:
    """
    Load all evaluation results for a benchmark.
    Extracts only minimal fields needed for analysis to reduce memory usage.

    Args:
        results_dir: Path to results directory
        benchmark: Benchmark name
    """
    results = defaultdict(lambda: defaultdict(list))

    benchmark_dir = results_dir / benchmark
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        return {}

    print(f"📂 Loading results from: {benchmark_dir}")
    print("   (extracting minimal fields for memory efficiency)")

    run_dirs = [d for d in sorted(benchmark_dir.glob("*")) if d.is_dir()]
    total_dirs = len(run_dirs)
    loaded_count = 0

    for run_dir in run_dirs:
        upload_files = list(run_dir.glob("*_UPLOAD.json"))
        if not upload_files:
            continue

        try:
            with open(upload_files[0], "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n⚠️  Error loading {run_dir.name}: {e}")
            continue

        agent_name = extract_agent_name(run_dir.name, benchmark)
        run_type = detect_run_type(data, run_dir.name)

        # Extract minimal data from both logging and eval results
        raw_logging = data.get("raw_logging_results", [])
        logging_data = extract_minimal_logging_data(raw_logging)

        raw_eval = data.get("raw_eval_results", {})
        error_entries = count_error_entries(raw_eval)
        eval_data = extract_minimal_eval_data(raw_eval)

        # Extract task levels for GAIA benchmark
        task_levels = {}
        if benchmark == "gaia":
            task_levels = extract_gaia_task_levels(run_dir)

        run_data = {
            "run_id": run_dir.name,
            "raw_eval_results": eval_data,
            "raw_logging_results": logging_data,
            "latencies": data.get("results", {}).get("latencies", {}),
            "metadata": data.get("metadata", {}),
            "results": data.get("results", {}),
            "costs": data.get("results", {}).get("costs", {}),
            "task_levels": task_levels,  # Added for GAIA level-stratified analysis
            # Number of "ERROR: ..." task entries (dropped, or scored as failures
            # when HAL_ERRORS_AS_FAILURES=1).
            "error_entries": error_entries,
        }

        # Clear reference to allow GC of full data
        del data, raw_logging, raw_eval

        results[agent_name][run_type].append(run_data)
        loaded_count += 1
        print(f"\r   Loaded {loaded_count}/{total_dirs} runs...", end="", flush=True)

    print()  # Newline after progress
    for agent_name, run_types in results.items():
        counts = {rt: len(runs) for rt, runs in run_types.items()}
        print(f"✅ {agent_name}: {counts}")

    return results
