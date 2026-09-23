"""Golden tests for tool_calling.run: the whole episode, byte for byte.

The published rows came out of tool_calling.run, so a change to its structure
must not change what it stores or what it asks the model. Each scenario runs
one real airline episode against a scripted LLM (agent, user simulator,
confidence and analysis calls alike) and compares six parts (PARTS) with a
stored golden: the result dict; every call that reached litellm, after run()'s
routing wrapper; a probe of the acompletion wrapper run() leaves installed
(replies, _hidden_params, calls, stdout, and the fault injectors' stats and
events); the CLI bridge's arguments; litellm's global flags; and stdout. The
comparison is on canonical JSON text, so key order and int/float types count.

The scripted LLM replaces litellm.completion before run() wraps it, and
tau-bench's modules call litellm.completion late, as they do in a task process
that imports tau-bench only after run() has patched litellm.

Regenerate the goldens only for an intended behaviour change:
    HAL_UPDATE_GOLDEN=1 python -m pytest tests/agents/test_tool_calling_golden.py
"""

import asyncio
import json
import os
import random
import time
import shutil
import subprocess
import sys
import types
from pathlib import Path

import litellm
import pytest
import tau_bench.agents.chat_react_agent as react_agent
import tau_bench.agents.tool_calling_agent as tc_agent
import tau_bench.envs.user as tb_user
from tau_bench.envs.airline.tasks_test import TASKS
from tau_bench.types import RESPOND_ACTION_NAME, Action, SolveResult

import hal.utils.compliance_checkers as compliance_checkers
import hal.utils.fault_injection as fault_injection
from agents.taubench_tool_calling import tool_calling
from tests.agents.scripted_llm import (
    AGENT,
    AUX_BASE,
    RESERVATION,
    SCENARIO_ENV,
    SCENARIOS,
    TASK_INDEX,
    USER,
    Clock,
    FrozenDatetime,
    canonical,
    normalize,
    recorder,
    task_input,
)
from tests.agents.test_tool_calling_actions import install_faults_like_bridge

HARNESS = Path(__file__).resolve().parents[2]
AGENT_DIR = HARNESS / "agents" / "taubench_tool_calling"
GOLDEN = Path(__file__).parent / "golden" / "tool_calling"
TASK_ID = "7"
PARTS = ("result", "calls", "async_probe", "cli_calls", "litellm_flags", "stdout")


def golden(name):
    return json.loads((GOLDEN / f"{name}.json").read_text())


def assert_same_text(got, want, parts):
    for part in parts:
        assert canonical(got[part]) == canonical(want[part]), part


@pytest.fixture
def scripted(monkeypatch):
    """Scripted LLM underneath run()'s wrapper; every call it sees is recorded,
    and so is every FaultInjector run() creates."""
    calls = []
    injectors = []
    real_init = fault_injection.FaultInjector.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        injectors.append(self)

    monkeypatch.setattr(fault_injection.FaultInjector, "__init__", init)
    completion, acompletion = recorder(calls)
    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "drop_params", litellm.drop_params)
    monkeypatch.setattr(litellm, "modify_params", litellm.modify_params)
    # In a task process tau-bench is first imported inside run(), after the
    # wrapper is installed; here it was imported first, so look it up late.
    late = lambda *a, **k: litellm.completion(*a, **k)  # noqa: E731
    for module in (tc_agent, react_agent, tb_user):
        monkeypatch.setattr(module, "completion", late)
    # run() overrides the instruction on tau-bench's shared Task object, which
    # a task process discards and a test session would carry to the next test.
    monkeypatch.setattr(TASKS[TASK_INDEX], "instruction", TASKS[TASK_INDEX].instruction)
    monkeypatch.setattr(fault_injection, "time", Clock)
    monkeypatch.setattr(fault_injection, "datetime", FrozenDatetime)
    monkeypatch.setattr(compliance_checkers, "datetime", FrozenDatetime)
    for var in (
        "HAL_AGENT_API_BASE",
        "HAL_AGENT_API_KEY",
        "HAL_JUDGE_MAX_TOKENS",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ):
        monkeypatch.delenv(var, raising=False)
    for var in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHERAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(var, f"key-{var}")

    cli_calls = []

    def solve_cli(scaffold, env, **kwargs):
        cli_calls.append(
            {"scaffold": scaffold, **kwargs, "tools_info": len(kwargs["tools_info"])}
        )
        env.reset(task_index=kwargs["task_index"])
        env.step(
            Action(
                name="get_reservation_details", kwargs={"reservation_id": RESERVATION}
            )
        )
        done = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Goodbye"}))
        return SolveResult(
            reward=done.reward,
            info={"scaffold": {"name": scaffold, "turns": 2}},
            messages=[],
        )

    bridge = types.ModuleType("rp.scaffolds.bridge")
    bridge.install_tool_faults = install_faults_like_bridge
    bridge.CLI_SCAFFOLDS = ("opencode", "codex", "claude")
    bridge.solve_cli = solve_cli
    for name in ("rp", "rp.scaffolds"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "rp.scaffolds.bridge", bridge)
    return calls, cli_calls, injectors


async def _no_sleep(_):
    return None


def _async_probe(monkeypatch):
    """Drive the acompletion wrapper run() leaves installed: agent and non-agent
    calls, one reply with content None, with whatever fault injector the
    scenario configured. Its recovery clock and back-off are frozen."""
    monkeypatch.setattr(time, "time", lambda: 0.0)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    outcomes = []
    probes = [(AGENT, "probe"), (USER, "probe"), (AGENT, "probe-none")]
    for model, content in probes + [(AGENT, "probe")] * 14:
        try:
            res = asyncio.run(
                litellm.acompletion(
                    model=model, messages=[{"role": "user", "content": content}]
                )
            )
            outcomes.append(
                {
                    "message": res.choices[0].message.model_dump(),
                    "hidden_params": res._hidden_params,
                }
            )
        except Exception as e:  # the injector raises when recovery fails
            outcomes.append(f"{type(e).__name__}: {e}")
    return outcomes


def _run(name, monkeypatch, capsys, scripted):
    calls, cli_calls, injectors = scripted
    kwargs = {"model_name": AGENT, **SCENARIOS[name][0]}
    for var, value in SCENARIO_ENV.get(name, {}).items():
        monkeypatch.setenv(var, value)
    random.seed(1234)
    try:
        result = tool_calling.run({TASK_ID: task_input(name)}, **kwargs)
    except Exception as e:
        result = {"raised": f"{type(e).__name__}: {e}"}
    n_sync = len(calls)
    stdout = capsys.readouterr().out
    random.seed(99)
    probe = _async_probe(monkeypatch)
    return normalize(
        {
            "result": result,
            "calls": calls[:n_sync],
            "async_probe": {
                "outcomes": probe,
                "calls": calls[n_sync:],
                "stdout": capsys.readouterr().out,
                "fault_injectors": [
                    {
                        "stats": i.get_stats(),
                        "events": [e.to_dict() for e in i.get_fault_events()],
                    }
                    for i in injectors
                ],
            },
            "cli_calls": cli_calls,
            "litellm_flags": {
                "drop_params": litellm.drop_params,
                "modify_params": litellm.modify_params,
            },
            "stdout": stdout,
        }
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_run_matches_golden(name, monkeypatch, capsys, scripted):
    got = _run(name, monkeypatch, capsys, scripted)
    path = GOLDEN / f"{name}.json"
    if os.environ.get("HAL_UPDATE_GOLDEN"):
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(got))
    assert_same_text(got, golden(name), PARTS)
    assert canonical(got) == path.read_text()


def test_scenarios_cover_the_published_units(monkeypatch, capsys, scripted):
    """The goldens are only evidence if the episodes they record did something:
    the tc episode cancels the reservation, is scored, abstains and is rated."""
    rec = _run("tc_rep", monkeypatch, capsys, scripted)["result"][TASK_ID]
    assert [a["name"] for a in rec["taken_actions"]] == [
        "get_reservation_details",
        "cancel_reservation",
        RESPOND_ACTION_NAME,
    ]
    assert rec["reward"] == 1.0
    assert rec["abstention"]["abstained"] is True
    assert rec["confidence"] == 0.85


# The way hal's local_runner runs the entrypoint: a copy of the agent directory,
# a bare `python` in it, tool_calling.py loaded by path (not as a package).
DRIVER = """
import importlib.util, json, os, random, sys

import hal.utils.compliance_checkers as compliance_checkers
import hal.utils.fault_injection as fault_injection
import litellm
from tests.agents.scripted_llm import Clock, FrozenDatetime, normalize, recorder

calls = []
litellm.completion, litellm.acompletion = recorder(calls)
fault_injection.time = Clock
fault_injection.datetime = compliance_checkers.datetime = FrozenDatetime

spec = importlib.util.spec_from_file_location(
    "tool_calling", os.path.join(os.getcwd(), "tool_calling.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "tau_bench" not in sys.modules, "tau-bench imported before run() patched litellm"

with open("input.json") as f:
    input_data = json.load(f)
with open("agent_args.json") as f:
    agent_args = json.load(f)
random.seed(1234)
result = module.run(input_data, **agent_args)
with open("output.json", "w") as f:
    json.dump(normalize({"result": result, "calls": calls}), f)
"""


@pytest.mark.parametrize(
    "name",
    ["tc_rep", "tc_fault", "tc_struct", "tc_prompt", "react_rep", "react_struct"],
)
def test_runner_copy_reproduces_golden(name, tmp_path):
    """Same episode through local_runner's loading path: sibling modules must
    import from the copied directory, and tau-bench must bind the wrapper."""
    shutil.copytree(AGENT_DIR, tmp_path, dirs_exist_ok=True)
    (tmp_path / "input.json").write_text(json.dumps({TASK_ID: task_input(name)}))
    (tmp_path / "agent_args.json").write_text(
        json.dumps({"model_name": AGENT, **SCENARIOS[name][0]})
    )
    (tmp_path / "run_agent.py").write_text(DRIVER)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HAL_AGENT_")}
    env["PYTHONPATH"] = str(HARNESS)
    proc = subprocess.run(
        [sys.executable, "run_agent.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads((tmp_path / "output.json").read_text())
    assert_same_text(
        {**got, "stdout": normalize(proc.stdout)},
        golden(name),
        ("result", "calls", "stdout"),
    )


def test_empty_tool_arguments_reach_the_env_as_an_empty_object(
    monkeypatch, capsys, scripted
):
    rec = _run("tc_empty_tool_args", monkeypatch, capsys, scripted)["result"][TASK_ID]
    first = rec["conversation_history"][2]
    assert first["tool_calls"][0]["function"]["arguments"] == "{}"
    assert rec["taken_actions"][0] == {"name": "list_all_airports", "kwargs": {}}


def test_llm_analysis_refuses_to_run_without_a_self_hosted_endpoint(
    monkeypatch, capsys, scripted
):
    """Zero dollars: with no aux endpoint, litellm would send the default
    gpt-4o-mini analysis to api.openai.com. run() raises before the episode."""
    got = _run("tc_llm_analysis_no_endpoint", monkeypatch, capsys, scripted)
    assert got["result"]["raised"].startswith("RuntimeError: enable_llm_analysis")
    assert got["calls"] == []


def test_llm_analysis_calls_go_to_the_aux_endpoint(monkeypatch, capsys, scripted):
    got = _run("tc_llm_analysis", monkeypatch, capsys, scripted)
    judged = [
        c["kwargs"] for c in got["calls"] if c["kwargs"]["model"] == "gpt-4o-mini"
    ]
    assert len(judged) == 2
    assert all(k["api_base"] == AUX_BASE for k in judged)
