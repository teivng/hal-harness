"""Offline tests for what tool_calling.run stores about the episode.

No GPU and no LLM: the tau-bench user simulator is a scripted one and the
agent loop a scripted stand-in for ToolCallingAgent, so the real airline Env
(its step, its calculate_reward and its gold-action replay) runs unmodified.
"""

import json
import random
import sys
import types

import litellm
import pytest
import tau_bench.agents.tool_calling_agent as tc_agent
import tau_bench.envs.base as tb_base
from tau_bench.envs import get_env
from tau_bench.types import RESPOND_ACTION_NAME, Action, EnvInfo, EnvResponse, SolveResult

from agents.taubench_tool_calling import tool_calling

TASK_INDEX = 1  # gold: cancel_reservation(Z7GOZK)
RESERVATION = "Z7GOZK"


class ScriptedUser:
    """Stands in for the LLM user simulator: fixed opener, stops on goodbye."""

    def reset(self, instruction=None):
        return "Hi, I want to cancel my reservation."

    def step(self, content):
        return "###STOP###" if "goodbye" in content.lower() else "Please go ahead."

    def get_total_cost(self):
        return 0.0


class ScriptedAgent:
    """Stands in for ToolCallingAgent: looks the reservation up (under whatever
    parameter name the tool schema it was given uses), then says goodbye, which
    ends the episode and makes the env score it. Never cancels, so the gold
    cancel_reservation appears in env.actions only through the reward replay."""

    def __init__(self, tools_info, **kwargs):
        self.tools_info = tools_info

    def solve(self, env, task_index=None):
        env.reset(task_index=task_index)
        schema = next(
            t["function"]
            for t in self.tools_info
            if t["function"]["name"] == "get_reservation_details"
        )
        (param,) = schema["parameters"]["properties"]
        messages = [
            {"role": "system", "content": "wiki"},
            {"role": "user", "content": "Hi"},
        ]
        call = {"name": "get_reservation_details", "arguments": json.dumps({param: RESERVATION})}
        tool = env.step(Action(name="get_reservation_details", kwargs={param: RESERVATION}))
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "function": call}]},
            {"role": "tool", "tool_call_id": "c0", "content": tool.observation},
        ]
        bye = env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Goodbye"}))
        messages += [
            {"role": "assistant", "content": "Goodbye"},
            {"role": "user", "content": bye.observation},
        ]
        return SolveResult(reward=env.reward, info={}, messages=messages)


def install_faults_like_bridge(env, injector):
    """The calculate_reward/step wrapping of rp.scaffolds.bridge.faults
    install_tool_faults, which run() installs for fault_mode=tool: every tool
    call fails with probability fault_rate, but never during the reward replay."""
    inner = env.step
    inner_reward = env.calculate_reward
    scoring = [False]

    def calculate_reward():
        scoring[0] = True
        try:
            return inner_reward()
        finally:
            scoring[0] = False

    def step(action):
        if scoring[0]:
            return inner(action)
        if action.name in env.tools_map and random.random() < injector.fault_rate:
            return EnvResponse(
                observation="Error: request timed out",
                reward=0.0,
                done=False,
                info=EnvInfo(task=env.task, source=action.name),
            )
        return inner(action)

    env.step = step
    env.calculate_reward = calculate_reward
    return lambda: None


@pytest.fixture
def offline(monkeypatch):
    """Scripted user and agent; undo run()'s global litellm patch afterwards."""
    monkeypatch.setattr(tb_base, "load_user", lambda **kwargs: ScriptedUser())
    monkeypatch.setattr(tc_agent, "ToolCallingAgent", ScriptedAgent)
    monkeypatch.setattr(litellm, "completion", litellm.completion)
    monkeypatch.setattr(litellm, "acompletion", litellm.acompletion)
    bridge = types.ModuleType("rp.scaffolds.bridge")
    bridge.install_tool_faults = install_faults_like_bridge
    for name in ("rp", "rp.scaffolds"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "rp.scaffolds.bridge", bridge)


def run_episode(**kwargs):
    task = {
        "env": "airline",
        "user_strategy": "llm",
        "user_model": "stub",
        "task_split": "test",
        "task_index": TASK_INDEX,
    }
    out = tool_calling.run(
        {"1": task},
        model_name="served/model",
        provider="openai",
        api_base="http://127.0.0.1:9/v1",
        store_conversation_history=True,
        **kwargs,
    )
    return out["1"]


def names(actions):
    return [a["name"] for a in actions]


def airline_env():
    original = tb_base.load_user
    tb_base.load_user = lambda **kwargs: ScriptedUser()
    try:
        return get_env("airline", "llm", "stub", "test", "openai", TASK_INDEX)
    finally:
        tb_base.load_user = original


# ── Bug 1: the gold replay must not be stored as the agent's actions ──────────


@pytest.mark.parametrize("faults_first", [False, True])
def test_track_reward_replay_splits_off_the_gold_replay(faults_first):
    env = airline_env()
    injector = types.SimpleNamespace(fault_rate=0.0)
    if faults_first:
        install_faults_like_bridge(env, injector)
    split = tool_calling._track_reward_replay(env)
    if not faults_first:
        install_faults_like_bridge(env, injector)

    env.reset(task_index=TASK_INDEX)
    env.step(Action(name="get_reservation_details", kwargs={"reservation_id": RESERVATION}))
    assert split() == (env.actions, [])  # not scored yet: nothing replayed
    assert env.step(Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Goodbye"})).done

    agent, replay = split()
    assert [a.name for a in agent] == ["get_reservation_details", RESPOND_ACTION_NAME]
    assert [a.model_dump() for a in replay] == [a.model_dump() for a in env.task.actions]
    assert agent + replay == env.actions


def test_stored_taken_actions_exclude_the_reward_replay(offline, monkeypatch):
    seen = {}

    def fake_confidence(**kwargs):
        seen["confidence"] = list(kwargs["actions_taken"])
        return 0.5, {"num_actions": len(kwargs["actions_taken"])}

    real_abstention = tool_calling._detect_abstention

    def spy_abstention(conversation_history, actions_taken):
        seen["abstention"] = list(actions_taken)
        return real_abstention(conversation_history, actions_taken)

    monkeypatch.setattr(tool_calling, "_compute_confidence_score", fake_confidence)
    monkeypatch.setattr(tool_calling, "_detect_abstention", spy_abstention)

    rec = run_episode(compute_confidence=True, store_confidence_details=True)

    agent = ["get_reservation_details", RESPOND_ACTION_NAME]
    assert names(rec["taken_actions"]) == agent
    assert names(rec["reward_replay_actions"]) == ["cancel_reservation"]
    assert rec["reward_replay_actions"] == rec["task"]["actions"]
    assert rec["reward"] == 0.0  # the agent never cancelled
    assert rec["confidence_details"]["num_actions"] == 2
    assert [a.name for a in seen["confidence"]] == agent
    assert [a.name for a in seen["abstention"]] == agent


def test_taken_actions_exclude_the_replay_under_tool_faults(offline):
    # Every tool call faults, so the lookup never reaches the env; the replay
    # runs through the fault wrapper's scoring bypass and still must not count.
    rec = run_episode(enable_fault_injection=True, fault_rate=1.0, fault_mode="tool")

    assert names(rec["taken_actions"]) == [RESPOND_ACTION_NAME]
    assert names(rec["reward_replay_actions"]) == ["cancel_reservation"]


# ── Bug 2: the struct unit must perturb what tools return ────────────────────


def tool_and_user_observations(rec):
    history = rec["conversation_history"]
    tool = next(m["content"] for m in history if m["role"] == "tool")
    return tool, history[-1]["content"]


def test_struct_run_perturbs_tool_observations(offline):
    rec = run_episode(enable_structural_perturbations=True, perturbation_strength="medium")

    tool_obs, user_obs = tool_and_user_observations(rec)
    wrapped = json.loads(tool_obs)
    # medium: camelCase keys, wrapped in {status, data}, US dates
    assert wrapped["status"] == "success"
    assert wrapped["data"]["reservationId"] == RESERVATION
    assert "reservation_id" not in wrapped["data"]
    assert user_obs == "###STOP###"  # the customer's words are not a tool response
    # the env itself was called with the original parameter name
    assert rec["taken_actions"][0]["kwargs"] == {"reservation_id": RESERVATION}


def test_non_struct_run_passes_tool_observations_through(offline):
    rec = run_episode()

    tool_obs, user_obs = tool_and_user_observations(rec)
    env = airline_env()
    env.reset(task_index=TASK_INDEX)
    direct = env.step(
        Action(name="get_reservation_details", kwargs={"reservation_id": RESERVATION})
    )
    assert tool_obs == direct.observation
    assert json.loads(tool_obs)["reservation_id"] == RESERVATION
    assert user_obs == "###STOP###"
