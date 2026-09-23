"""The agent's own actions, without tau-bench's reward replay.

Env.calculate_reward replays the gold actions through env.step, so records
written before tool_calling.py split them off store taken_actions = the agent's
actions + that replay. Newer records carry the replay in reward_replay_actions.
"""

import json
from types import SimpleNamespace

from reliability_eval.loaders.actions import KEEP_REWARD_REPLAY_ENV, agent_actions
from reliability_eval.loaders.results import extract_minimal_eval_data
from reliability_eval.phases import abstention, safety
from reliability_eval.types import EvaluationLog

LOOKUP = {"name": "get_reservation_details", "kwargs": {"reservation_id": "Z7GOZK"}}
CANCEL = {"name": "cancel_reservation", "kwargs": {"reservation_id": "Z7GOZK"}}
TRANSFER = {"name": "transfer_to_human_agents", "kwargs": {"summary": "x"}}
BYE = {"name": "respond", "kwargs": {"content": "Goodbye"}}
TASK = {"user_id": "u", "instruction": "cancel", "actions": [CANCEL], "outputs": []}


def legacy(taken, task=TASK, **extra):
    """A record as stored before the split: replay (if any) inside taken_actions."""
    return {"reward": 0.0, "taken_actions": taken, "task": task, **extra}


class TestAgentActions:
    def test_split_record_is_already_clean(self):
        rec = legacy([LOOKUP, BYE], reward_replay_actions=[CANCEL])
        assert agent_actions(rec) == [LOOKUP, BYE]

    def test_legacy_record_drops_the_replay_after_the_customer_stop(self):
        assert agent_actions(legacy([LOOKUP, BYE, CANCEL])) == [LOOKUP, BYE]

    def test_legacy_record_drops_the_replay_after_a_terminate_tool(self):
        task = {**TASK, "actions": [CANCEL, TRANSFER]}  # the replay skips TRANSFER
        rec = legacy([LOOKUP, TRANSFER, CANCEL], task=task)
        assert agent_actions(rec) == [LOOKUP, TRANSFER]

    def test_legacy_record_keeps_the_agents_own_gold_actions(self):
        # The agent did the right thing, then the replay repeated it.
        rec = legacy([LOOKUP, CANCEL, BYE, CANCEL])
        assert agent_actions(rec) == [LOOKUP, CANCEL, BYE]

    def test_unscored_episode_is_kept(self):
        # The gold tail does not follow a customer reply or terminate tool, so
        # the episode ended unscored (step cap) and the tail is the agent's own.
        assert agent_actions(legacy([LOOKUP, CANCEL])) == [LOOKUP, CANCEL]
        assert agent_actions(legacy([LOOKUP, BYE])) == [LOOKUP, BYE]

    def test_task_without_replayable_gold_is_kept(self):
        rec = legacy([LOOKUP, TRANSFER], task={**TASK, "actions": [TRANSFER]})
        assert agent_actions(rec) == [LOOKUP, TRANSFER]

    def test_matches_the_transcript_rule(self):
        # Every action but the replay comes from one assistant message.
        history = [
            {"role": "system", "content": "wiki"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "Goodbye"},
            {"role": "user", "content": "###STOP###"},
        ]
        rec = legacy([LOOKUP, BYE, CANCEL], conversation_history=history)
        n = sum(m["role"] == "assistant" for m in history)
        assert agent_actions(rec) == rec["taken_actions"][:n]

    def test_non_taubench_record_is_kept(self):
        rec = {"reward": 1, "taken_actions": [BYE, CANCEL], "task": "a GAIA question"}
        assert agent_actions(rec) == [BYE, CANCEL]
        assert agent_actions({"taken_actions": [BYE, CANCEL]}) == [BYE, CANCEL]

    def test_missing_actions(self):
        assert agent_actions({"reward": 0.0, "task": TASK}) == []

    def test_legacy_view_returns_what_the_dashboard_scored(self, monkeypatch):
        monkeypatch.setenv(KEEP_REWARD_REPLAY_ENV, "1")
        assert agent_actions(legacy([LOOKUP, BYE, CANCEL])) == [LOOKUP, BYE, CANCEL]
        split = legacy([LOOKUP, BYE], reward_replay_actions=[CANCEL])
        assert agent_actions(split) == [LOOKUP, BYE, CANCEL]


class TestLoader:
    def test_trajectory_and_action_count_are_the_agents(self):
        rec = legacy([LOOKUP, BYE, CANCEL], confidence_details={"num_actions": 3})
        out = extract_minimal_eval_data({"1": rec})["1"]
        assert out["action_names"] == ["get_reservation_details", "respond"]
        assert out["confidence_details"]["num_actions"] == 2

    def test_split_record_counts_are_unchanged(self):
        rec = legacy(
            [LOOKUP, BYE],
            reward_replay_actions=[CANCEL],
            confidence_details={"num_actions": 2},
        )
        out = extract_minimal_eval_data({"1": rec})["1"]
        assert out["action_names"] == ["get_reservation_details", "respond"]
        assert out["confidence_details"]["num_actions"] == 2

    def test_legacy_view(self, monkeypatch):
        monkeypatch.setenv(KEEP_REWARD_REPLAY_ENV, "1")
        rec = legacy(
            [LOOKUP, BYE],
            reward_replay_actions=[CANCEL],
            confidence_details={"num_actions": 2},
        )
        out = extract_minimal_eval_data({"1": rec})["1"]
        assert out["action_names"] == ["get_reservation_details", "respond", "cancel_reservation"]
        assert out["confidence_details"]["num_actions"] == 3

    def test_action_count_not_derived_from_taken_actions_is_kept(self):
        rec = legacy([LOOKUP, BYE, CANCEL], confidence_details={"num_actions": 7})
        out = extract_minimal_eval_data({"1": rec})["1"]
        assert out["confidence_details"]["num_actions"] == 7


def write_run(tmp_path, record):
    run_dir = tmp_path / "taubench_airline" / "taubench_airline_agentx_rep1"
    run_dir.mkdir(parents=True)
    upload = run_dir / "taubench_airline_agentx_rep1_UPLOAD.json"
    upload.write_text(json.dumps({"raw_eval_results": {"1": record}}))
    return upload


def evaluation_log():
    return EvaluationLog(start_time="t", config={}, phases_to_run=[])


class FakeAnalyzer:
    """Records the actions the LLM judge would be shown."""

    seen = []

    def __init__(self, **kwargs):
        pass

    def analyze_compliance(self, conversation_history, actions_taken, constraints):
        FakeAnalyzer.seen.append(("compliance", actions_taken))
        return SimpleNamespace(S_comp=1.0, violations=[])

    def analyze_error_severity(self, conversation_history, actions_taken, task_result):
        FakeAnalyzer.seen.append(("severity", actions_taken))
        return SimpleNamespace(errors=[], S_cost=0.0, S_tail_max=0.0)


def test_safety_judge_sees_only_the_agents_actions(tmp_path, monkeypatch):
    import hal.utils.llm_log_analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "LLMLogAnalyzer", FakeAnalyzer)
    FakeAnalyzer.seen = []
    history = [{"role": "assistant", "content": "Goodbye"}]
    upload = write_run(tmp_path, legacy([LOOKUP, BYE, CANCEL], conversation_history=history))

    n = safety.run_safety_phase(
        [({"name": "agentx"}, {}, "taubench_airline")],
        tmp_path,
        "judge",
        ["no_pii_exposure"],
        evaluation_log(),
        tmp_path / "log.json",
    )

    assert n == 1
    assert FakeAnalyzer.seen == [("compliance", [LOOKUP, BYE]), ("severity", [LOOKUP, BYE])]
    # the stored record itself is not rewritten
    stored = json.loads(upload.read_text())["raw_eval_results"]["1"]
    assert stored["taken_actions"] == [LOOKUP, BYE, CANCEL]


def test_abstention_sees_only_the_agents_actions(tmp_path, monkeypatch):
    seen = []
    real = abstention.detect_abstention

    def spy(conversation_history, actions_taken):
        seen.append(actions_taken)
        return real(conversation_history, actions_taken)

    monkeypatch.setattr(abstention, "detect_abstention", spy)
    history = [{"role": "assistant", "content": "Goodbye"}]
    write_run(tmp_path, legacy([LOOKUP, BYE, CANCEL], conversation_history=history))

    abstention.run_abstention_phase(
        [({"name": "agentx"}, {}, "taubench_airline")],
        tmp_path,
        evaluation_log(),
        tmp_path / "log.json",
    )

    assert seen == [[LOOKUP, BYE]]
