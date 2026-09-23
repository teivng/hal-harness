"""The agent's own actions, without tau-bench's reward replay.

Env.calculate_reward replays the gold actions through env.step, so records
written before tool_calling.py split them off store taken_actions = the agent's
actions + that replay. Newer records carry the replay in reward_replay_actions.
"""

import json
from types import SimpleNamespace

import pytest

from reliability_eval.loaders.actions import (
    KEEP_REWARD_REPLAY_ENV,
    actions_view,
    agent_actions,
    judged_view,
)
from reliability_eval.loaders.results import extract_minimal_eval_data
from reliability_eval.metrics.safety import (
    compute_safety_metrics,
    warn_on_mixed_actions_views,
)
from reliability_eval.phases import abstention, safety
from reliability_eval.types import EvaluationLog

LOOKUP = {"name": "get_reservation_details", "kwargs": {"reservation_id": "Z7GOZK"}}
CANCEL = {"name": "cancel_reservation", "kwargs": {"reservation_id": "Z7GOZK"}}
TRANSFER = {"name": "transfer_to_human_agents", "kwargs": {"summary": "x"}}
BYE = {"name": "respond", "kwargs": {"content": "Goodbye"}}
TASK = {"user_id": "u", "instruction": "cancel", "actions": [CANCEL], "outputs": []}
# A scored episode's transcript ends with the customer's stop (tau-bench's
# ToolCallingAgent appends the customer's reply after every `respond`).
SCORED = [
    {"role": "assistant", "content": "Goodbye"},
    {"role": "user", "content": "###STOP###"},
]


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

    def test_scored_transcript_drops_the_replay(self):
        rec = legacy([LOOKUP, BYE, CANCEL], conversation_history=SCORED)
        assert agent_actions(rec) == [LOOKUP, BYE]

    def test_unscored_transcript_keeps_a_tail_that_matches_the_gold(self):
        # Step cap: the customer answered the goodbye without ###STOP###, and the
        # agent's own last action happens to be the gold one.
        history = [*SCORED[:1], {"role": "user", "content": "Wait, one more thing."}]
        rec = legacy([LOOKUP, BYE, CANCEL], conversation_history=history)
        assert agent_actions(rec) == [LOOKUP, BYE, CANCEL]

    @pytest.mark.parametrize("end", ["customer_stop", "terminal_tool:transfer_to_human_agents"])
    def test_scored_cli_episode_drops_the_replay(self, end):
        rec = legacy([LOOKUP, BYE, CANCEL], scaffold={"name": "opencode", "end_reason": end})
        assert agent_actions(rec) == [LOOKUP, BYE]

    @pytest.mark.parametrize("end", ["step_cap", "cli_no_reply"])
    def test_unscored_cli_episode_keeps_its_actions(self, end):
        # taubench_ocode_qwen3_4b fault_rep5 task 36: step cap, and the agent's
        # 30th action (get_reservation_details) equals the replay.
        task = {**TASK, "actions": [LOOKUP, TRANSFER]}
        rec = legacy(
            [BYE, BYE, LOOKUP], task=task, scaffold={"name": "opencode", "end_reason": end}
        )
        assert agent_actions(rec) == [BYE, BYE, LOOKUP]

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


class TestActionsView:
    def test_default_view_is_the_agents(self):
        assert actions_view() == "agent"

    def test_legacy_view(self, monkeypatch):
        monkeypatch.setenv(KEEP_REWARD_REPLAY_ENV, "1")
        assert actions_view() == "agent+replay"

    def test_an_unstamped_judgement_saw_the_replay(self):
        assert judged_view({"analyzed": True}) == "agent+replay"
        assert judged_view({"analyzed": True, "actions_view": "agent"}) == "agent"


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


def judge(tmp_path, **kwargs):
    return safety.run_safety_phase(
        [({"name": "agentx"}, {}, "taubench_airline")],
        tmp_path,
        "judge",
        ["no_pii_exposure"],
        evaluation_log(),
        tmp_path / "log.json",
        **kwargs,
    )


@pytest.fixture
def fake_judge(monkeypatch):
    import hal.utils.llm_log_analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "LLMLogAnalyzer", FakeAnalyzer)
    FakeAnalyzer.seen = []
    return FakeAnalyzer.seen


# The transcripts below end with the customer's ###STOP###, as a scored
# tau-bench transcript does. They used to end with the agent's goodbye, which
# no tau-bench transcript does and which the tightened rule reads as unscored.


def test_safety_judge_sees_only_the_agents_actions(tmp_path, fake_judge):
    upload = write_run(tmp_path, legacy([LOOKUP, BYE, CANCEL], conversation_history=SCORED))

    assert judge(tmp_path) == 1
    assert fake_judge == [("compliance", [LOOKUP, BYE]), ("severity", [LOOKUP, BYE])]
    # the stored record keeps its actions, and the verdict says what it saw
    stored = json.loads(upload.read_text())["raw_eval_results"]["1"]
    assert stored["taken_actions"] == [LOOKUP, BYE, CANCEL]
    assert stored["llm_safety"]["actions_view"] == "agent"


def test_safety_judge_stamps_the_legacy_view(tmp_path, fake_judge, monkeypatch):
    monkeypatch.setenv(KEEP_REWARD_REPLAY_ENV, "1")
    upload = write_run(tmp_path, legacy([LOOKUP, BYE, CANCEL], conversation_history=SCORED))

    judge(tmp_path)
    assert fake_judge[0] == ("compliance", [LOOKUP, BYE, CANCEL])
    stored = json.loads(upload.read_text())["raw_eval_results"]["1"]
    assert stored["llm_safety"]["actions_view"] == "agent+replay"


def prior_verdict(**extra):
    return {"analyzed": True, "model": "judge", "compliance_violations": [], **extra}


@pytest.mark.parametrize(
    "prior, rejudged",
    [
        (prior_verdict(), True),  # judged before the split: it saw the replay
        (prior_verdict(actions_view="agent+replay"), True),
        (prior_verdict(actions_view="agent"), False),
        ({"analyzed": False, "error": "truncated"}, True),
    ],
)
def test_resumed_judge_skips_only_verdicts_on_the_current_view(
    tmp_path, fake_judge, monkeypatch, prior, rejudged
):
    monkeypatch.setenv("HAL_SAFETY_SKIP_ANALYZED", "1")
    rec = legacy([LOOKUP, BYE, CANCEL], conversation_history=SCORED, llm_safety=prior)
    write_run(tmp_path, rec)

    assert judge(tmp_path) == int(rejudged)


def test_a_redo_rejudges_every_task(tmp_path, fake_judge, monkeypatch):
    """skip_analyzed=False overrides an inherited HAL_SAFETY_SKIP_ANALYZED=1:
    the posthoc jobs export it, and a redo must not keep a single old verdict."""
    monkeypatch.setenv("HAL_SAFETY_SKIP_ANALYZED", "1")
    rec = legacy(
        [LOOKUP, BYE, CANCEL],
        conversation_history=SCORED,
        llm_safety=prior_verdict(actions_view="agent"),
    )
    write_run(tmp_path, rec)

    assert judge(tmp_path, skip_analyzed=False) == 1


def test_abstention_sees_only_the_agents_actions(tmp_path, monkeypatch):
    seen = []
    real = abstention.detect_abstention

    def spy(conversation_history, actions_taken):
        seen.append(actions_taken)
        return real(conversation_history, actions_taken)

    monkeypatch.setattr(abstention, "detect_abstention", spy)
    upload = write_run(tmp_path, legacy([LOOKUP, BYE, CANCEL], conversation_history=SCORED))

    abstention.run_abstention_phase(
        [({"name": "agentx"}, {}, "taubench_airline")],
        tmp_path,
        evaluation_log(),
        tmp_path / "log.json",
    )

    assert seen == [[LOOKUP, BYE]]
    stored = json.loads(upload.read_text())["raw_eval_results"]["1"]
    assert stored["abstention"]["actions_view"] == "agent"


class TestMixedViews:
    def runs(self, *verdicts):
        return [{"raw_eval_results": {str(i): {"llm_safety": v} for i, v in enumerate(verdicts)}}]

    def test_safety_metrics_report_the_views_they_read(self):
        out = compute_safety_metrics(
            self.runs(prior_verdict(), prior_verdict(actions_view="agent"))
        )
        assert out["actions_views"] == ["agent", "agent+replay"]

    def test_a_panel_on_one_view_is_quiet(self, capsys):
        assert not warn_on_mixed_actions_views({"a": ["agent"], "b": ["agent"], "c": []})
        assert capsys.readouterr().err == ""

    def test_a_panel_mixing_views_is_flagged(self, capsys):
        assert warn_on_mixed_actions_views({"a": ["agent"], "b": ["agent+replay"]})
        err = capsys.readouterr().err
        assert "WARNING" in err and "a: agent" in err and "b: agent+replay" in err
