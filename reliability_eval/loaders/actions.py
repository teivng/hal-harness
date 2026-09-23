"""The agent's own actions from a stored tau-bench task record.

tau-bench's Env.calculate_reward scores an episode by replaying the task's gold
actions through env.step, which appends them to env.actions. Until
tool_calling.py split them off (into reward_replay_actions), every scored
episode stored taken_actions = the agent's actions + that replay, so trajectory
consistency, the safety judge and abstention all saw actions the agent never
took. agent_actions() is the one place that undoes this for stored records.
"""

import os

# "1" returns what the published dashboard scored: the agent's actions followed
# by the reward replay, for side-by-side reporting against it.
KEEP_REWARD_REPLAY_ENV = "HAL_KEEP_REWARD_REPLAY"

# Which of the two views a judgement was made on, stamped into what the safety
# judge and the abstention phase write (under ACTIONS_VIEW_KEY). A judgement
# without the stamp predates the split and saw the replay.
ACTIONS_VIEW_KEY = "actions_view"
AGENT_VIEW = "agent"
REPLAY_VIEW = "agent+replay"

_RESPOND = "respond"
# Env.terminate_tools in both tau-bench domains (airline, retail). The replay
# skips them, and calling one ends (and scores) the episode.
_TERMINATE_TOOLS = ("transfer_to_human_agents",)
# What the simulated customer says to end the episode (Env.step checks it).
_CUSTOMER_STOP = "###STOP###"


def keep_reward_replay() -> bool:
    return os.environ.get(KEEP_REWARD_REPLAY_ENV) == "1"


def actions_view() -> str:
    """The view agent_actions() returns in this process."""
    return REPLAY_VIEW if keep_reward_replay() else AGENT_VIEW


def judged_view(judgement: dict) -> str:
    """The view a stored llm_safety / abstention entry was made on."""
    return judgement.get(ACTIONS_VIEW_KEY, REPLAY_VIEW)


def _scored(task_eval: dict, last_action: dict) -> bool:
    """Was the episode scored, given the action before the candidate replay?

    Only a scored episode has the replay appended. The evidence, strongest first:
    the CLI bridge's end_reason says outright; a terminate tool always ends and
    scores the episode; and a final `respond` scored it only if the customer
    answered with ###STOP###, which the transcript shows. Without a transcript
    (tc/tctf/react fault units store none) a final `respond` is taken as the
    customer's stop: an unscored episode would need its last actions to equal
    the gold actions in order, and on the stored records every such candidate
    replay is two or more gold actions long.
    """
    end_reason = (task_eval.get("scaffold") or {}).get("end_reason")
    if end_reason is not None:
        return end_reason == "customer_stop" or end_reason.startswith("terminal_tool:")
    name = last_action.get("name")
    if name in _TERMINATE_TOOLS:
        return True
    if name != _RESPOND:
        return False
    history = task_eval.get("conversation_history")
    if history:
        return _CUSTOMER_STOP in str(history[-1].get("content", ""))
    return True


def agent_actions(task_eval: dict) -> list:
    """taken_actions without tau-bench's reward replay (see module docstring).

    Records that carry reward_replay_actions are already clean. Older ones end
    with the replay exactly when the episode was scored (see _scored); the
    replay is then the task's gold actions minus the terminate tools, in order.
    Records that are not tau-bench's (no gold actions in "task") are returned
    unchanged.
    """
    taken = task_eval.get("taken_actions", [])
    if "reward_replay_actions" in task_eval:
        if keep_reward_replay():
            return taken + task_eval["reward_replay_actions"]
        return taken
    if keep_reward_replay():
        return taken

    task = task_eval.get("task")
    gold = task.get("actions") if isinstance(task, dict) else None
    if not gold:
        return taken
    replay = [a for a in gold if a.get("name") not in _TERMINATE_TOOLS]
    n = len(taken) - len(replay)
    if replay and n > 0 and taken[n:] == replay and _scored(task_eval, taken[n - 1]):
        return taken[:n]
    return taken
