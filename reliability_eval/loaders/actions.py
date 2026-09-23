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

_RESPOND = "respond"
# Env.terminate_tools in both tau-bench domains (airline, retail). The replay
# skips them, and calling one ends (and scores) the episode.
_TERMINATE_TOOLS = ("transfer_to_human_agents",)


def keep_reward_replay() -> bool:
    return os.environ.get(KEEP_REWARD_REPLAY_ENV) == "1"


def agent_actions(task_eval: dict) -> list:
    """taken_actions without tau-bench's reward replay (see module docstring).

    Records that carry reward_replay_actions are already clean. Older ones end
    with the replay exactly when the episode was scored, i.e. when the
    agent's last action was a customer reply (the customer said ###STOP###)
    or a terminate tool; the replay is then the task's gold actions minus the
    terminate tools, in order. Records that are not tau-bench's (no gold
    actions in "task") are returned unchanged.
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
    if (
        replay
        and n > 0
        and taken[n:] == replay
        and taken[n - 1].get("name") in (_RESPOND, *_TERMINATE_TOOLS)
    ):
        return taken[:n]
    return taken
