"""The episode's tau-bench environment and the reliability instruments that
alter it: fault injection, compliance monitoring, structural perturbation, the
prompt-sensitivity instruction and user style, and the reward-replay tracker.

tau_bench is imported inside the functions, never at module level: its modules
bind litellm.completion when first imported, which must happen after run() has
installed its wrapper (see litellm_patch).
"""

from typing import Dict, Optional

from hal.utils.compliance_checkers import ComplianceMonitor
from hal.utils.fault_injection import FaultInjector
from hal.utils.taubench_perturbations import (
    TauBenchPerturbator,
    create_taubench_perturbator,
)


def enabled(kwargs: dict, flag: str) -> bool:
    """An on/off agent arg, given as a bool or as the string "true"."""
    return kwargs.get(flag) == "true" or kwargs.get(flag) is True


def make_fault_injector(kwargs: dict) -> Optional[FaultInjector]:
    if not enabled(kwargs, "enable_fault_injection"):
        return None
    fault_rate = float(kwargs.get("fault_rate", 0.2))
    fault_injector = FaultInjector(fault_rate=fault_rate)
    print(f"🔧 Fault injection enabled with rate={fault_rate}")
    return fault_injector


def make_compliance_monitor(kwargs: dict) -> Optional[ComplianceMonitor]:
    if not enabled(kwargs, "enable_compliance_monitoring"):
        return None
    constraints_input = kwargs.get(
        "compliance_constraints", "no_pii_exposure,no_destructive_ops"
    )
    # Handle both string and list inputs
    if isinstance(constraints_input, list):
        constraints = [c.strip() for c in constraints_input if c.strip()]
    else:
        constraints = [
            c.strip() for c in str(constraints_input).split(",") if c.strip()
        ]
    compliance_monitor = ComplianceMonitor(constraints=constraints)
    print(f"📋 Compliance monitoring enabled with constraints: {constraints}")
    return compliance_monitor


def make_perturbator(kwargs: dict) -> Optional[TauBenchPerturbator]:
    """The tau-bench structural perturbator: it modifies tool responses,
    parameter names and data formats to test robustness to API variations."""
    if not enabled(kwargs, "enable_structural_perturbations"):
        return None
    perturbation_strength = kwargs.get("perturbation_strength", "medium")
    taubench_perturbator = create_taubench_perturbator(strength=perturbation_strength)
    print(
        f"🔀 Tau-bench structural perturbations enabled: strength={perturbation_strength}"
    )
    print(
        f"   Config: key_case={taubench_perturbator.config.key_case}, "
        f"time_format={taubench_perturbator.config.time_format}, "
        f"param_style={taubench_perturbator.config.param_name_style}"
    )
    return taubench_perturbator


def make_env(task: dict, task_id: str, env_provider: str):
    """tau-bench's env for the task, with the prompt-sensitivity instruction
    and the user's communication style applied when the task carries them."""
    from tau_bench.envs import get_env

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        task["env"],
        task["user_strategy"],
        task["user_model"],
        task["task_split"],
        env_provider,  # Use env_provider for user simulation (always OpenAI-compatible)
        task["task_index"],
    )

    # Support for prompt sensitivity: Override instruction if provided
    if "instruction" in task and task["instruction"] is not None:
        # Override the task instruction with the provided one (for prompt sensitivity evaluation)
        original_instruction = isolated_env.task.instruction
        isolated_env.task.instruction = task["instruction"]
        print(f"🔀 Using custom instruction for task {task_id}")
        print(f"   Original length: {len(original_instruction)} chars")
        print(f"   Variation length: {len(task['instruction'])} chars")

    # Support for prompt variation style: Inject style directive into user's system prompt
    # This ensures the simulated user communicates in the specified style (casual, naturalistic, etc.)
    if "prompt_variation_strength" in task:
        from hal.utils.prompt_variation import get_user_style_directive

        style_strength = task["prompt_variation_strength"]
        style_directive = get_user_style_directive(style_strength)

        if style_directive:
            # Monkey-patch the user's build_system_prompt for any future calls
            original_build_system_prompt = isolated_env.user.build_system_prompt

            def styled_build_system_prompt(instruction=None):
                base_prompt = original_build_system_prompt(instruction)
                return base_prompt + style_directive

            isolated_env.user.build_system_prompt = styled_build_system_prompt

            # CRITICAL: Also update the EXISTING system prompt in user's messages
            # get_env() already called user.reset() which created the messages list
            # with the original system prompt. We need to patch that too.
            if hasattr(isolated_env.user, "messages") and isolated_env.user.messages:
                for msg in isolated_env.user.messages:
                    if msg.get("role") == "system":
                        msg["content"] = msg["content"] + style_directive
                        break

            print(f"🎭 User communication style set to: {style_strength}")

    return isolated_env


def perturb_env(isolated_env, taubench_perturbator: Optional[TauBenchPerturbator]):
    """Apply the structural perturbations, if any, to what the agent sees:

    1. Tool parameter names (what the agent must use to call tools)
    2. Tool responses (what the agent receives back)
    3. Wiki/knowledge base format (a no-op, see 2. below)

    Returns (tools_info, wiki, param_mapping) for the agent; param_mapping maps
    tool_name -> {new_param: old_param} and is empty when nothing is perturbed.
    """
    perturbed_tools_info = isolated_env.tools_info
    perturbed_wiki = isolated_env.wiki
    param_mapping: Dict[str, Dict[str, str]] = {}
    if not taubench_perturbator:
        return perturbed_tools_info, perturbed_wiki, param_mapping

    from tau_bench.types import RESPOND_ACTION_NAME

    # 1. Perturb tool definitions (including parameter names)
    # This returns a mapping so we can reverse param names when tools are called
    perturbed_tools_info, param_mapping = taubench_perturbator.perturb_tool_definitions(
        isolated_env.tools_info
    )

    # 2. "Perturb" the wiki. A no-op: perturb_tool_response only rewrites
    # JSON and returns anything else unchanged, and the wiki is markdown.
    perturbed_wiki = taubench_perturbator.perturb_tool_response(isolated_env.wiki)

    # 3. Wrap the environment's step function to perturb tool responses
    # and reverse parameter name mapping
    original_step = isolated_env.step

    def perturbed_step(action):
        """Intercept tool calls to reverse param names and perturb responses."""
        # Reverse parameter names if this tool was perturbed
        if hasattr(action, "kwargs") and hasattr(action, "name"):
            if action.name in param_mapping:
                # Create a copy with original param names for the real env
                original_kwargs = taubench_perturbator.reverse_param_mapping(
                    action.name, action.kwargs, param_mapping
                )
                # Modify kwargs in place (action is mutable)
                action.kwargs = original_kwargs

        # Call original step
        result = original_step(action)

        # Perturb the response (tool output). Env.step returns an
        # EnvResponse whose observation is the tool's (usually JSON) output;
        # for `respond` it is the simulated customer's reply, not a tool
        # response, so it passes through untouched.
        if action.name != RESPOND_ACTION_NAME:
            return result.model_copy(
                update={
                    "observation": taubench_perturbator.perturb_tool_response(
                        result.observation
                    )
                }
            )

        return result

    # Monkey-patch the step function
    isolated_env.step = perturbed_step

    summary = taubench_perturbator.get_perturbation_summary()
    print(f"🔀 Applied {summary['total_perturbations']} structural perturbations")
    if summary["by_type"]:
        print(f"   By type: {summary['by_type']}")

    return perturbed_tools_info, perturbed_wiki, param_mapping


def track_reward_replay(env):
    """Tell the agent's actions apart from the reward replay in env.actions.

    tau-bench's Env.calculate_reward scores an episode by replaying the task's
    gold actions through env.step, which appends each of them to env.actions,
    so after scoring env.actions ends with the answer key. This wraps
    env.calculate_reward to snapshot env.actions on entry, before the replay.
    It only passes the call through, so it chains with any other wrapper on
    the same env (e.g. the tool-fault one), installed before or after it.

    Returns split() -> (agent_actions, reward_replay_actions).
    """
    inner_reward = env.calculate_reward
    snapshot = []

    def calculate_reward():
        # The first scoring ends the episode; keep its snapshot.
        if not snapshot:
            snapshot.append(list(env.actions))
        return inner_reward()

    def split():
        agent = snapshot[0] if snapshot else list(env.actions)
        return agent, env.actions[len(agent) :]

    env.calculate_reward = calculate_reward
    return split
