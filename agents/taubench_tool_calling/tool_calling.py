"""The tau-bench agent entrypoint (`--agent_function tool_calling.run`).

run() patches litellm (litellm_patch), sets up the episode's environment and
reliability instruments (env_setup), runs the agent loop the `scaffold` agent
arg picks, and assembles the result record (postprocess).
"""

# The detector the post-hoc abstention phase runs; this file carried a copy of it,
# which returned the same output on every stored task record.
from reliability_eval.metrics.abstention import detect_abstention as _detect_abstention

# hal's local_runner loads this file by path from a copy of its directory, where
# the siblings are top-level modules; the tests import it as a package module.
if __package__:
    from . import env_setup, litellm_patch, postprocess
else:
    import env_setup
    import litellm_patch
    import postprocess

# The patch surface. run() looks these three up in this module's globals, so a
# test replaces them here (monkeypatch.setattr(tool_calling, "_detect_abstention",
# ...)); replacing them in env_setup, postprocess or reliability_eval would not
# reach run(). Everything else run() calls as an attribute of its module
# (env_setup.make_env, ...), so it is replaced on that module.
_track_reward_replay = env_setup.track_reward_replay
_compute_confidence_score = postprocess.compute_confidence_score

# The alternative agent loops live in the `rp` package (rp/scaffolds/), not in
# this directory: they are experiment code with their own lifecycle, and this
# file is the published protocol. local_runner copies this directory to a tmpdir
# and spawns a bare `python`, so `rp` has to be importable from the venv that
# owns hal-eval; hal-job.sbatch exports PYTHONPATH for that, and
# `rp check imports` verifies it before any GPU time is spent.


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert "provider" in kwargs, "provider is required. choose from openai or anthropic"
    task_id = list(input.keys())[0]

    import litellm

    litellm.drop_params = True
    litellm_patch.install_anthropic_adaptive_thinking_patch()

    # ========== RELIABILITY METRICS INITIALIZATION ==========
    fault_injector = env_setup.make_fault_injector(kwargs)

    # Agent loop: tc (published protocol), react, or a CLI scaffold (rp.scaffolds.bridge).
    scaffold = kwargs.get("scaffold", "tc")
    # fault_mode=llm (published protocol): faults wrap every litellm call, with
    # simulated internal recovery. fault_mode=tool: tool calls fail before
    # reaching the env and the agent sees the error, whatever the scaffold.
    fault_mode = kwargs.get("fault_mode", "llm")
    llm_fault_injector = fault_injector if fault_mode == "llm" else None

    compliance_monitor = env_setup.make_compliance_monitor(kwargs)
    taubench_perturbator = env_setup.make_perturbator(kwargs)
    # Resolved before the episode, so a missing endpoint fails the task at once.
    llm_analysis_api_base = (
        postprocess.llm_analysis_api_base()
        if env_setup.enabled(kwargs, "enable_llm_analysis")
        else None
    )

    # Separate providers for user simulation vs agent
    # User simulation needs OpenAI-compatible API (for tau-bench internals)
    env_provider = "openai"
    route = litellm_patch.resolve_agent_route(kwargs)
    model_name, agent_provider = route.model_name, route.provider
    original_completion = litellm_patch.install_wrappers(
        litellm, kwargs, route, llm_fault_injector
    )

    # Only now: tau-bench binds litellm.completion on import.
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent

    isolated_env = env_setup.make_env(input[task_id], task_id, env_provider)
    perturbed_tools_info, perturbed_wiki, param_mapping = env_setup.perturb_env(
        isolated_env, taubench_perturbator
    )

    finalize_tool_faults = None
    if fault_injector and fault_mode == "tool":
        from rp.scaffolds.bridge import install_tool_faults

        finalize_tool_faults = install_tool_faults(isolated_env, fault_injector)

    split_actions = _track_reward_replay(isolated_env)

    ### YOUR AGENT CODE HERE ###
    temperature = kwargs["temperature"] if "temperature" in kwargs else 0.0
    if scaffold == "tc":
        agent = ToolCallingAgent(
            tools_info=perturbed_tools_info,
            wiki=perturbed_wiki,
            model=model_name,
            provider=agent_provider,  # Use agent_provider for the agent (can be anthropic)
            temperature=temperature,
        )
        output = agent.solve(isolated_env, task_index=input[task_id]["task_index"])
    elif scaffold == "react":
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        agent = ChatReActAgent(
            tools_info=perturbed_tools_info,
            wiki=perturbed_wiki,
            model=model_name,
            provider=agent_provider,
            use_reasoning=True,
            temperature=temperature,
        )
        output = agent.solve(isolated_env, task_index=input[task_id]["task_index"])
    else:
        from rp.scaffolds.bridge import CLI_SCAFFOLDS, solve_cli

        assert scaffold in CLI_SCAFFOLDS, f"unknown scaffold {scaffold!r}"
        assert route.api_base, "CLI scaffolds need a self-hosted api_base"
        output = solve_cli(
            scaffold,
            isolated_env,
            tools_info=perturbed_tools_info,
            wiki=perturbed_wiki,
            task_index=input[task_id]["task_index"],
            served_model=model_name,
            api_base=route.api_base,
            task_id=task_id,
            context_window=int(kwargs.get("context_window", 65536)),
        )

    if finalize_tool_faults:
        finalize_tool_faults()

    agent_actions, reward_replay_actions = split_actions()
    conversation_history = output.messages if hasattr(output, "messages") else []

    ### DETECT ABSTENTION/DEFERRAL BEHAVIOR ###
    # Always compute abstention detection (lightweight, rule-based)
    abstention_result = _detect_abstention(
        conversation_history=conversation_history,
        actions_taken=agent_actions,
    )
    if abstention_result["abstained"]:
        print(
            f"🛑 Abstention detected: type={abstention_result['abstention_type']}, "
            f"strength={abstention_result['abstention_strength']:.2f}"
        )

    ### COMPUTE CONFIDENCE (OPTIONAL) ###
    confidence = None
    confidence_details = None
    if kwargs.get("compute_confidence", False):
        confidence, confidence_details = _compute_confidence_score(
            model_name=model_name,
            provider=agent_provider,
            api_base=route.api_base,
            api_key=route.api_key,
            task_description=isolated_env.task.model_dump(),
            conversation_history=output.messages,
            reward=isolated_env.reward,
            actions_taken=agent_actions,
            original_completion_fn=original_completion,  # Use unwrapped litellm.completion
            confidence_max_tokens=int(kwargs.get("confidence_max_tokens", 65536)),
        )

    ### COMPLIANCE CHECKING (OPTIONAL) ###
    compliance_violations = []
    if compliance_monitor:
        compliance_violations = postprocess.check_compliance(
            compliance_monitor, isolated_env.task, agent_actions
        )

    ### LLM-BASED LOG ANALYSIS (OPTIONAL) ###
    llm_compliance_result = None
    llm_recovery_result = None
    if llm_analysis_api_base:
        llm_compliance_result, llm_recovery_result = postprocess.analyze_with_llm(
            kwargs, llm_analysis_api_base, conversation_history, agent_actions
        )

    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {
        task_id: postprocess.build_record(
            env=isolated_env,
            output=output,
            scaffold=scaffold,
            agent_actions=agent_actions,
            reward_replay_actions=reward_replay_actions,
            abstention_result=abstention_result,
            confidence=confidence,
            confidence_details=confidence_details,
            store_confidence_details=kwargs.get("store_confidence_details", False),
            fault_injector=fault_injector,
            fault_mode=fault_mode,
            compliance_monitor=compliance_monitor,
            compliance_violations=compliance_violations,
            taubench_perturbator=taubench_perturbator,
            param_mapping=param_mapping,
            llm_compliance_result=llm_compliance_result,
            llm_recovery_result=llm_recovery_result,
            store_conversation_history=env_setup.enabled(
                kwargs, "store_conversation_history"
            ),
            conversation_history=conversation_history,
        )
    }
