"""What run() does to litellm before the episode starts.

It picks where the agent's model is served (resolve_agent_route) and replaces
litellm.completion / litellm.acompletion with wrappers (install_wrappers) that
route agent calls there, add reasoning parameters, inject the published
LLM-level faults and repair provider quirks in the response. tau-bench's
modules bind litellm.completion when first imported, so every LLM call of the
episode, agent and user simulator alike, goes through the wrapper as long as
tau-bench is imported after install_wrappers; nothing here imports it.
"""

import asyncio
import os
import random
import time
from typing import NamedTuple, Optional

from hal.utils.fault_injection import FaultEvent, FaultInjector

_ANTHROPIC_ADAPTIVE_REQUIRED = ("claude-opus-4-7",)
_BUDGET_TO_EFFORT = {1024: "low", 8192: "medium", 24576: "high"}
_anthropic_adaptive_patch_installed = False


def install_anthropic_adaptive_thinking_patch():
    """Translate `thinking.type=enabled` + `budget_tokens` -> `thinking.type=adaptive`
    + `output_config.effort` for Anthropic models that reject the old format.
    litellm 1.76.1 always emits the old format from `reasoning_effort`; Claude
    Opus 4.7 rejects it. The patch rewrites the request body just before send.

    Note: install-once flag is kept at module scope, not as a class attribute on
    AnthropicConfig, because litellm's `get_config()` merges every class attr
    into the request body (Anthropic would 400 on the unknown field).
    """
    global _anthropic_adaptive_patch_installed
    if _anthropic_adaptive_patch_installed:
        return
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    _original = AnthropicConfig.transform_request

    def _patched(self, model, messages, optional_params, litellm_params, headers):
        data = _original(
            self, model, messages, optional_params, litellm_params, headers
        )
        if not any(slug in model for slug in _ANTHROPIC_ADAPTIVE_REQUIRED):
            return data
        thinking = data.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            effort = _BUDGET_TO_EFFORT.get(thinking.get("budget_tokens"), "medium")
            data["thinking"] = {"type": "adaptive"}
            data["output_config"] = {"effort": effort}
        # Opus 4.7 deprecated `temperature` (the API rejects it). Strip silently.
        data.pop("temperature", None)
        return data

    AnthropicConfig.transform_request = _patched
    _anthropic_adaptive_patch_installed = True


class AgentRoute(NamedTuple):
    """Where the agent's own calls go. A call is the agent's when its model is
    model_name, so user simulation falls through to litellm's defaults
    (OPENAI_BASE_URL / OPENAI_API_BASE) unless the user model is served under
    the agent's name, in which case its calls take this route too."""

    provider: str
    api_base: Optional[str]
    api_key: Optional[str]
    model_name: str
    is_native_openai: bool


def _responses_model_name(model: str) -> str:
    return model if model.startswith("responses/") else f"responses/{model}"


def resolve_agent_route(kwargs: dict) -> AgentRoute:
    """Determine provider configuration from the agent args and model_name prefix."""
    if kwargs.get("api_base") or os.getenv("HAL_AGENT_API_BASE"):
        # Self-hosted OpenAI-compatible endpoint (e.g. vLLM). Model name is the
        # served id, passed verbatim. Only agent calls get this api_base; user
        # simulation falls through to OPENAI_API_BASE / OPENAI_BASE_URL.
        agent_provider = kwargs.get("provider", "openai")
        api_base = kwargs.get("api_base") or os.getenv("HAL_AGENT_API_BASE")
        api_key = (
            kwargs.get("api_key")
            or os.getenv("HAL_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        model_name = kwargs["model_name"]
    elif "openrouter/" in kwargs["model_name"]:
        # Use OpenRouter - strip the openrouter/ prefix
        agent_provider = "openai"
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        model_name = kwargs["model_name"].replace(
            "openrouter/", ""
        )  # Remove openrouter/ prefix

        # Configure litellm for OpenRouter
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
    elif "together_ai" in kwargs["model_name"]:
        # Use Together AI via OpenAI-compatible API
        agent_provider = "openai"
        api_base = "https://api.together.xyz/v1"
        api_key = os.getenv("TOGETHERAI_API_KEY")
        model_name = kwargs["model_name"].replace("together_ai/", "")
    elif "gemini" in kwargs["model_name"]:
        # Use Google Gemini via OpenAI-compatible API
        agent_provider = "openai"
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = kwargs["model_name"].replace("gemini/", "")
    elif "claude" in kwargs["model_name"]:
        # Use Anthropic directly
        agent_provider = "anthropic"
        api_base = None  # litellm handles Anthropic API endpoint automatically
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model_name = kwargs["model_name"]  # Use the full model name
    else:
        # Default to provider parameter
        agent_provider = kwargs["provider"]
        api_base = None
        api_key = None
        model_name = kwargs["model_name"]

    is_native_openai = agent_provider == "openai" and api_base is None
    if is_native_openai:
        model_name = _responses_model_name(model_name)
    return AgentRoute(agent_provider, api_base, api_key, model_name, is_native_openai)


def install_wrappers(
    litellm,
    kwargs: dict,
    route: AgentRoute,
    llm_fault_injector: Optional[FaultInjector],
):
    """Replace litellm.completion and litellm.acompletion; return the unwrapped
    litellm.completion (the confidence call bypasses the wrapper).

    The sync and async wrappers share everything but the faulted call itself:
    the sync one goes through FaultInjector.wrap_call, the async one draws its
    own faults, and the two do not behave alike (wrap_call's exception-type
    faults skip the recovery loop), so each keeps its own.
    """
    model_name = route.model_name
    api_base, api_key = route.api_base, route.api_key
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    def prepare(completion_kwargs):
        """Add the route's endpoint, key and headers, and reasoning parameters,
        to agent calls."""
        # Check if this is a call with our agent's model
        is_agent_call = (
            "model" in completion_kwargs and completion_kwargs["model"] == model_name
        )

        # Send agent calls to the route's api_base (self-hosted vLLM, OpenRouter,
        # Together or Gemini) with its key and headers; other calls (the user
        # simulator's, under a different model name) keep litellm's defaults.
        if api_base and is_agent_call:
            completion_kwargs["api_base"] = api_base
            completion_kwargs["api_key"] = api_key
            extra_headers = completion_kwargs.get("extra_headers", {})
            extra_headers["HTTP-Referer"] = (
                "https://github.com/benediktstroebl/hal-harness"
            )
            extra_headers["X-Title"] = "HAL Harness"
            completion_kwargs["extra_headers"] = extra_headers

        if is_agent_call:
            if "reasoning_effort" in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs["temperature"] = 1.0

                if "openrouter/" in kwargs["model_name"]:
                    # OpenRouter convention: reasoning.max_tokens via extra_body.
                    # Aligned with Anthropic-direct / Gemini-direct conventions:
                    # medium=8192 matches direct API behaviour. Was {low:1024, medium:2048, high:4096}.
                    effort_to_tokens = {"low": 1024, "medium": 8192, "high": 24576}
                    reasoning_tokens = effort_to_tokens.get(
                        kwargs["reasoning_effort"], 4096
                    )
                    extra_body = completion_kwargs.get("extra_body", {})
                    extra_body["reasoning"] = {"max_tokens": reasoning_tokens}
                    extra_body["include_reasoning"] = True
                    completion_kwargs["extra_body"] = extra_body
                    print(
                        f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}"
                    )
                elif route.is_native_openai:
                    completion_kwargs["reasoning"] = {
                        "effort": kwargs["reasoning_effort"]
                    }
                    completion_kwargs.pop("reasoning_effort", None)
                    print(
                        f"Setting reasoning.effort to {kwargs['reasoning_effort']} for model {model_name}"
                    )
                else:
                    # For direct Anthropic and other non-OpenAI providers
                    completion_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
                    print(
                        f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}"
                    )

    def repair(response):
        """Fill in what some providers leave out of a response."""
        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if (
            hasattr(response, "_hidden_params")
            and response._hidden_params.get("response_cost") is None
        ):
            response._hidden_params["response_cost"] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, "function") and tool_call.function:
                        if (
                            not tool_call.function.arguments
                            or tool_call.function.arguments.strip() == ""
                        ):
                            tool_call.function.arguments = "{}"

            # Fix null content (Gemini returns null content when making tool calls)
            # The OpenAI API rejects messages with null content, so we set it to empty string
            if message.content is None:
                message.content = ""

        return response

    def completion_with_reasoning(*args, **completion_kwargs):
        prepare(completion_kwargs)
        # Call the original function - with fault injection if enabled
        if llm_fault_injector and llm_fault_injector.enabled:
            try:
                response = llm_fault_injector.wrap_call(
                    original_completion, *args, **completion_kwargs
                )
            except Exception as call_error:
                # Not an injected fault: wrap_call turns those into a retry or a
                # stand-in result and never raises them. This is the call
                # itself failing while fault injection is on.
                print(
                    f"⚠️  LLM call failed under fault injection: "
                    f"{type(call_error).__name__}: {call_error}"
                )
                raise
        else:
            response = original_completion(*args, **completion_kwargs)
        return repair(response)

    async def acompletion_with_reasoning(*args, **completion_kwargs):
        prepare(completion_kwargs)
        # Call the original function - with fault injection if enabled
        if llm_fault_injector and llm_fault_injector.enabled:
            response = await _afaulted(
                llm_fault_injector, original_acompletion, *args, **completion_kwargs
            )
        else:
            response = await original_acompletion(*args, **completion_kwargs)
        return repair(response)

    # Replace both sync and async completion functions
    litellm.completion = completion_with_reasoning
    litellm.acompletion = acompletion_with_reasoning
    return original_completion


async def _afaulted(fault_injector, original_acompletion, *args, **completion_kwargs):
    """The async counterpart of FaultInjector.wrap_call. wrap_call is sync, so
    faults are injected here by hand to keep the call awaitable."""
    if random.random() < fault_injector.fault_rate:
        # Inject fault
        fault_type = fault_injector._select_fault_type()
        fault_injector.state["faults_injected"] += 1
        print(f"⚡ Async fault injected: {fault_type.value}")

        # Attempt recovery with retries
        max_retries = fault_injector.config.get("max_recovery_attempts", 3)
        recovered = False
        recovery_start = time.time()

        for attempt in range(max_retries):
            recovery_prob = 0.3 + (attempt * 0.2)
            if random.random() < recovery_prob:
                response = await original_acompletion(*args, **completion_kwargs)
                recovered = True
                fault_injector.state["recoveries_successful"] += 1
                break
            await asyncio.sleep(0.1 * (attempt + 1))

        if not recovered:
            fault_injector.state["recoveries_failed"] += 1
            # Raise fault exception - recovery failed after all attempts
            raise RuntimeError(
                f"Simulated fault after {max_retries} recovery attempts: {fault_type.value}"
            )

        recovery_time = time.time() - recovery_start
        fault_injector.state["total_recovery_time"] += recovery_time

        # Log fault event
        fault_event = FaultEvent(
            fault_type=fault_type,
            recovered=recovered,
            recovery_time=recovery_time,
            context={
                "recovery_attempts": attempt + 1,
                "function_name": "acompletion",
            },
        )
        fault_injector.fault_events.append(fault_event)
        return response
    return await original_acompletion(*args, **completion_kwargs)
