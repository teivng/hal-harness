"""compute_confidence_score keeps a thinking model's reasoning, and only then."""

import types

import pytest

from agents.taubench_tool_calling import postprocess


def reply(content, reasoning=None, completion_tokens=7):
    message = types.SimpleNamespace(content=content, role="assistant", tool_calls=None, function_call=None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(
        choices=[choice], usage=types.SimpleNamespace(completion_tokens=completion_tokens)
    )


def confidence(response):
    return postprocess.compute_confidence_score(
        model_name="m",
        provider="openai",
        api_base="http://127.0.0.1:1/v1",
        api_key="EMPTY",
        task_description={},
        conversation_history=[{"role": "user", "content": "hi"}],
        reward=1.0,
        actions_taken=[],
        original_completion_fn=lambda **kwargs: response,
        confidence_max_tokens=16,
    )


def test_a_thinking_reply_keeps_its_reasoning_and_token_count():
    score, details = confidence(reply("80", reasoning="the booking went through", completion_tokens=412))
    assert score == pytest.approx(0.8)
    assert details["reasoning"] == "the booking went through"
    assert details["completion_tokens"] == 412


@pytest.mark.parametrize("reasoning", [None, ""])
def test_a_reply_without_reasoning_stores_what_it_always_did(reasoning):
    _, details = confidence(reply("80", reasoning=reasoning))
    assert set(details) == {"prompt", "model_response", "parsed_score", "num_actions", "num_errors", "task_reward", "model"}
