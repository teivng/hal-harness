"""Tests that the self-hosted open-weight agent names survive the loaders intact.

Run-dir naming contract: taubench_airline_<agent>_<unit> with units
rep1..5, fault_rep1..5, prompt_naturalistic_var1..5, struct (no timestamps).
"""

import pytest

from reliability_eval.constants import (
    MODEL_METADATA,
    PROVIDER_COLORS,
    PROVIDER_MARKERS,
    PROVIDER_ORDER,
)
from reliability_eval.loaders.agent_names import (
    extract_agent_name,
    get_model_category,
    get_provider,
    strip_agent_prefix,
)
from reliability_eval.loaders.results import detect_run_type

BENCHMARK = "taubench_airline"

OPEN_WEIGHT_AGENTS = {
    "taubench_toolcalling_qwen3_30b_a3b": ("Alibaba", "large"),
    "taubench_toolcalling_qwen3_8b": ("Alibaba", "small"),
    "taubench_toolcalling_qwen3_4b": ("Alibaba", "small"),
    "taubench_toolcalling_gpt_oss_20b": ("OpenAI", "reasoning"),
    "taubench_toolcalling_gpt_oss_120b": ("OpenAI", "reasoning"),
    "taubench_toolcalling_qwen3_5_35b_a3b": ("Alibaba", "large"),
    "taubench_toolcalling_qwen3_5_9b": ("Alibaba", "small"),
    "taubench_toolcalling_gemma4_26b_a4b": ("Google", "large"),
    "taubench_toolcalling_glm47_flash": ("Zhipu", "large"),
    "taubench_toolcalling_olmo31_32b": ("Ai2", "large"),
    "taubench_toolcalling_llama33_70b_fp8": ("Meta", "large"),
}

# unit suffix -> expected run type from detect_run_type
UNITS = {
    "rep1": "baseline",
    "fault_rep3": "fault",
    "prompt_naturalistic_var2": "prompt",
    "struct": "structural",
}


def _minimal_data(unit: str) -> dict:
    """Minimal UPLOAD-json-like dict: no metadata key, config only flags prompt runs."""
    config = {"prompt_sensitivity": True} if unit.startswith("prompt") else {}
    return {"config": config}


@pytest.mark.parametrize("agent", sorted(OPEN_WEIGHT_AGENTS))
@pytest.mark.parametrize("unit", sorted(UNITS))
def test_extract_agent_name_strips_unit_suffix(agent, unit):
    run_dir = f"{BENCHMARK}_{agent}_{unit}"
    assert extract_agent_name(run_dir, BENCHMARK) == agent


@pytest.mark.parametrize("agent", sorted(OPEN_WEIGHT_AGENTS))
@pytest.mark.parametrize("unit,expected", sorted(UNITS.items()))
def test_detect_run_type_from_unit_suffix(agent, unit, expected):
    run_dir = f"{BENCHMARK}_{agent}_{unit}"
    assert detect_run_type(_minimal_data(unit), run_dir) == expected


@pytest.mark.parametrize("agent,expected", sorted(OPEN_WEIGHT_AGENTS.items()))
def test_metadata_and_category(agent, expected):
    provider, category = expected
    assert agent in MODEL_METADATA
    assert get_provider(agent) == provider
    assert get_model_category(agent) == category


@pytest.mark.parametrize("agent", sorted(OPEN_WEIGHT_AGENTS))
def test_display_name_is_mapped(agent):
    # strip_agent_prefix falls back to the raw suffix when no display name exists
    display = strip_agent_prefix(agent)
    assert display != agent.replace("taubench_toolcalling_", "")


def test_every_provider_present_in_every_provider_table():
    # PROVIDER_ORDER is consumed via Series.map(); a missing key yields NaN.
    providers = {meta["provider"] for meta in MODEL_METADATA.values()}
    for table in (PROVIDER_COLORS, PROVIDER_MARKERS, PROVIDER_ORDER):
        assert providers <= set(table)
    assert PROVIDER_ORDER["Unknown"] == max(PROVIDER_ORDER.values())
