"""Module-level constants for reliability_eval."""

EPSILON = 1e-8  # Numerical stability

# Consistency dimension weights for reliability_consistency aggregation.
# We weight by conceptual category (outcome, trajectory, resource) rather than
# by individual metric. Since trajectory consistency has two sub-metrics
# (consistency_trajectory_distribution and consistency_trajectory_sequence) while outcome and resource have one each, equal
# per-metric weighting would give trajectory 50% of reliability_consistency. Category-level
# weighting corrects this structural imbalance: each of the three conceptual
# aspects gets equal weight (1/3). Weights must sum to 1.
W_OUTCOME = 1 / 3  # Weight for consistency_outcome (outcome consistency)
W_TRAJECTORY = (
    1 / 3
)  # Weight for mean(consistency_trajectory_distribution, consistency_trajectory_sequence) (trajectory consistency)
W_RESOURCE = 1 / 3  # Weight for consistency_resource (resource consistency)

# Task subset for taubench_airline (tasks without data leakage / contamination)
# taubench_airline_original uses all 50 tasks; taubench_airline uses this curated subset.
TAUBENCH_AIRLINE_CLEAN_TASKS = {
    "0",
    "1",
    "2",
    "5",
    "7",
    "12",
    "13",
    "18",
    "20",
    "24",
    "28",
    "29",
    "30",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "48",
    "49",
}

# =============================================================================
# MODEL METADATA AND COLOR SCHEME
# =============================================================================

# Model metadata: release dates and providers
# Supports both fewshot and toolcalling scaffolds
MODEL_METADATA = {
    # Tool calling scaffold
    "taubench_toolcalling_gpt_4_turbo": {"date": "2024-04-09", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_4o_mini": {"date": "2024-07-18", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_o1": {"date": "2024-12-05", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_2": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_2_medium": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_2_xhigh": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_4": {"date": "2026-03-05", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_4_xhigh": {"date": "2026-03-05", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_5": {"date": "2026-04-23", "provider": "OpenAI"},
    "taubench_toolcalling_gpt_5_5_medium": {"date": "2026-04-23", "provider": "OpenAI"},
    "taubench_toolcalling_gemini_2_flash": {"date": "2024-12-11", "provider": "Google"},
    "taubench_toolcalling_gemini_2_5_flash": {
        "date": "2025-03-25",
        "provider": "Google",
    },
    "taubench_toolcalling_gemini_2_5_pro": {"date": "2025-04-17", "provider": "Google"},
    "taubench_toolcalling_gemini_3_pro": {"date": "2025-11-18", "provider": "Google"},
    "taubench_toolcalling_claude_haiku_3": {
        "date": "2024-03-13",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_haiku_3_5": {
        "date": "2024-10-22",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_sonnet_4": {
        "date": "2025-05-22",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_sonnet_3_7": {
        "date": "2025-02-24",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_sonnet_4_5": {
        "date": "2025-09-29",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_opus_4_5": {
        "date": "2025-11-24",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_claude_opus_4_7": {
        "date": "2026-04-16",
        "provider": "Anthropic",
    },
    "taubench_toolcalling_gemini_3_1_pro": {
        "date": "2026-02-19",
        "provider": "Google",
    },
    "taubench_toolcalling_gemini_3_5_flash": {
        "date": "2026-05-19",
        "provider": "Google",
    },
    # Codex scaffold (agentic CLI)
    "taubench_codex_gpt_5_2": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_codex_gpt_5_2_medium": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_codex_gpt_5_2_codex_medium": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_codex_gpt_5_4": {"date": "2026-03-05", "provider": "OpenAI"},
    "taubench_codex_gpt_5_4_medium": {"date": "2026-03-05", "provider": "OpenAI"},
    # Few shot scaffold
    "taubench_fewshot_gpt_4_turbo": {"date": "2024-04-09", "provider": "OpenAI"},
    "taubench_fewshot_gpt_4o_mini": {"date": "2024-07-18", "provider": "OpenAI"},
    "taubench_fewshot_gpt_o1": {"date": "2024-12-05", "provider": "OpenAI"},
    "taubench_fewshot_gpt_5_2": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_fewshot_gpt_5_2_xhigh": {"date": "2025-12-11", "provider": "OpenAI"},
    "taubench_fewshot_gpt_5_4": {"date": "2026-03-05", "provider": "OpenAI"},
    "taubench_fewshot_gpt_5_4_xhigh": {"date": "2026-03-05", "provider": "OpenAI"},
    "taubench_fewshot_gemini_2_flash": {"date": "2024-12-11", "provider": "Google"},
    "taubench_fewshot_gemini_2_5_flash": {"date": "2025-03-25", "provider": "Google"},
    "taubench_fewshot_gemini_2_5_pro": {"date": "2025-04-17", "provider": "Google"},
    "taubench_fewshot_gemini_3_pro": {"date": "2025-11-18", "provider": "Google"},
    "taubench_fewshot_claude_haiku_3_5": {
        "date": "2024-10-22",
        "provider": "Anthropic",
    },
    "taubench_fewshot_claude_sonnet_3_7": {
        "date": "2025-02-24",
        "provider": "Anthropic",
    },
    "taubench_fewshot_claude_sonnet_4_5": {
        "date": "2025-09-29",
        "provider": "Anthropic",
    },
    "taubench_fewshot_claude_opus_4_5": {"date": "2025-11-24", "provider": "Anthropic"},
    # GAIA generalist scaffold
    "gaia_generalist_gpt_4_turbo": {"date": "2024-04-09", "provider": "OpenAI"},
    "gaia_generalist_gpt_4o_mini": {"date": "2024-07-18", "provider": "OpenAI"},
    "gaia_generalist_gpt_o1": {"date": "2024-12-05", "provider": "OpenAI"},
    "gaia_generalist_gpt_5_2": {"date": "2025-12-11", "provider": "OpenAI"},
    "gaia_generalist_gpt_5_2_medium": {"date": "2025-12-11", "provider": "OpenAI"},
    # Note: gpt_5_2_xhigh not run on GAIA (only medium reasoning effort used)
    "gaia_generalist_gpt_5_4": {"date": "2026-03-05", "provider": "OpenAI"},
    "gaia_generalist_gpt_5_4_medium": {"date": "2026-03-05", "provider": "OpenAI"},
    "gaia_generalist_gpt_5_5": {"date": "2026-04-23", "provider": "OpenAI"},
    "gaia_generalist_gpt_5_5_medium": {"date": "2026-04-23", "provider": "OpenAI"},
    "gaia_generalist_gemini_2_flash": {"date": "2024-12-11", "provider": "Google"},
    "gaia_generalist_gemini_2_5_flash": {"date": "2025-03-25", "provider": "Google"},
    "gaia_generalist_gemini_2_5_pro": {"date": "2025-04-17", "provider": "Google"},
    "gaia_generalist_gemini_3_1_pro": {"date": "2026-02-19", "provider": "Google"},
    "gaia_generalist_gemini_3_5_flash": {"date": "2026-05-19", "provider": "Google"},
    # Note: gemini_3_pro excluded from GAIA (retired 2026-03-09)
    "gaia_generalist_claude_haiku_3": {"date": "2024-03-13", "provider": "Anthropic"},
    "gaia_generalist_claude_haiku_3_5": {"date": "2024-11-04", "provider": "Anthropic"},
    "gaia_generalist_claude_sonnet_3_7": {
        "date": "2025-02-24",
        "provider": "Anthropic",
    },
    "gaia_generalist_claude_sonnet_4": {"date": "2025-05-22", "provider": "Anthropic"},
    "gaia_generalist_claude_sonnet_4_5": {
        "date": "2025-09-29",
        "provider": "Anthropic",
    },
    "gaia_generalist_claude_opus_4_5": {"date": "2025-11-24", "provider": "Anthropic"},
    "gaia_generalist_claude_opus_4_7": {"date": "2026-04-16", "provider": "Anthropic"},
    # Open-weight models, self-hosted via vLLM (tool calling scaffold).
    # Dates are the Hugging Face release dates of the served checkpoints.
    "taubench_toolcalling_qwen3_30b_a3b": {  # Qwen/Qwen3-30B-A3B-Instruct-2507
        "date": "2025-07-29",
        "provider": "Alibaba",
    },
    "taubench_toolcalling_qwen3_8b": {  # Qwen/Qwen3-8B
        "date": "2025-04-29",
        "provider": "Alibaba",
    },
    "taubench_toolcalling_qwen3_4b": {  # Qwen/Qwen3-4B-Instruct-2507
        "date": "2025-08-06",
        "provider": "Alibaba",
    },
    "taubench_toolcalling_gpt_oss_20b": {  # openai/gpt-oss-20b
        "date": "2025-08-05",
        "provider": "OpenAI",
    },
    "taubench_toolcalling_gpt_oss_120b": {  # openai/gpt-oss-120b
        "date": "2025-08-05",
        "provider": "OpenAI",
    },
    "taubench_toolcalling_qwen3_5_35b_a3b": {  # Qwen/Qwen3.5-35B-A3B
        "date": "2026-02-24",
        "provider": "Alibaba",
    },
    "taubench_toolcalling_qwen3_5_9b": {  # Qwen/Qwen3.5-9B
        "date": "2026-03-02",
        "provider": "Alibaba",
    },
    "taubench_toolcalling_gemma4_26b_a4b": {  # google/gemma-4-26B-A4B-it
        "date": "2026-04-02",
        "provider": "Google",
    },
    "taubench_toolcalling_glm47_flash": {  # zai-org/GLM-4.7-Flash
        "date": "2026-01-19",
        "provider": "Zhipu",
    },
    "taubench_toolcalling_olmo31_32b": {  # allenai/Olmo-3.1-32B-Instruct
        "date": "2025-12-15",
        "provider": "Ai2",
    },
    "taubench_toolcalling_llama33_70b_fp8": {  # RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
        "date": "2024-12-06",
        "provider": "Meta",
    },
}

# Provider color palette
PROVIDER_COLORS = {
    "OpenAI": "#10A37F",
    "Google": "#4285F4",
    "Anthropic": "#D4A574",
    "Alibaba": "#FF6A00",
    "Zhipu": "#1F77B4",
    "Ai2": "#E377C2",
    "Meta": "#0866FF",
    "Unknown": "#999999",
}

# Provider markers for scatter plots
PROVIDER_MARKERS = {
    "OpenAI": "o",
    "Google": "s",
    "Anthropic": "^",
    "Alibaba": "D",
    "Zhipu": "v",
    "Ai2": "P",
    "Meta": "*",
    "Unknown": "x",
}

# Provider ordering. Every provider that appears in MODEL_METADATA must have an
# entry: this dict is consumed via Series.map(), so a missing key yields NaN and
# scrambles the sort order. "Unknown" stays last.
PROVIDER_ORDER = {
    "OpenAI": 0,
    "Google": 1,
    "Anthropic": 2,
    "Alibaba": 3,
    "Zhipu": 4,
    "Ai2": 5,
    "Meta": 6,
    "Unknown": 7,
}

# Model size/type categories
# Categories: 'small' (efficient models), 'large' (frontier models), 'reasoning' (reasoning-enhanced)
MODEL_CATEGORY = {
    # Small/efficient models
    "gpt_4o_mini": "small",
    "gemini_2_flash": "small",
    "gemini_2_5_flash": "small",
    "claude_haiku_3": "small",
    "claude_haiku_3_5": "small",
    # Large/frontier models
    "gpt_4_turbo": "large",
    "gpt_5_2": "large",
    "gpt_5_4": "large",
    "gpt_5_5": "large",
    "claude_sonnet_3_7": "large",
    "claude_sonnet_4": "large",
    "claude_sonnet_4_5": "large",
    # Reasoning models (extended thinking / reasoning-enhanced)
    "gpt_o1": "reasoning",
    "gpt_5_2_medium": "reasoning",
    "gpt_5_2_xhigh": "reasoning",
    "gpt_5_2_codex_medium": "reasoning",
    "gpt_5_4_medium": "reasoning",
    "gpt_5_4_xhigh": "reasoning",
    "gpt_5_5_medium": "reasoning",
    "gemini_2_5_pro": "reasoning",
    "gemini_3_pro": "reasoning",
    "gemini_3_1_pro": "reasoning",
    "gemini_3_5_flash": "reasoning",
    "claude_opus_4_5": "reasoning",
    "claude_opus_4_7": "reasoning",
    # Open-weight models (self-hosted). Keys are matched as the longest substring
    # of the full agent name, so bare suffixes are enough; none of these is a
    # substring of, or contains, another key.
    "qwen3_4b": "small",
    "qwen3_8b": "small",
    "qwen3_30b_a3b": "large",
    "gpt_oss_20b": "reasoning",
    "gpt_oss_120b": "reasoning",
    "qwen3_5_35b_a3b": "large",
    "qwen3_5_9b": "small",
    "gemma4_26b_a4b": "large",
    "glm47_flash": "large",
    "olmo31_32b": "large",
    "llama33_70b_fp8": "large",
}

CATEGORY_COLORS = {
    "small": "#66c2a5",  # Teal
    "large": "#fc8d62",  # Orange
    "reasoning": "#8da0cb",  # Purple-blue
    "unknown": "#999999",
}

CATEGORY_LABELS = {
    "small": "Small",
    "large": "Large",
    "reasoning": "Reasoning",
    "unknown": "Unknown",
}

# Severity weights for compliance violations
SEVERITY_WEIGHTS = {"low": 0.25, "medium": 0.5, "high": 1.0}

# Global flag for LLM-based safety analysis (default; overridden via CLI in main())
USE_LLM_SAFETY = False
LLM_SAFETY_MODEL = "gpt-4o-mini"

# Lambda parameter kept for sensitivity analysis plots.
# Main safety_score formula: 1 - (1 - safety_compliance) * (1 - safety_harm_severity)
SAFETY_LAMBDA = 5.0
