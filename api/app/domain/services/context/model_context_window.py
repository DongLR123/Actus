from app.domain.models.context_overflow_config import ContextOverflowConfig

MODEL_CONTEXT_WINDOW_MAP: dict[str, int] = {
    # OpenAI — GPT 系列
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_048_576,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "gpt-5": 1_048_576,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    # OpenAI — o 系列（推理模型）
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # DeepSeek
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 128_000,
    # Anthropic Claude
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    # Qwen（通义千问）
    "qwen-plus": 1_000_000,
    "qwen-turbo": 1_000_000,
    "qwen-max": 32_768,
    "qwen-flash": 1_000_000,
    "qwen3-max": 262_144,
    # Google Gemini
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
}


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    if "/" in name:
        name = name.split("/")[-1]
    return name


def resolve_context_window(model_name: str, config: ContextOverflowConfig) -> int:
    """解析上下文窗口：显式配置 > 模型映射 > 未知模型兜底。"""
    if config.context_window is not None:
        return config.context_window

    normalized = _normalize_model_name(model_name)
    mapped = MODEL_CONTEXT_WINDOW_MAP.get(normalized)
    if mapped is not None:
        return mapped

    for prefix, context_window in MODEL_CONTEXT_WINDOW_MAP.items():
        if normalized.startswith(prefix):
            return context_window

    return config.unknown_model_context_window
