from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.services.context.model_context_window import resolve_context_window


def test_resolve_context_window_prefers_explicit_config() -> None:
    config = ContextOverflowConfig(
        context_window=200000,
        unknown_model_context_window=32768,
    )

    assert resolve_context_window(model_name="gpt-4o", config=config) == 200000


def test_resolve_context_window_falls_back_to_model_map() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=32768,
    )

    assert resolve_context_window(model_name="gpt-4o", config=config) == 128000


def test_resolve_context_window_uses_unknown_model_fallback() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=65536,
    )

    assert (
        resolve_context_window(model_name="unknown-provider-model", config=config)
        == 65536
    )


def test_resolve_context_window_claude_model() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=32768,
    )
    assert resolve_context_window(model_name="claude-sonnet-4", config=config) == 200000


def test_resolve_context_window_prefix_match() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=32768,
    )
    assert resolve_context_window(model_name="claude-3-5-sonnet-20241022", config=config) == 200000


def test_resolve_context_window_with_provider_prefix() -> None:
    config = ContextOverflowConfig(
        context_window=None,
        unknown_model_context_window=32768,
    )
    assert resolve_context_window(model_name="openai/gpt-4o", config=config) == 128000


def test_resolve_context_window_deepseek_v3() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="deepseek-chat", config=config) == 128_000
    assert resolve_context_window(model_name="deepseek-reasoner", config=config) == 128_000


def test_resolve_context_window_o_series() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="o3", config=config) == 200_000
    assert resolve_context_window(model_name="o4-mini", config=config) == 200_000


def test_resolve_context_window_gpt5() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="gpt-5.4", config=config) == 1_050_000
    assert resolve_context_window(model_name="gpt-5.4-mini", config=config) == 400_000


def test_resolve_context_window_claude_4_6() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="claude-opus-4-6", config=config) == 1_000_000
    assert resolve_context_window(model_name="claude-sonnet-4-6", config=config) == 1_000_000


def test_resolve_context_window_qwen() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="qwen-plus", config=config) == 1_000_000
    assert resolve_context_window(model_name="qwen-max", config=config) == 32_768


def test_resolve_context_window_gemini() -> None:
    config = ContextOverflowConfig(context_window=None, unknown_model_context_window=32768)
    assert resolve_context_window(model_name="gemini-2.5-pro", config=config) == 1_000_000
    assert resolve_context_window(model_name="gemini-2.5-flash", config=config) == 1_000_000


def test_overflow_config_from_llm_config_includes_model_name() -> None:
    from app.domain.models.app_config import LLMConfig

    llm_config = LLMConfig(model_name="gpt-4o", api_key="test")
    overflow = ContextOverflowConfig.from_llm_config(llm_config)
    assert overflow.model_name == "gpt-4o"
