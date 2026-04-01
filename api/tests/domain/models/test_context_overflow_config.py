from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.app_config import LLMConfig


def test_default_target_ratio():
    cfg = ContextOverflowConfig()
    assert cfg.target_ratio == 0.65


def test_default_summary_max_chars():
    cfg = ContextOverflowConfig()
    assert cfg.summary_max_chars == 16_000


def test_from_llm_config_uses_defaults_for_new_fields():
    """target_ratio and summary_max_chars use ContextOverflowConfig defaults,
    not LLMConfig fields (they don't exist on LLMConfig)."""
    llm_cfg = LLMConfig(model_name="gpt-4o")
    overflow = ContextOverflowConfig.from_llm_config(llm_cfg)
    assert overflow.target_ratio == 0.65
    assert overflow.summary_max_chars == 16_000
