"""Tests for SkillGuideInjector — on-demand Tier 2 guide injection."""
from app.domain.models.skill import Skill, SkillSourceType, SkillRuntimeType

def _build_skill(skill_id, name, context_blob="", tools=None):
    manifest = {}
    if context_blob:
        manifest["context_blob"] = context_blob
    if tools:
        manifest["tools"] = tools
    return Skill(
        id=skill_id, slug=name.lower(), name=name, description="desc",
        source_type=SkillSourceType.LOCAL, source_ref="t",
        runtime_type=SkillRuntimeType.NATIVE, manifest=manifest,
    )

class TestSkillGuideInjector:
    def test_returns_guide_on_first_call(self):
        from app.domain.services.agent_task_runner import SkillGuideInjector
        skill = _build_skill("s1", "Notion", context_blob="Full Notion guide here", tools=[{"name": "notion_query"}])
        injector = SkillGuideInjector([skill], preloaded_ids=set())
        guide = injector("notion_query")
        assert guide is not None
        assert "Full Notion guide" in guide

    def test_returns_none_on_second_call(self):
        from app.domain.services.agent_task_runner import SkillGuideInjector
        skill = _build_skill("s1", "Notion", context_blob="guide", tools=[{"name": "notion_query"}])
        injector = SkillGuideInjector([skill], preloaded_ids=set())
        injector("notion_query")
        assert injector("notion_query") is None

    def test_returns_none_for_preloaded_skill(self):
        from app.domain.services.agent_task_runner import SkillGuideInjector
        skill = _build_skill("s1", "Notion", context_blob="guide", tools=[{"name": "notion_query"}])
        injector = SkillGuideInjector([skill], preloaded_ids={"s1"})
        assert injector("notion_query") is None

    def test_returns_none_for_unknown_tool(self):
        from app.domain.services.agent_task_runner import SkillGuideInjector
        injector = SkillGuideInjector([], preloaded_ids=set())
        assert injector("nonexistent") is None

    def test_truncates_long_guide(self):
        from app.domain.services.agent_task_runner import SkillGuideInjector
        long_blob = "x" * 5000
        skill = _build_skill("s1", "X", context_blob=long_blob, tools=[{"name": "x_tool"}])
        injector = SkillGuideInjector([skill], preloaded_ids=set())
        guide = injector("x_tool")
        assert guide is not None
        assert len(guide) <= 1200
