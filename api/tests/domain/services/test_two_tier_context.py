"""Tests for two-tier _build_skill_context_prompt (Tier 2 full guide / Tier 1 lightweight card)."""

from unittest.mock import MagicMock

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.agent_task_runner import AgentTaskRunner


def _make_runner():
    """Create a partially-mocked runner with real context-building methods."""
    runner = MagicMock(spec=AgentTaskRunner)
    runner._build_skill_context_prompt = AgentTaskRunner._build_skill_context_prompt.__get__(runner)
    runner._get_skill_guide_body = AgentTaskRunner._get_skill_guide_body.__get__(runner)
    runner._strip_skill_frontmatter = AgentTaskRunner._strip_skill_frontmatter.__get__(runner)
    runner._tier2_preloaded_skill_ids = set()
    return runner


def _make_skill(
    skill_id: str,
    name: str,
    slug: str,
    description: str = "",
    context_blob: str = "",
    skill_md: str = "",
    tools: list | None = None,
) -> Skill:
    manifest: dict = {}
    if context_blob:
        manifest["context_blob"] = context_blob
    if skill_md:
        manifest["skill_md"] = skill_md
    if tools is not None:
        manifest["tools"] = tools
    return Skill(
        id=skill_id,
        slug=slug,
        name=name,
        description=description,
        source_type=SkillSourceType.LOCAL,
        source_ref="test",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest=manifest,
    )


class TestTwoTierContextWithScores:
    """With explicit scores, only skills above threshold get Tier 2."""

    def test_high_scores_get_tier2_low_scores_get_tier1(self):
        runner = _make_runner()
        skills = [
            _make_skill("s1", "Alpha", "alpha", description="Alpha desc",
                        context_blob="Alpha full guide content here.",
                        tools=[{"name": "alpha_run"}, {"name": "alpha_check"}]),
            _make_skill("s2", "Beta", "beta", description="Beta desc",
                        context_blob="Beta full guide content here.",
                        tools=[{"name": "beta_exec"}]),
            _make_skill("s3", "Gamma", "gamma", description="Gamma desc",
                        context_blob="Gamma full guide content here.",
                        tools=[{"name": "gamma_tool"}]),
        ]
        scores = [0.8, 0.3, 0.1]
        result = runner._build_skill_context_prompt(skills, scores=scores)

        # s1 (score=0.8 >= 0.5) -> Tier 2 full guide
        assert "Alpha full guide content here." in result
        # s2 (score=0.3 < 0.5) -> Tier 1 lightweight card
        assert "Beta desc" in result
        assert "Tools: beta_exec" in result
        assert "Beta full guide content here." not in result
        # s3 (score=0.1 < 0.5) -> Tier 1 lightweight card
        assert "Gamma desc" in result
        assert "Tools: gamma_tool" in result
        assert "Gamma full guide content here." not in result

        # Only s1 should be in tier2 preloaded set
        assert runner._tier2_preloaded_skill_ids == {"s1"}

    def test_multiple_high_scores_capped_at_tier2_max(self):
        """At most TIER2_MAX_COUNT=2 skills get Tier 2, even if more score above threshold."""
        runner = _make_runner()
        skills = [
            _make_skill("s1", "A", "a", context_blob="Guide A", tools=[{"name": "t1"}]),
            _make_skill("s2", "B", "b", context_blob="Guide B", tools=[{"name": "t2"}]),
            _make_skill("s3", "C", "c", description="C desc", context_blob="Guide C", tools=[{"name": "t3"}]),
        ]
        scores = [0.9, 0.7, 0.6]
        result = runner._build_skill_context_prompt(skills, scores=scores)

        # First two get Tier 2
        assert "Guide A" in result
        assert "Guide B" in result
        # Third is above threshold but capped -> Tier 1
        assert "Guide C" not in result
        assert "C desc" in result
        assert "Tools: t3" in result

        assert runner._tier2_preloaded_skill_ids == {"s1", "s2"}


class TestTwoTierContextNoScores:
    """Without scores, top-1 gets Tier 2, rest get Tier 1."""

    def test_no_scores_top1_gets_tier2(self):
        runner = _make_runner()
        skills = [
            _make_skill("s1", "First", "first", description="First desc",
                        context_blob="First full guide.",
                        tools=[{"name": "first_tool"}]),
            _make_skill("s2", "Second", "second", description="Second desc",
                        context_blob="Second full guide.",
                        tools=[{"name": "second_tool"}]),
        ]
        result = runner._build_skill_context_prompt(skills, scores=None)

        # Top-1 -> Tier 2
        assert "First full guide." in result
        # Second -> Tier 1
        assert "Second desc" in result
        assert "Tools: second_tool" in result
        assert "Second full guide." not in result

        assert runner._tier2_preloaded_skill_ids == {"s1"}


class TestTier1Card:
    """Tier 1 card should contain skill name, description, and tool names."""

    def test_tier1_card_format(self):
        runner = _make_runner()
        skills = [
            _make_skill("s1", "Primary", "primary", context_blob="Primary guide."),
            _make_skill("s2", "Secondary", "secondary",
                        description="A helper skill",
                        tools=[{"name": "helper_run"}, {"name": "helper_check"}]),
        ]
        # No scores -> s1 Tier 2, s2 Tier 1
        result = runner._build_skill_context_prompt(skills)

        # s2 Tier 1 card
        assert "### Secondary (secondary)" in result
        assert "A helper skill" in result
        assert "Tools: helper_run, helper_check" in result

    def test_tier1_card_no_tools(self):
        runner = _make_runner()
        skills = [
            _make_skill("s1", "Main", "main", context_blob="Main guide."),
            _make_skill("s2", "Empty", "empty", description="Empty skill"),
        ]
        result = runner._build_skill_context_prompt(skills)

        assert "Tools: (none)" in result

    def test_tier1_card_no_description(self):
        runner = _make_runner()
        skills = [
            _make_skill("s1", "Main", "main", context_blob="Main guide."),
            _make_skill("s2", "Bare", "bare", tools=[{"name": "bare_tool"}]),
        ]
        result = runner._build_skill_context_prompt(skills)

        assert "No description." in result
        assert "Tools: bare_tool" in result


class TestEmptySkills:
    """Edge case: no skills produces empty string."""

    def test_empty_list(self):
        runner = _make_runner()
        result = runner._build_skill_context_prompt([], scores=[])
        assert result == ""

    def test_none_skills(self):
        runner = _make_runner()
        result = runner._build_skill_context_prompt([])
        assert result == ""
