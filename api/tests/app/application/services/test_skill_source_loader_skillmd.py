"""Tests for SkillSourceLoader SKILL.md frontmatter parsing on install."""

from __future__ import annotations

import pytest

from app.application.services.skill_source_loader import SkillSourceLoader
from app.domain.models.skill import SkillSourceType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


STANDARD_SKILL_MD = """\
---
name: demo-skill
slug: demo
description: A demo skill
version: "1.0"
runtime: native
tools:
  - name: demo_run
    description: Run demo
    parameters:
      input:
        type: string
    required:
      - input
    entry:
      command: python3 bundle/demo_run.py
---

# Demo Skill
Run with demo_run tool.
"""


class TestSkillSourceLoaderSkillMdOnly:

    async def test_load_local_with_skillmd(self, tmp_path):
        """Load a skill directory that has SKILL.md — should populate parsed fields."""
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(STANDARD_SKILL_MD)
        bundle_dir = skill_dir / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "demo_run.py").write_text("print('demo')")

        loader = SkillSourceLoader()
        bundle = await loader.load(SkillSourceType.LOCAL, f"local:{skill_dir}")

        assert bundle is not None
        assert bundle.skill_md == STANDARD_SKILL_MD
        assert "bundle/demo_run.py" in bundle.files

    async def test_load_local_extracts_parsed_meta(self, tmp_path):
        """parsed_meta should contain frontmatter metadata fields."""
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(STANDARD_SKILL_MD)
        bundle_dir = skill_dir / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "demo_run.py").write_text("print('demo')")

        loader = SkillSourceLoader()
        bundle = await loader.load(SkillSourceType.LOCAL, f"local:{skill_dir}")

        assert bundle.parsed_meta is not None
        assert bundle.parsed_meta["name"] == "demo-skill"
        assert bundle.parsed_meta["slug"] == "demo"
        assert bundle.parsed_meta["runtime_type"] == "native"

    async def test_load_local_extracts_parsed_manifest(self, tmp_path):
        """parsed_manifest should contain tools from frontmatter."""
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(STANDARD_SKILL_MD)
        bundle_dir = skill_dir / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "demo_run.py").write_text("print('demo')")

        loader = SkillSourceLoader()
        bundle = await loader.load(SkillSourceType.LOCAL, f"local:{skill_dir}")

        assert bundle.parsed_manifest is not None
        tools = bundle.parsed_manifest.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "demo_run"
