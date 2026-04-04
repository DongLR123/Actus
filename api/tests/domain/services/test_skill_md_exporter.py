"""Tests for SkillMdExporter — meta + manifest to standard SKILL.md."""

from __future__ import annotations

import pytest

from app.domain.services.skill_md_exporter import SkillMdExporter
from app.domain.services.skill_md_parser import SkillMdParser

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _sample_meta():
    return {
        "name": "xlsx-processor",
        "slug": "xlsx",
        "description": "Process Excel files",
        "version": "1.0",
        "runtime_type": "native",
        "metadata": {
            "author": "actus",
            "license": "MIT",
            "tags": ["spreadsheet"],
        },
    }


def _sample_manifest():
    return {
        "tools": [
            {
                "name": "xlsx_read",
                "description": "Read Excel file",
                "parameters": {
                    "filepath": {"type": "string", "description": "File path"},
                },
                "required": ["filepath"],
                "entry": {"command": "python3 scripts/read.py", "exec_dir": "."},
                "policy": {"model_invocable": True, "risk_level": "low"},
            },
        ],
        "security": {"allowed_tools": ["shell_execute", "file_read"]},
        "skill_md": "# Old content\nThis is the original guide.",
    }


class TestSkillMdExporterBasic:

    def test_export_produces_valid_frontmatter(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert md.startswith("---\n")
        assert "\n---\n" in md

    def test_export_contains_name_and_description(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert "xlsx-processor" in md
        assert "Process Excel files" in md

    def test_export_contains_tools(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert "xlsx_read" in md

    def test_export_contains_body(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert "# Old content" in md

    def test_export_contains_allowed_tools(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert "shell_execute" in md

    def test_export_contains_metadata(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        assert "actus" in md
        assert "spreadsheet" in md


class TestSkillMdExporterRoundTrip:

    def test_round_trip_preserves_core_fields(self):
        """Export -> Parse should preserve core fields."""
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        result = SkillMdParser.parse(md)
        assert result.meta["name"] == "xlsx-processor"
        assert result.meta["slug"] == "xlsx"
        assert result.meta["description"] == "Process Excel files"
        assert result.meta["version"] == "1.0"
        assert result.meta["runtime_type"] == "native"

    def test_round_trip_preserves_tools(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        result = SkillMdParser.parse(md)
        tools = result.manifest.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "xlsx_read"
        assert tools[0]["required"] == ["filepath"]

    def test_round_trip_preserves_allowed_tools(self):
        md = SkillMdExporter.export(_sample_meta(), _sample_manifest())
        result = SkillMdParser.parse(md)
        allowed = result.manifest.get("security", {}).get("allowed_tools", [])
        assert "shell_execute" in allowed

    def test_export_empty_manifest(self):
        md = SkillMdExporter.export({"name": "empty"}, {})
        assert "---" in md
        result = SkillMdParser.parse(md)
        assert result.meta["name"] == "empty"
