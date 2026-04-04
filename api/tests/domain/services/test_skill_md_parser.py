"""Tests for SkillMdParser — SKILL.md frontmatter to meta + manifest."""

from __future__ import annotations

import pytest

from app.domain.services.skill_md_parser import SkillMdParser

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


SAMPLE_SKILL_MD = """\
---
name: xlsx-processor
slug: xlsx
description: Process Excel files
version: "1.0"
runtime: native
author: actus
license: MIT
compatibility: Python 3.12+

tools:
  - name: xlsx_read
    description: Read Excel file
    parameters:
      filepath:
        type: string
        description: File path
    required:
      - filepath
    entry:
      command: python3 scripts/read.py
      exec_dir: .
    policy:
      model_invocable: true
      risk_level: low

allowed-tools:
  - shell_execute
  - file_read

metadata:
  tags:
    - spreadsheet
    - data
  category: data-processing
---

# XLSX Processor

## Usage
Use openpyxl to read Excel files.
"""


class TestSkillMdParserBasic:

    def test_parse_returns_meta_and_manifest(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        assert result is not None
        assert result.meta is not None
        assert result.manifest is not None
        assert result.body is not None

    def test_meta_fields(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        assert result.meta["name"] == "xlsx-processor"
        assert result.meta["slug"] == "xlsx"
        assert result.meta["description"] == "Process Excel files"
        assert result.meta["version"] == "1.0"
        assert result.meta["runtime_type"] == "native"

    def test_meta_metadata(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        metadata = result.meta.get("metadata", {})
        assert metadata.get("author") == "actus"
        assert metadata.get("license") == "MIT"
        assert metadata.get("compatibility") == "Python 3.12+"
        assert metadata.get("tags") == ["spreadsheet", "data"]
        assert metadata.get("category") == "data-processing"

    def test_manifest_tools(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        tools = result.manifest.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["name"] == "xlsx_read"
        assert tools[0]["parameters"]["filepath"]["type"] == "string"
        assert tools[0]["required"] == ["filepath"]
        assert tools[0]["entry"]["command"] == "python3 scripts/read.py"

    def test_manifest_security(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        security = result.manifest.get("security", {})
        assert security.get("allowed_tools") == ["shell_execute", "file_read"]

    def test_manifest_policy(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        tools = result.manifest.get("tools", [])
        assert tools[0].get("policy", {}).get("risk_level") == "low"

    def test_body_content(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        assert "# XLSX Processor" in result.body
        assert "openpyxl" in result.body

    def test_skill_md_stored_in_manifest(self):
        result = SkillMdParser.parse(SAMPLE_SKILL_MD)
        assert "skill_md" in result.manifest
        assert "# XLSX Processor" in result.manifest["skill_md"]


class TestSkillMdParserEdgeCases:

    def test_no_frontmatter(self):
        """Pure markdown without frontmatter should degrade gracefully."""
        result = SkillMdParser.parse("# Just a readme\n\nSome content.")
        assert result is not None
        assert result.meta.get("name") == ""
        assert result.body == "# Just a readme\n\nSome content."

    def test_malformed_yaml(self):
        """Invalid YAML should degrade to no-frontmatter mode."""
        md = "---\nname: [invalid: yaml: {\n---\n# Content"
        result = SkillMdParser.parse(md)
        assert result is not None
        assert result.body == "# Content"

    def test_empty_input(self):
        result = SkillMdParser.parse("")
        assert result is not None
        assert result.meta.get("name") == ""

    def test_missing_optional_fields(self):
        """Only name and description are truly needed."""
        md = "---\nname: minimal\ndescription: A minimal skill\n---\n# Minimal"
        result = SkillMdParser.parse(md)
        assert result.meta["name"] == "minimal"
        assert result.meta["description"] == "A minimal skill"
        assert result.manifest.get("tools", []) == []

    def test_runtime_field_mapping(self):
        """'runtime' in frontmatter maps to 'runtime_type' in meta."""
        md = "---\nname: test\nruntime: mcp\n---\n"
        result = SkillMdParser.parse(md)
        assert result.meta["runtime_type"] == "mcp"
