"""Parse SKILL.md with YAML frontmatter into meta + manifest structures.

Used on the INSTALL path: when a skill is distributed as a single SKILL.md file
(Agent Skills standard format), the parser extracts structured data to generate
the runtime meta.json + manifest.json.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Frontmatter fields that map directly to meta.json
_META_DIRECT_FIELDS = {"name", "slug", "description", "version", "enabled"}

# Frontmatter fields that go into meta.metadata dict
_META_METADATA_FIELDS = {"author", "license", "compatibility"}


@dataclass
class SkillMdParseResult:
    """Result of parsing a SKILL.md file."""
    meta: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    body: str = ""


class SkillMdParser:
    """Parse SKILL.md frontmatter YAML + body markdown."""

    @classmethod
    def parse(cls, skill_md: str) -> SkillMdParseResult:
        """Parse SKILL.md content into meta + manifest + body.

        Gracefully degrades on malformed input:
        - No frontmatter -> empty meta/manifest, full text as body
        - Invalid YAML -> empty meta/manifest, body after closing ---
        """
        if not skill_md or not skill_md.strip():
            return SkillMdParseResult(meta={"name": ""}, manifest={})

        match = _FRONTMATTER_RE.match(skill_md)
        if not match:
            return SkillMdParseResult(
                meta={"name": ""},
                manifest={"skill_md": skill_md},
                body=skill_md.strip(),
            )

        yaml_text = match.group(1)
        body = skill_md[match.end():].strip()

        try:
            parsed = yaml.safe_load(yaml_text)
            if not isinstance(parsed, dict):
                parsed = {}
        except yaml.YAMLError:
            logger.warning("SKILL.md frontmatter YAML 解析失败，降级为纯 Markdown")
            return SkillMdParseResult(
                meta={"name": ""},
                manifest={"skill_md": skill_md},
                body=body,
            )

        meta = cls._extract_meta(parsed)
        manifest = cls._extract_manifest(parsed, skill_md)

        return SkillMdParseResult(meta=meta, manifest=manifest, body=body)

    @classmethod
    def _extract_meta(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        """Extract meta.json-compatible fields from parsed frontmatter."""
        meta: dict[str, Any] = {}

        # Direct fields
        for key in _META_DIRECT_FIELDS:
            if key in parsed:
                meta[key] = parsed[key]

        # runtime -> runtime_type
        if "runtime" in parsed:
            meta["runtime_type"] = str(parsed["runtime"])
        elif "runtime_type" in parsed:
            meta["runtime_type"] = str(parsed["runtime_type"])

        # Metadata sub-dict (author, license, etc.)
        metadata: dict[str, Any] = {}
        for key in _META_METADATA_FIELDS:
            if key in parsed:
                metadata[key] = parsed[key]
        # Merge frontmatter metadata section
        fm_metadata = parsed.get("metadata")
        if isinstance(fm_metadata, dict):
            metadata.update(fm_metadata)
        if metadata:
            meta["metadata"] = metadata

        # Ensure name exists
        meta.setdefault("name", "")

        return meta

    @classmethod
    def _extract_manifest(cls, parsed: dict[str, Any], raw_skill_md: str) -> dict[str, Any]:
        """Extract manifest.json-compatible fields from parsed frontmatter."""
        manifest: dict[str, Any] = {}

        # Tools
        tools = parsed.get("tools")
        if isinstance(tools, list):
            manifest["tools"] = tools
        else:
            manifest["tools"] = []

        # Runtime type
        rt = parsed.get("runtime") or parsed.get("runtime_type")
        if rt:
            manifest["runtime_type"] = str(rt)

        # Activation, policy
        for key in ("activation", "policy"):
            if key in parsed and isinstance(parsed[key], dict):
                manifest[key] = parsed[key]

        # Security: allowed-tools -> security.allowed_tools
        allowed = parsed.get("allowed-tools") or parsed.get("allowed_tools")
        if isinstance(allowed, list):
            manifest.setdefault("security", {})["allowed_tools"] = allowed

        # Store full SKILL.md as skill_md (for get_skill_guide)
        manifest["skill_md"] = raw_skill_md

        return manifest
