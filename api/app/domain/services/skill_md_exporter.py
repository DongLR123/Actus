"""Export Skill meta + manifest to standard Agent Skills SKILL.md format.

Used on the EXPORT path: generates a distributable SKILL.md from the runtime
meta.json + manifest.json structures.
"""

from __future__ import annotations

import re
from typing import Any

import yaml


class SkillMdExporter:
    """Generate standard SKILL.md from meta + manifest dicts."""

    @classmethod
    def export(cls, meta: dict[str, Any], manifest: dict[str, Any]) -> str:
        """Build a complete SKILL.md with YAML frontmatter + markdown body."""
        frontmatter = cls._build_frontmatter(meta, manifest)
        body = cls._extract_body(manifest)
        yaml_str = yaml.dump(
            frontmatter, default_flow_style=False,
            allow_unicode=True, sort_keys=False,
        )
        parts = ["---", yaml_str.rstrip(), "---", ""]
        if body:
            parts.append(body)
        return "\n".join(parts) + "\n"

    @classmethod
    def _build_frontmatter(cls, meta: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        fm: dict[str, Any] = {}

        # Core fields (use `is not None` to preserve valid falsy values like enabled=False)
        for key in ("name", "slug", "description", "version"):
            val = meta.get(key)
            if val is not None:
                fm[key] = val

        # Runtime
        rt = meta.get("runtime_type") or manifest.get("runtime_type")
        if rt:
            fm["runtime"] = str(rt)

        # Tools
        tools = manifest.get("tools")
        if isinstance(tools, list) and tools:
            fm["tools"] = tools

        # Allowed tools
        allowed = (manifest.get("security") or {}).get("allowed_tools")
        if isinstance(allowed, list) and allowed:
            fm["allowed-tools"] = allowed

        # Activation
        activation = manifest.get("activation")
        if isinstance(activation, dict) and activation:
            fm["activation"] = activation

        # Policy (top-level)
        policy = manifest.get("policy")
        if isinstance(policy, dict) and policy:
            fm["policy"] = policy

        # Compatibility (top-level, per Agent Skills spec)
        compat = (meta.get("metadata") or {}).get("compatibility")
        if compat:
            fm["compatibility"] = compat

        # Metadata (author, license, tags, etc.)
        meta_metadata = meta.get("metadata")
        if isinstance(meta_metadata, dict) and meta_metadata:
            # Exclude compatibility from nested metadata (already top-level)
            filtered = {k: v for k, v in meta_metadata.items() if k != "compatibility"}
            if filtered:
                fm["metadata"] = filtered

        return fm

    @classmethod
    def _extract_body(cls, manifest: dict[str, Any]) -> str:
        """Extract markdown body from skill_md, stripping existing frontmatter."""
        raw = str(manifest.get("skill_md") or "").strip()
        if not raw:
            return ""
        # Strip existing frontmatter
        if raw.startswith("---"):
            match = re.match(r"^---\s*\n.*?\n---\s*\n?", raw, re.DOTALL)
            if match:
                return raw[match.end():].strip()
        return raw
