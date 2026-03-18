"""Tests for _extract_python_code utility."""

import pytest

from app.application.services.skill_creator_service import _extract_python_code


class TestExtractPythonCode:
    def test_extract_from_python_code_block(self) -> None:
        text = '一些说明\n```python\nimport sys\nprint("hello")\n```\n后续文字'
        result = _extract_python_code(text)
        assert result == 'import sys\nprint("hello")'

    def test_extract_from_generic_code_block(self) -> None:
        text = '```\nimport json\nprint(json.dumps({}))\n```'
        result = _extract_python_code(text)
        assert result == 'import json\nprint(json.dumps({}))'

    def test_python_block_takes_priority_over_generic(self) -> None:
        text = '```\ngeneric\n```\n```python\nspecific\n```'
        result = _extract_python_code(text)
        assert result == 'specific'

    def test_raw_python_code_no_fences(self) -> None:
        text = 'import sys\nimport json\n\ndef main():\n    pass'
        result = _extract_python_code(text)
        assert 'import sys' in result
        assert 'def main():' in result

    def test_strips_leading_explanation_lines(self) -> None:
        text = 'Here is the code:\n\nimport sys\nprint("ok")'
        result = _extract_python_code(text)
        assert result.startswith('import sys')

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            _extract_python_code('')

    def test_no_python_content_raises(self) -> None:
        with pytest.raises(ValueError):
            _extract_python_code('This is just plain English text with no code.')
