from app.domain.models.memory import Memory


class TestMemoryCompact:
    def test_compact_browser_view_keeps_summary_when_enabled(self) -> None:
        """compact(keep_summary=True) 应保留 browser 工具的摘要而非 (removed)"""
        memory = Memory(
            messages=[
                {"role": "system", "content": "system prompt"},
                {
                    "role": "tool",
                    "function_name": "browser_view",
                    "content": "<html><head><title>测试页面</title></head><body><p>这是页面正文内容，包含一些有用的信息。</p></body></html>",
                },
            ]
        )

        memory.compact(keep_summary=True)

        tool_msg = memory.messages[1]
        assert tool_msg["content"] != "(removed)"
        assert "测试页面" in tool_msg["content"]
        assert "[已执行]" in tool_msg["content"]
        assert "<html>" not in tool_msg["content"]

    def test_compact_browser_navigate_keeps_summary_when_enabled(self) -> None:
        """compact(keep_summary=True) 对 browser_navigate 同样保留摘要"""
        memory = Memory(
            messages=[
                {
                    "role": "tool",
                    "function_name": "browser_navigate",
                    "content": "<html><head><title>导航页</title></head><body>纯文本内容在这里</body></html>",
                },
            ]
        )

        memory.compact(keep_summary=True)

        tool_msg = memory.messages[0]
        assert "导航页" in tool_msg["content"]
        assert "<html>" not in tool_msg["content"]

    def test_compact_removes_content_when_keep_summary_false(self) -> None:
        """compact(keep_summary=False) 保持原有行为，直接 (removed)"""
        memory = Memory(
            messages=[
                {
                    "role": "tool",
                    "function_name": "browser_view",
                    "content": "<html><head><title>Test</title></head><body>text</body></html>",
                },
            ]
        )

        memory.compact(keep_summary=False)

        assert memory.messages[0]["content"] == "(removed)"

    def test_compact_default_backward_compatible(self) -> None:
        """无参数调用 compact() 保持原有行为 (removed)"""
        memory = Memory(
            messages=[
                {
                    "role": "tool",
                    "function_name": "browser_view",
                    "content": "some content",
                },
            ]
        )

        memory.compact()

        assert memory.messages[0]["content"] == "(removed)"

    def test_compact_removes_reasoning_content(self) -> None:
        """compact 始终删除 reasoning_content"""
        memory = Memory(
            messages=[
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "long reasoning text here",
                },
            ]
        )

        memory.compact(keep_summary=True)

        assert "reasoning_content" not in memory.messages[0]

    def test_compact_html_without_title_uses_text(self) -> None:
        """没有 title 标签时，使用纯文本前 200 字符"""
        memory = Memory(
            messages=[
                {
                    "role": "tool",
                    "function_name": "browser_view",
                    "content": "<html><body><p>一些重要的内容</p></body></html>",
                },
            ]
        )

        memory.compact(keep_summary=True)

        tool_msg = memory.messages[0]
        assert "一些重要的内容" in tool_msg["content"]
        assert "[已执行]" in tool_msg["content"]

    def test_compact_non_browser_tool_untouched(self) -> None:
        """非 browser 工具的 tool 消息不被 compact 修改"""
        memory = Memory(
            messages=[
                {
                    "role": "tool",
                    "function_name": "shell_exec",
                    "content": "command output",
                },
            ]
        )

        memory.compact(keep_summary=True)

        assert memory.messages[0]["content"] == "command output"
