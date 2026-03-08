from app.domain.services.prompts.planner import UPDATE_PLAN_PROMPT


def test_update_plan_prompt_has_execution_summary_placeholder() -> None:
    """UPDATE_PLAN_PROMPT 必须包含 {execution_summary} 占位符"""
    assert "{execution_summary}" in UPDATE_PLAN_PROMPT


def test_update_plan_prompt_formats_with_execution_summary() -> None:
    """模板可以正确格式化 execution_summary"""
    import json

    step_data = json.dumps({"id": "1", "description": "test step"})
    plan_data = json.dumps({"steps": []})
    summary = "执行了 shell_exec 命令，成功安装了依赖包"

    result = UPDATE_PLAN_PROMPT.format(
        step=step_data, plan=plan_data, execution_summary=summary
    )

    assert "执行了 shell_exec 命令" in result
    assert "成功安装了依赖包" in result
