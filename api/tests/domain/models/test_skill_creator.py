from app.domain.models.skill_creator import (
    ScriptFile,
    SkillBlueprint,
    SkillCreationProgress,
    SkillGeneratedFiles,
    ToolDef,
    ToolParamDef,
)


class TestSkillBlueprint:
    def test_create_blueprint_with_required_fields(self) -> None:
        tool = ToolDef(
            name="download_video",
            description="下载B站视频",
            parameters=[
                ToolParamDef(
                    name="url",
                    type="string",
                    description="视频URL",
                    required=True,
                )
            ],
        )
        blueprint = SkillBlueprint(
            skill_name="bilibili-downloader",
            description="下载B站视频并转录",
            tools=[tool],
            search_keywords=["bilibili download python", "yt-dlp"],
            estimated_deps=["yt-dlp"],
        )

        assert blueprint.skill_name == "bilibili-downloader"
        assert len(blueprint.tools) == 1
        assert blueprint.tools[0].parameters[0].required is True

    def test_blueprint_slug_normalization(self) -> None:
        blueprint = SkillBlueprint(
            skill_name="My Cool Skill!",
            description="test",
            tools=[],
            search_keywords=[],
            estimated_deps=[],
        )
        assert blueprint.normalized_slug == "my-cool-skill"

    def test_blueprint_accepts_parameters_dict_and_required_list(self) -> None:
        blueprint = SkillBlueprint.model_validate(
            {
                "skill_name": "video-summary",
                "description": "下载并总结视频",
                "tools": [
                    {
                        "name": "download_video",
                        "description": "下载视频",
                        "parameters": {
                            "url": {
                                "type": "string",
                                "description": "视频地址",
                            },
                            "output_dir": {
                                "type": "string",
                                "description": "输出目录",
                                "default": "./videos",
                            },
                        },
                        "required": ["url"],
                    }
                ],
                "search_keywords": ["yt-dlp python"],
                "estimated_deps": ["yt-dlp"],
            }
        )

        assert len(blueprint.tools) == 1
        assert len(blueprint.tools[0].parameters) == 2
        assert blueprint.tools[0].parameters[0].name == "url"
        assert blueprint.tools[0].parameters[0].required is True


class TestSkillGeneratedFiles:
    def test_generated_files_structure(self) -> None:
        files = SkillGeneratedFiles(
            skill_md="---\nname: test\n---\n# Test",
            manifest={"name": "test", "tools": []},
            scripts=[ScriptFile(path="bundle/run.py", content="print('hello')")],
            dependencies=["requests"],
        )

        assert len(files.scripts) == 1
        assert files.dependencies == ["requests"]

    def test_dependencies_dict_normalized_to_list(self) -> None:
        """LLM 返回 dependencies 为 dict 时应自动提取 pip 包名并过滤中文。"""
        files = SkillGeneratedFiles.model_validate(
            {
                "skill_md": "---\nname: test\n---\n# Test",
                "manifest": {"name": "test", "tools": []},
                "scripts": [{"path": "bundle/run.py", "content": "print('hello')"}],
                "dependencies": {
                    "external_capabilities": [
                        "视频页面解析/平台识别能力",
                        "字幕抓取或音频转写能力",
                    ],
                    "pip_packages": ["requests", "beautifulsoup4", "yt-dlp"],
                },
            }
        )
        # 中文描述应被过滤，仅保留 pip 包名
        assert files.dependencies == ["requests", "beautifulsoup4", "yt-dlp"]

    def test_dependencies_dict_all_chinese_yields_empty(self) -> None:
        """dependencies 为 dict 且全为中文描述时，应返回空列表。"""
        files = SkillGeneratedFiles.model_validate(
            {
                "skill_md": "---\nname: test\n---\n# Test",
                "manifest": {"name": "test", "tools": []},
                "scripts": [{"path": "bundle/run.py", "content": "print('hello')"}],
                "dependencies": {
                    "external_capabilities": [
                        "视频页面解析能力",
                        "字幕接口访问模块",
                    ],
                },
            }
        )
        assert files.dependencies == []

    def test_dependencies_list_of_dicts_normalized(self) -> None:
        """LLM 返回 dependencies 为 [{name, description}] 时应提取包名。"""
        files = SkillGeneratedFiles.model_validate(
            {
                "skill_md": "---\nname: test\n---\n# Test",
                "manifest": {"name": "test", "tools": []},
                "scripts": [{"path": "bundle/run.py", "content": "print('hello')"}],
                "dependencies": [
                    {"name": "requests", "description": "HTTP library"},
                    {"name": "中文包名", "description": "should be filtered"},
                ],
            }
        )
        assert files.dependencies == ["requests"]


class TestSkillCreationProgress:
    def test_progress_serialization(self) -> None:
        progress = SkillCreationProgress(step="analyzing", message="正在分析需求...")
        data = progress.model_dump()
        assert data["step"] == "analyzing"
        assert "message" in data
