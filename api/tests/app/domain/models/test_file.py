from app.domain.models.file import File


class TestFileDimensionFields:
    def test_default_none(self) -> None:
        f = File(filename="test.txt")
        assert f.width is None
        assert f.height is None
        assert f.original_width is None
        assert f.original_height is None
        assert f.multimodal_eligible is None

    def test_set_values(self) -> None:
        f = File(
            filename="img.png",
            width=800,
            height=600,
            original_width=1600,
            original_height=1200,
            multimodal_eligible=True,
        )
        assert f.width == 800
        assert f.original_width == 1600
        assert f.multimodal_eligible is True

    def test_backward_compat_model_dump(self) -> None:
        f = File(filename="old.png")
        d = f.model_dump(mode="json")
        assert "width" in d
        assert "multimodal_eligible" in d
        assert d["width"] is None
