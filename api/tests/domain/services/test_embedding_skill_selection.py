"""Tests for embedding-based skill selection integration."""


class TestComputeHasPositiveMatch:

    def test_empty_scores(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([]) is False

    def test_low_top_score(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([0.10, 0.05]) is False

    def test_high_score_with_gap(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([0.6, 0.3]) is True

    def test_single_score_above_minimum(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([0.3]) is True

    def test_high_score_no_gap(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([0.45, 0.44]) is True

    def test_moderate_score_no_gap(self):
        from app.domain.services.agent_task_runner import _compute_has_positive_match
        assert _compute_has_positive_match([0.30, 0.28]) is False
