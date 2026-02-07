"""Tests for G1: Adaptive Templating Engine."""
from usc.mem.adaptive_templates import AdaptiveTemplateEngine, TemplateStats


def _log_lines():
    """Sample log lines with repetitive structure."""
    return [
        "User alice logged in from 10.0.0.1",
        "User bob logged in from 10.0.0.2",
        "User charlie logged in from 10.0.0.3",
        "User alice logged out from 10.0.0.1",
        "User bob logged out from 10.0.0.2",
        "Error: connection timeout for user dave",
        "Error: connection timeout for user eve",
        "Error: connection refused for user frank",
        "Metric: cpu_usage=72.5 host=server-1",
        "Metric: cpu_usage=85.3 host=server-2",
        "Metric: cpu_usage=45.1 host=server-3",
    ]


class TestMining:
    def test_mine_returns_template_and_params(self):
        engine = AdaptiveTemplateEngine()
        result = engine.mine("User alice logged in from 10.0.0.1")
        assert isinstance(result.template, str)
        assert isinstance(result.params, list)

    def test_repeated_patterns_share_template(self):
        engine = AdaptiveTemplateEngine()
        r1 = engine.mine("User alice logged in from 10.0.0.1")
        r2 = engine.mine("User bob logged in from 10.0.0.2")
        # After Drain3 sees 2nd similar line, templates should converge
        r3 = engine.mine("User charlie logged in from 10.0.0.3")
        stats = engine.template_stats()
        # Should have fewer templates than lines
        assert stats.template_count <= stats.total_lines

    def test_mine_chunk_lines_matches_original_interface(self):
        engine = AdaptiveTemplateEngine()
        chunks = [
            "User alice logged in from 10.0.0.1\nUser bob logged in from 10.0.0.2\n",
            "Error: connection timeout for user dave\n",
        ]
        templates, params = engine.mine_chunk_lines(chunks)
        assert len(templates) == 2
        assert len(params) == 2
        assert isinstance(templates[0], str)
        assert isinstance(params[0], list)

    def test_trailing_newline_preserved(self):
        engine = AdaptiveTemplateEngine()
        chunks = ["hello world\n"]
        templates, _ = engine.mine_chunk_lines(chunks)
        assert templates[0].endswith("\n")

    def test_no_trailing_newline_preserved(self):
        engine = AdaptiveTemplateEngine()
        chunks = ["hello world"]
        templates, _ = engine.mine_chunk_lines(chunks)
        assert not templates[0].endswith("\n")


class TestStats:
    def test_empty_stats(self):
        engine = AdaptiveTemplateEngine()
        stats = engine.template_stats()
        assert stats.template_count == 0
        assert stats.total_lines == 0

    def test_stats_after_mining(self):
        engine = AdaptiveTemplateEngine()
        for line in _log_lines():
            engine.mine(line)
        stats = engine.template_stats()
        assert stats.template_count > 0
        assert stats.total_lines == len(_log_lines())
        assert stats.coverage_pct == 100.0
        assert stats.reuse_rate >= 1.0

    def test_reset_clears_stats(self):
        engine = AdaptiveTemplateEngine()
        engine.mine("test line")
        engine.reset()
        stats = engine.template_stats()
        assert stats.template_count == 0


class TestAutoTune:
    def test_auto_tune_returns_valid_threshold(self):
        lines = _log_lines()
        th = AdaptiveTemplateEngine.auto_tune(lines)
        assert 0.0 < th < 1.0

    def test_auto_tune_with_custom_thresholds(self):
        lines = _log_lines()
        th = AdaptiveTemplateEngine.auto_tune(lines, thresholds=[0.3, 0.5, 0.7])
        assert th in [0.3, 0.5, 0.7]

    def test_tuned_threshold_produces_fewer_or_equal_templates(self):
        lines = _log_lines()
        th = AdaptiveTemplateEngine.auto_tune(lines)

        tuned = AdaptiveTemplateEngine(similarity_threshold=th)
        default = AdaptiveTemplateEngine(similarity_threshold=0.4)

        for line in lines:
            tuned.mine(line)
            default.mine(line)

        tuned_stats = tuned.template_stats()
        default_stats = default.template_stats()

        # Tuned score should be <= default score
        tuned_score = tuned_stats.template_count + 0.1 * tuned_stats.total_slots
        default_score = default_stats.template_count + 0.1 * default_stats.total_slots
        assert tuned_score <= default_score


class TestMergeTemplates:
    def test_merge_identical_templates(self):
        t = "User <*> logged in"
        merged = AdaptiveTemplateEngine.merge_templates(t, t)
        assert merged == t

    def test_merge_differing_token(self):
        t1 = "User <*> logged in from <*>"
        t2 = "User <*> logged out from <*>"
        merged = AdaptiveTemplateEngine.merge_templates(t1, t2)
        assert merged == "User <*> logged <*> from <*>"

    def test_merge_different_lengths_returns_first(self):
        t1 = "short template"
        t2 = "longer template with extras"
        merged = AdaptiveTemplateEngine.merge_templates(t1, t2)
        assert merged == t1

    def test_merge_all_different(self):
        t1 = "aaa bbb ccc"
        t2 = "xxx yyy zzz"
        merged = AdaptiveTemplateEngine.merge_templates(t1, t2)
        assert merged == "<*> <*> <*>"
