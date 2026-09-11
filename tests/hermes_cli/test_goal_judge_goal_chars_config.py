"""`auxiliary.goal_judge.goal_max_chars` must actually reach the judge prompt.

Same bug class as the timeout key (#91022): the char budget for the GOAL text
(card body) in the judge prompt was hardcoded (`_truncate(goal, 2000)`), so a
long card was judged against half a contract — acceptance criteria living below
character 2000 were invisible to the judge. The key is read via the same
`_goal_judge_setting` reader as max_tokens/timeout; default preserves the old
2000 behavior.
"""

import pytest

import hermes_cli.goals as goals
from hermes_cli.goals import _goal_judge_goal_max_chars


def _patch_config(monkeypatch, cfg):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)


def test_reader_returns_configured_limit(monkeypatch):
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"goal_max_chars": 16000}}})
    assert _goal_judge_goal_max_chars() == 16000


def test_reader_falls_back_when_key_absent(monkeypatch):
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"max_tokens": 512}}})
    assert _goal_judge_goal_max_chars() == goals._JUDGE_GOAL_CHARS


def test_reader_falls_back_on_non_positive_or_garbage(monkeypatch):
    for bad in (0, -5, "big", None):
        _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"goal_max_chars": bad}}})
        assert _goal_judge_goal_max_chars() == goals._JUDGE_GOAL_CHARS


def test_judge_goal_sends_goal_text_within_configured_budget(monkeypatch):
    """The wiring that matters: with a raised budget, judge_goal must pass the
    FULL goal text through to the prompt-building path — no hardcoded 2000
    cliff. Verified via the truncated-response invariant: goal text longer than
    the old hardcoded limit but shorter than the configured one reaches
    ``call_llm`` intact inside the user prompt."""
    captured = {}

    class _FakeAux:
        @staticmethod
        def call_llm(*args, **kwargs):
            captured.update(kwargs)
            for m in kwargs.get("messages") or []:
                if isinstance(m, dict) and "SUBSTANTIVE-MARKER" in str(m.get("content", "")):
                    captured["user_prompt"] = m["content"]
            raise TimeoutError("simulated transport timeout")

    import sys

    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _FakeAux)
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"goal_max_chars": 16000}}})

    # 3000 chars: above the old 2000 cliff, below the configured 16000 budget.
    long_goal = "SUBSTANTIVE-MARKER " + ("x" * 3000)
    goals.judge_goal(long_goal, "here is a substantive response to evaluate")

    prompt = captured.get("user_prompt") or ""
    assert "SUBSTANTIVE-MARKER" in prompt, "goal text must reach the judge prompt"
    assert prompt.count("x") >= 3000, (
        "goal text must arrive untruncated when within the configured budget "
        f"(saw {prompt.count('x')} of 3000 chars)"
    )


def test_judge_goal_still_truncates_above_configured_budget(monkeypatch):
    """Raising the budget removes the cliff, not the ceiling: text beyond the
    configured limit is still truncated (context-bomb guard preserved)."""
    captured = {}

    class _FakeAux:
        @staticmethod
        def call_llm(*args, **kwargs):
            for m in kwargs.get("messages") or []:
                if isinstance(m, dict) and "SUBSTANTIVE-MARKER" in str(m.get("content", "")):
                    captured["user_prompt"] = m["content"]
            raise TimeoutError("simulated transport timeout")

    import sys

    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _FakeAux)
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"goal_max_chars": 2000}}})

    long_goal = "SUBSTANTIVE-MARKER " + ("x" * 5000)
    goals.judge_goal(long_goal, "here is a substantive response to evaluate")

    prompt = captured.get("user_prompt") or ""
    assert prompt.count("x") < 5000, "budget ceiling must still apply"
