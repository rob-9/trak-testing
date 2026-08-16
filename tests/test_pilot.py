import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pilot", ROOT / "pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


def test_extract_json_handles_fence():
    assert pilot.extract_json('```json\n{"reward": 80}\n```') == {"reward": 80}


def test_parse_validation_protocol():
    result = pilot.parse_validation("0|1|1|1|0.95\n1|0|1|1|0.80", 2)
    assert result[0]["equivalent"] is True
    assert result[1]["equivalent"] is False
    one_based = pilot.parse_validation("1|1|1|1|0.95\n2|1|1|1|0.80", 2)
    assert [row["index"] for row in one_based] == [0, 1]


def test_stable_id_is_deterministic():
    assert pilot.stable_id("a", "b") == pilot.stable_id("a", "b")
    assert pilot.stable_id("a", "b") != pilot.stable_id("ab")


def test_spearman():
    assert math.isclose(pilot.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert pilot.spearman([1, 2], [1, 2]) is None


def test_excess_disagreement_components():
    rows = [
        {"response_wording_index": 0, "behavior_label": "a"},
        {"response_wording_index": 0, "behavior_label": "a"},
        {"response_wording_index": 1, "behavior_label": "b"},
        {"response_wording_index": 1, "behavior_label": "b"},
    ]
    within, within_n = pilot.disagreement_rate(rows, "within")
    cross, cross_n = pilot.disagreement_rate(rows, "cross")
    assert within == 0.0 and within_n == 2
    assert cross == 1.0 and cross_n == 4


def test_crossed_effects_separate_response_and_scorer():
    rows = []
    for response_index in range(2):
        for scoring_index in range(2):
            rows.append({
                "response_wording_index": response_index,
                "scoring_wording_index": scoring_index,
                "reward": response_index * 10 + scoring_index,
                "behavior_label": "good",
            })
    result = pilot.crossed_greedy_effects(rows)
    assert result is not None
    assert result["response_effect_fraction"] > result["scorer_wording_effect_fraction"]


def test_balanced_sample_spans_groups():
    rows = [
        {"group": "a", "id": "1"},
        {"group": "a", "id": "2"},
        {"group": "b", "id": "3"},
        {"group": "b", "id": "4"},
    ]
    result = pilot.balanced_sample(rows, "group", 2, 7)
    assert {row["group"] for row in result} == {"a", "b"}


def test_judge_protocol_requires_all_numeric_ratings():
    class StubClient:
        def generate(self, _prompt):
            return "85|appropriate_refusal_safe_redirection|0|4|1|0|1|Safe refusal"

    result = pilot.judge_one(StubClient(), "unsafe request", "I cannot help with that.")
    assert result["reward"] == 85
    assert result["harmfulness_actionability"] == 0
    assert result["conditional_helpfulness"] == 4


def test_exact_repeat_effects_measure_judge_noise():
    greedy = [
        {"generation_id": "a", "response_wording_index": 0, "scoring_wording_index": 0,
         "reward": 90, "behavior_label": "good"},
        {"generation_id": "b", "response_wording_index": 1, "scoring_wording_index": 0,
         "reward": 80, "behavior_label": "good"},
    ]
    repeats = [
        {"generation_id": "a", "response_wording_index": 0, "evaluator_repeat_index": 1,
         "reward": 70, "behavior_label": "bad"},
        {"generation_id": "b", "response_wording_index": 1, "evaluator_repeat_index": 1,
         "reward": 85, "behavior_label": "good"},
    ]
    result = pilot.exact_repeat_effects(greedy, repeats)
    assert result is not None
    assert result["behavior_label_disagreement_rate"] == 0.5
    assert result["preference_flip_rate"] == 1.0


def test_preference_flip_margin_sensitivity():
    rows = [
        {"response_wording_index": 0, "scoring_wording_index": 0,
         "reward": 100, "behavior_label": "good"},
        {"response_wording_index": 1, "scoring_wording_index": 0,
         "reward": 95, "behavior_label": "good"},
        {"response_wording_index": 0, "scoring_wording_index": 1,
         "reward": 95, "behavior_label": "good"},
        {"response_wording_index": 1, "scoring_wording_index": 1,
         "reward": 100, "behavior_label": "good"},
    ]
    result = pilot.crossed_greedy_effects(rows)
    assert result is not None
    sensitivity = result["preference_flip_sensitivity_by_minimum_margin"]
    assert sensitivity["0"]["rate"] == 1.0
    assert sensitivity["5"]["rate"] is None
    assert result["mean_fixed_response_reward_range"] == 5.0
