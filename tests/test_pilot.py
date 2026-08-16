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
