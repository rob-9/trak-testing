import importlib.util
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


def test_auc():
    assert pilot.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert pilot.roc_auc([0, 0], [0.1, 0.2]) is None


def test_balanced_sample_spans_groups():
    rows = [
        {"group": "a", "id": "1"},
        {"group": "a", "id": "2"},
        {"group": "b", "id": "3"},
        {"group": "b", "id": "4"},
    ]
    result = pilot.balanced_sample(rows, "group", 2, 7)
    assert {row["group"] for row in result} == {"a", "b"}
