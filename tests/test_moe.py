"""P2 C3: three-way gate and safe/risky/unknown MoE routing."""
from __future__ import annotations

import unittest

from ipd.core.primitives import C, D
from ipd.policies.moe import DefectExpert, ThreeWayMoE
from ipd.training.opponent_model import (
    THREE_WAY_CLASSES, ThreeWayGate, three_way_gate_label,
    three_way_labels,
)


def _features(excited: bool) -> list[float]:
    row = [0.0] * 85
    row[35] = 1.0 if excited else 0.0
    return row


class ThreeWayLabelTests(unittest.TestCase):
    def test_label_requires_excitation(self):
        self.assertEqual(three_way_gate_label("late", _features(True)),
                         "safe")
        self.assertEqual(three_way_gate_label("adaptive", _features(True)),
                         "risky")
        self.assertEqual(three_way_gate_label("late", _features(False)),
                         "unknown")
        self.assertEqual(three_way_gate_label("defect", _features(False)),
                         "unknown")

    def test_label_vector(self):
        labels = three_way_labels(
            ["random", "noisy", "late"],
            [_features(True), _features(True), _features(False)])
        self.assertEqual(labels, ["safe", "risky", "unknown"])
        self.assertTrue(set(labels) <= set(THREE_WAY_CLASSES))


class _DummyClassifier:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def predict(self, features):
        return self.label, self.confidence


class ThreeWayGateTests(unittest.TestCase):
    def test_safe_and_risky_thresholds(self):
        gate = ThreeWayGate(_DummyClassifier("safe", 0.8),
                            safe_threshold=0.7, min_day=0)
        route, info = gate(1, [C], [C])
        self.assertEqual(route, "safe")
        self.assertEqual(info["label"], "safe")

        gate = ThreeWayGate(_DummyClassifier("safe", 0.5),
                            safe_threshold=0.7, min_day=0)
        self.assertEqual(gate(1, [C], [C])[0], "unknown")

        gate = ThreeWayGate(_DummyClassifier("risky", 0.9),
                            risky_threshold=0.7, min_day=0)
        self.assertEqual(gate(1, [C], [C])[0], "risky")

    def test_warmup_is_unknown(self):
        gate = ThreeWayGate(_DummyClassifier("safe", 0.99), min_day=5)
        self.assertEqual(gate(0, [C], [C])[0], "unknown")
        self.assertEqual(gate.diagnostics()["unknown_calls"], 1)

    def test_response_veto_turns_safe_into_risky(self):
        class Response:
            classes = ["unresponsive", "responsive"]

            def probabilities(self, features):
                return [0.2, 0.8]

        gate = ThreeWayGate(
            _DummyClassifier("safe", 0.9), min_day=0,
            response_model=Response(), max_response_probability=0.5)
        route, info = gate(1, [C], [C])
        self.assertEqual(route, "risky")
        self.assertAlmostEqual(info["response_probability"], 0.8)


class _DummyPolicy:
    def __init__(self, action: int, name: str = "dummy"):
        self.action = action
        self.name = name
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, t, mine, theirs):
        self.calls += 1
        return self.action


class _FixedGate:
    def __init__(self, route):
        self.route = route
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, t, mine, theirs):
        self.calls += 1
        return self.route, {"route": self.route}

    def diagnostics(self):
        return {"calls": self.calls}


class MoETests(unittest.TestCase):
    def test_only_selected_expert_is_called(self):
        base = _DummyPolicy(C, "base")
        safe = _DummyPolicy(D, "safe")
        gate = _FixedGate("safe")
        moe = ThreeWayMoE(base, gate, safe_expert=safe)
        self.assertEqual(moe.act(1, [], []), D)
        self.assertEqual(safe.calls, 1)
        self.assertEqual(base.calls, 0)
        self.assertEqual(moe.diagnostics()["safe"], 1)

    def test_risky_and_unknown_fall_back(self):
        base = _DummyPolicy(C, "base")
        gate = _FixedGate("risky")
        moe = ThreeWayMoE(base, gate)
        self.assertEqual(moe.act(1, [], []), C)
        self.assertEqual(base.calls, 1)
        gate.route = "unknown"
        self.assertEqual(moe.act(2, [], []), C)
        self.assertEqual(base.calls, 2)
        diag = moe.diagnostics()
        self.assertEqual(diag["risky"], 1)
        self.assertEqual(diag["unknown"], 1)
        self.assertEqual(diag["fallback_gate"], 2)

    def test_defect_expert(self):
        self.assertEqual(DefectExpert().act(0, [], []), D)

    def test_reset_resets_experts_and_gate(self):
        base = _DummyPolicy(C)
        safe = _DummyPolicy(D)
        gate = _FixedGate("safe")
        moe = ThreeWayMoE(base, gate, safe_expert=safe)
        moe.act(1, [], [])
        moe.reset()
        self.assertEqual(moe.stats["rounds"], 0)
        # One reset in __init__, one explicit reset.
        self.assertEqual(gate.resets, 2)
        # base and safe are distinct, so both must be reset.
        self.assertEqual(base.resets, 2)
        self.assertEqual(safe.resets, 2)


if __name__ == "__main__":
    unittest.main()
