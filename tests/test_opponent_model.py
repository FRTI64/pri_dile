"""P2 C2/C3: learned opponent classifier, next-action model and gate."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ipd.core.opponents.bank import TYPE_BANK
from ipd.core.primitives import C, D
from ipd.training.opponent_model import (
    CLASSES, EXPLORATION_TEMPLATES, RESPONSE_CLASSES,
    LinearOpponentModel, MLPOpponentModel, OpponentGate, accuracy,
    allow_label, coarse_class, collect_counterfactual_dataset,
    collect_dataset, collect_exploratory_dataset, concat_datasets,
    exploration_actions, per_class_accuracy, split_specs, suite_class,
    three_way_calibrated_labels, train_mlp, train_softmax,
)


class LabelTests(unittest.TestCase):
    def test_coarse_class(self):
        self.assertEqual(coarse_class("TFT", "nice"), "cooperative")
        self.assertEqual(coarse_class("NoisyTFT.10", "forgiving"), "noisy")
        self.assertEqual(coarse_class("Joss.05", "forgiving"), "noisy")
        self.assertEqual(coarse_class("GTFT.10", "forgiving"), "forgiving")
        self.assertEqual(coarse_class("Extort2", "adaptive"), "adaptive")
        self.assertEqual(coarse_class("LateD25", "late"), "late")
        self.assertIn(coarse_class("AllD", "defect"), CLASSES)

    def test_suite_class(self):
        self.assertEqual(suite_class("noisy_v2"), "noisy")
        self.assertEqual(suite_class("extort_v2"), "adaptive")
        self.assertEqual(suite_class("random_v2"), "random")
        self.assertIsNone(suite_class("unknown_family"))

    def test_allow_label(self):
        self.assertEqual(allow_label("defect"), "allow")
        self.assertEqual(allow_label("late"), "allow")
        self.assertEqual(allow_label("random"), "allow")
        self.assertEqual(allow_label("noisy"), "deny")
        self.assertEqual(allow_label("adaptive"), "deny")

    def test_type_split_is_disjoint_and_stratified(self):
        train, val = split_specs(TYPE_BANK, validation_fraction=0.2, seed=1)
        self.assertTrue(train)
        self.assertTrue(val)
        train_names = {spec.name for spec in train}
        val_names = {spec.name for spec in val}
        self.assertFalse(train_names & val_names)
        self.assertEqual(train_names | val_names,
                         {spec.name for spec in TYPE_BANK})
        train_classes = {coarse_class(s.name, s.family) for s in train}
        self.assertEqual(train_classes, set(CLASSES))


class DatasetTests(unittest.TestCase):
    def test_collect_dataset_shapes(self):
        specs = [spec for spec in TYPE_BANK
                 if spec.name in ("AllC", "AllD", "TFT", "LateD10",
                                  "Random.50", "Extort1.5")]
        data = collect_dataset(specs, episodes=1, rounds=12, seed=3)
        self.assertEqual(len(data), len(specs) * 12)
        self.assertTrue(all(len(row) == 85 for row in data.features))
        self.assertTrue(all(action in (C, D) for action in data.next_actions))
        self.assertEqual(set(data.labels),
                         {coarse_class(s.name, s.family) for s in specs})

    def test_collect_dataset_with_stats(self):
        specs = [spec for spec in TYPE_BANK
                 if spec.name in ("AllC", "AllD", "TFT")]
        data, stats = collect_dataset(
            specs, episodes=1, rounds=12, seed=3, with_stats=True)
        self.assertEqual(stats["samples"], len(data))
        self.assertEqual(set(stats["by_class"]),
                         {coarse_class(s.name, s.family) for s in specs})


class ExploratoryDataTests(unittest.TestCase):
    def test_exploration_templates_are_valid_and_exciting(self):
        for template in EXPLORATION_TEMPLATES:
            rng = random.Random(3)
            actions, chosen = exploration_actions(rng, rounds=40,
                                                  template=template)
            self.assertEqual(chosen, template)
            self.assertEqual(len(actions), 40)
            self.assertTrue(all(action in (C, D) for action in actions))
            if template == "cooperate":
                self.assertEqual(sum(actions), 0)
            else:
                self.assertGreater(sum(actions), 0)

    def test_exploration_is_deterministic(self):
        a, _ = exploration_actions(random.Random(11), 60, "single_probes")
        b, _ = exploration_actions(random.Random(11), 60, "single_probes")
        self.assertEqual(a, b)

    def test_exploratory_dataset_has_response_excitation(self):
        specs = [spec for spec in TYPE_BANK
                 if spec.name in ("AllC", "TFT", "Grim", "AllD",
                                  "Random.50", "Extort2")]
        data, stats = collect_exploratory_dataset(
            specs, episodes=2, rounds=40, seed=3)
        self.assertEqual(len(data), len(specs) * 2 * 40)
        self.assertTrue(all(len(row) == 85 for row in data.features))
        self.assertGreater(stats["excited_samples"], 0)
        self.assertGreater(stats["excited_rate"], 0.1)
        self.assertGreater(stats["defection_opportunities"], 0)
        self.assertIn("adaptive", stats["by_class"])
        self.assertGreater(
            stats["by_class"]["adaptive"]["excited_samples"], 0)
        self.assertTrue(any(row[35] > 0 for row in data.features))

    def test_concat_datasets_preserves_rows(self):
        specs = [spec for spec in TYPE_BANK if spec.name in ("AllC", "AllD")]
        first = collect_dataset(specs, episodes=1, rounds=8, seed=2)
        second, _ = collect_exploratory_dataset(
            specs, episodes=1, rounds=8, seed=2)
        combined = concat_datasets(first, second)
        self.assertEqual(len(combined), len(first) + len(second))
        self.assertEqual(combined.labels, first.labels + second.labels)
        self.assertEqual(combined.next_actions,
                         first.next_actions + second.next_actions)


class CounterfactualDataTests(unittest.TestCase):
    def _collect(self, name):
        spec = [spec for spec in TYPE_BANK if spec.name == name][0]
        return collect_counterfactual_dataset(
            [spec], episodes=1, rounds=30, seed=5, branches=4, horizon=4,
            prefix_modes=("explore",))

    def test_counterfactual_shapes_and_branch_count(self):
        data, response, stats = self._collect("TFT")
        self.assertGreater(len(data), 0)
        self.assertEqual(stats["branch_pairs"], len(response))
        self.assertEqual(stats["branch_pairs"],
                         stats["responsive_pairs"]
                         + stats["unresponsive_pairs"])
        self.assertTrue(set(response.labels) <= set(RESPONSE_CLASSES))

    def test_cooperator_and_defector_are_unresponsive(self):
        for name in ("AllC", "AllD", "Random.50"):
            _, response, stats = self._collect(name)
            self.assertEqual(stats["responsive_pairs"], 0, name)
            self.assertTrue(all(label == "unresponsive"
                                for label in response.labels), name)

    def test_tft_is_responsive_to_counterfactual_action(self):
        _, response, stats = self._collect("TFT")
        self.assertGreater(stats["responsive_pairs"], 0)
        self.assertTrue(all(label == "responsive"
                            for label in response.labels))

    def test_repeated_counterfactual_probabilities(self):
        spec = [spec for spec in TYPE_BANK if spec.name == "TFT"][0]
        data, response, stats = collect_counterfactual_dataset(
            [spec], episodes=1, rounds=30, seed=5, branches=4, horizon=4,
            repeats=3, prefix_modes=("explore",))
        self.assertEqual(stats["branch_pairs"], len(response))
        self.assertEqual(len(response.response_probabilities), len(response))
        self.assertTrue(all(0.0 <= p <= 1.0
                            for p in response.response_probabilities))
        self.assertTrue(all(p == 1.0
                            for p in response.response_probabilities))
        self.assertGreater(len(data), 0)

    def test_calibrated_three_way_labels(self):
        labels = three_way_calibrated_labels(
            [0.0, 0.1, 0.5, 0.9],
            safe_threshold=0.0, risky_threshold=0.5)
        self.assertEqual(labels, ["safe", "unknown", "risky", "risky"])


def _synthetic():
    rng = random.Random(7)
    classes = ["a", "b", "c"]
    centers = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    x = []
    y = []
    for index, center in enumerate(centers):
        for _ in range(80):
            x.append([center[j] + rng.gauss(0, 0.35) for j in range(3)])
            y.append(classes[index])
    return x, y, classes


class ModelTests(unittest.TestCase):
    def test_softmax_learns_separable_data(self):
        x, y, classes = _synthetic()
        model, report = train_softmax(
            x[:180], y[:180], classes=classes, epochs=120, lr=0.2,
            validation=(x[180:], y[180:]))
        self.assertGreater(report["val_accuracy"], 0.9)
        self.assertGreater(accuracy(model, x, y), 0.9)
        per_class = per_class_accuracy(model, x, y)
        self.assertEqual(set(per_class["accuracy"]), set(classes))

    def test_save_load_roundtrip(self):
        x, y, classes = _synthetic()
        model, _ = train_softmax(x, y, classes=classes, epochs=30, lr=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = LinearOpponentModel.load(path)
            self.assertEqual(loaded.classes, model.classes)
            f = x[0]
            self.assertEqual(loaded.predict(f), model.predict(f))
            self.assertEqual(loaded.to_dict(), model.to_dict())

    def test_next_action_binary_model(self):
        x, y, _ = _synthetic()
        actions = [C if label == "a" else D for label in y]
        model, report = train_softmax(
            x, ["C" if a == C else "D" for a in actions],
            classes=["C", "D"], epochs=80, lr=0.2)
        self.assertGreater(report["train_accuracy"], 0.8)
        self.assertIn(model.predict_next_action(x[0]), (C, D))

    def test_mlp_learns_separable_data_and_roundtrips(self):
        x, y, classes = _synthetic()
        model, report = train_mlp(
            x[:180], y[:180], classes=classes, hidden=(16, 8),
            epochs=120, lr=1e-3, validation=(x[180:], y[180:]))
        self.assertGreater(report["val_accuracy"], 0.9)
        self.assertGreater(accuracy(model, x, y), 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp.json"
            model.save(path)
            loaded = MLPOpponentModel.load(path)
            self.assertEqual(loaded.classes, model.classes)
            self.assertEqual(loaded.predict(x[0]), model.predict(x[0]))

    def test_mlp_binary_gate_scores(self):
        x, y, _ = _synthetic()
        labels = ["allow" if label == "a" else "deny" for label in y]
        model, report = train_mlp(
            x[:180], labels[:180], classes=["allow", "deny"],
            hidden=(16, 8), epochs=120, lr=1e-3,
            validation=(x[180:], labels[180:]))
        self.assertGreater(report["val_accuracy"], 0.9)
        self.assertIn(model.predict(x[0])[0], ("allow", "deny"))


class GateTests(unittest.TestCase):
    def test_gate_allows_high_confidence_safe_class(self):
        class Dummy:
            def predict(self, features):
                return "defect", 0.9

        gate = OpponentGate(Dummy(), allowed=("defect",), threshold=0.6,
                            min_day=1)
        allowed, info = gate(1, [C], [D])
        self.assertTrue(allowed)
        self.assertEqual(info["label"], "defect")
        self.assertEqual(gate.allowed_calls, 1)
        gate.reset()
        self.assertEqual(gate.calls, 0)

    def test_gate_rejects_low_confidence_and_warmup(self):
        class Dummy:
            def predict(self, features):
                return "defect", 0.5

        gate = OpponentGate(Dummy(), allowed=("defect",), threshold=0.6,
                            min_day=3)
        self.assertFalse(gate(0, [], [])[0])
        self.assertFalse(gate(4, [C], [D])[0])

    def test_gate_rejects_forbidden_class(self):
        class Dummy:
            def predict(self, features):
                return "noisy", 0.99

        gate = OpponentGate(Dummy(), allowed=("defect", "late", "random"),
                            threshold=0.6, min_day=0)
        allowed, info = gate(1, [C], [D])
        self.assertFalse(allowed)
        self.assertEqual(info["label"], "noisy")

    def test_gate_response_model_veto(self):
        class Dummy:
            def predict(self, features):
                return "defect", 0.9

        class Response:
            classes = ["unresponsive", "responsive"]

            def __init__(self, p_responsive):
                self.p_responsive = p_responsive

            def probabilities(self, features):
                return [1.0 - self.p_responsive, self.p_responsive]

        gate = OpponentGate(
            Dummy(), allowed=("defect",), threshold=0.6, min_day=0,
            response_model=Response(0.7), max_response_probability=0.5)
        allowed, info = gate(1, [C], [D])
        self.assertFalse(allowed)
        self.assertAlmostEqual(info["response_probability"], 0.7)

        gate = OpponentGate(
            Dummy(), allowed=("defect",), threshold=0.6, min_day=0,
            response_model=Response(0.2), max_response_probability=0.5)
        allowed, info = gate(1, [C], [D])
        self.assertTrue(allowed)
        self.assertAlmostEqual(info["response_probability"], 0.2)


if __name__ == "__main__":
    unittest.main()
