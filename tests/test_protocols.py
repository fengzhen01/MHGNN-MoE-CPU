from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mhgnn_moe.data import audit_split, load_dataset, load_or_create_split, make_protocol_split


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_dataset(ROOT / "data" / "processed", "HIT", torch.device("cpu"))

    def test_protocol_entity_isolation_and_balance(self) -> None:
        positives = self.bundle.positives_unique
        settings = (("strict_unique", 10), ("cold_herb", 5), ("cold_symptom", 5), ("cold_both", 5))
        for protocol, folds in settings:
            sizes = []
            for fold in range(folds):
                split = make_protocol_split(positives, protocol, folds, fold, 0)
                sizes.append(len(split["valid_pos"]))
                train = set(map(tuple, split["train_pos"].tolist()))
                valid = set(map(tuple, split["valid_pos"].tolist()))
                self.assertTrue(train.isdisjoint(valid))
                if protocol in {"cold_herb", "cold_both"}:
                    self.assertTrue(set(split["train_pos"][:, 0]).isdisjoint(split["heldout_herbs"]))
                    self.assertTrue(set(split["valid_pos"][:, 0]).issubset(split["heldout_herbs"]))
                if protocol in {"cold_symptom", "cold_both"}:
                    self.assertTrue(set(split["train_pos"][:, 1]).isdisjoint(split["heldout_symptoms"]))
                    self.assertTrue(set(split["valid_pos"][:, 1]).issubset(split["heldout_symptoms"]))
            self.assertGreater(min(sizes), 0)
            if protocol != "cold_both":
                self.assertLess(max(sizes) / min(sizes), 1.25, msg=f"{protocol}: {sizes}")

    def test_negative_sampling_audit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for protocol, folds in (("cold_herb", 2), ("cold_symptom", 2), ("cold_both", 5)):
                split = load_or_create_split(
                    self.bundle, protocol, folds, 0, 0, 10, 0.5, True, Path(folder) / protocol
                )
                result = audit_split(self.bundle, split, protocol)
                self.assertTrue(all(value for key, value in result.items() if isinstance(value, bool)))

    def test_unlabeled_confidence_range(self) -> None:
        edges = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
        values = self.bundle.unlabeled_confidence(edges, minimum=0.25, power=1.0)
        self.assertTrue(np.all(values >= 0.25))
        self.assertTrue(np.all(values <= 1.0))

    def test_pair_stats_hide_proximity_by_default(self) -> None:
        edges = torch.as_tensor([[0, 0], [0, 1]], dtype=torch.long)
        safe = self.bundle.pair_stats(edges)
        legacy = self.bundle.pair_stats(edges, include_proximity=True)
        self.assertTrue(torch.all(safe[:, 2:] == 0))
        self.assertEqual(safe.shape, legacy.shape)

    def test_network_close_negatives_are_closer_than_network_far(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            close = load_or_create_split(
                self.bundle, "strict_unique", 2, 0, 0, 10, 0.5, True,
                Path(folder) / "close", validation_negative_strategy="network_close",
            )
            far = load_or_create_split(
                self.bundle, "strict_unique", 2, 0, 0, 10, 0.5, True,
                Path(folder) / "far", validation_negative_strategy="network_far",
            )
            close_d = self.bundle.proximity_d[close["valid_neg"][:, 0], close["valid_neg"][:, 1]]
            far_d = self.bundle.proximity_d[far["valid_neg"][:, 0], far["valid_neg"][:, 1]]
            self.assertLess(float(np.nanmedian(close_d)), float(np.nanmedian(far_d)))


if __name__ == "__main__":
    unittest.main()
