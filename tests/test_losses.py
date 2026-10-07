from __future__ import annotations

import unittest

import torch

from mhgnn_moe.experiment import classification_loss


class LossTests(unittest.TestCase):
    def test_pairwise_bpr_prefers_positive_above_negative(self) -> None:
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        weights = torch.ones_like(labels)
        good = torch.tensor([3.0, 2.0, -2.0, -1.0, -3.0, -2.0])
        bad = -good
        config = {"loss_mode": "pairwise_bpr", "ranking_bce_weight": 0.2}
        self.assertLess(
            float(classification_loss(good, labels, weights, config)),
            float(classification_loss(bad, labels, weights, config)),
        )

    def test_pairwise_bpr_requires_groupable_negatives(self) -> None:
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            classification_loss(
                torch.zeros_like(labels), labels, torch.ones_like(labels),
                {"loss_mode": "pairwise_bpr"},
            )


if __name__ == "__main__":
    unittest.main()
