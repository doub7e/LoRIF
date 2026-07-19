import unittest

import numpy as np
import torch

from lorif.lds import linear_datamodeling_score


class LDSTests(unittest.TestCase):
    def test_known_additive_utilities_score_one(self):
        scores = torch.tensor(
            [[0.2, 1.1, -0.3, 0.7], [-0.4, 0.9, 1.3, 0.1]],
            dtype=torch.float64,
        )
        membership = torch.tensor(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=torch.bool,
        )
        utilities = (scores @ membership.to(scores.dtype).T).T.numpy()
        result = linear_datamodeling_score(
            scores,
            membership,
            utilities,
            num_bootstrap=200,
            bootstrap_seed=9,
        )
        self.assertAlmostEqual(result.mean, 1.0)
        self.assertAlmostEqual(result.ci_lower, 1.0)
        self.assertAlmostEqual(result.ci_upper, 1.0)
        np.testing.assert_allclose(result.per_query, np.ones(2))

    def test_negative_cross_entropy_is_the_explicit_utility(self):
        scores = torch.tensor([[1.0, 3.0, 2.0]])
        membership = torch.eye(3, dtype=torch.bool)
        losses = np.array([[9.0], [1.0], [4.0]])
        result = linear_datamodeling_score(
            scores, membership, -losses, num_bootstrap=20
        )
        self.assertAlmostEqual(result.mean, 1.0)

    def test_tensor_utilities_are_detached_and_membership_is_binary(self):
        scores = torch.tensor([[1.0, 2.0]])
        membership = torch.tensor([[1, 0], [0, 1]])
        utilities = torch.tensor([[1.0], [2.0]], requires_grad=True)
        result = linear_datamodeling_score(
            scores, membership, utilities, num_bootstrap=20
        )
        self.assertAlmostEqual(result.mean, 1.0)
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            linear_datamodeling_score(
                scores, torch.tensor([[2, 0], [0, 1]]), utilities, num_bootstrap=20
            )


if __name__ == "__main__":
    unittest.main()
