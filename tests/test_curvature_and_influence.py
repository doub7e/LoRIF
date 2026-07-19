import unittest

import torch

from lorif.curvature import CurvatureModel, fit_curvature
from lorif.factors import Factors
from lorif.influence import influence_scores


def exact_factors(matrices: torch.Tensor) -> Factors:
    columns = matrices.shape[2]
    identity = torch.eye(columns, dtype=matrices.dtype).expand(
        matrices.shape[0], -1, -1
    )
    return Factors(matrices, identity)


class CurvatureInfluenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_full_rank_streaming_fit_matches_dense_inverse(self):
        train_matrices = torch.randn(9, 3, 2, dtype=torch.float64)
        query_matrices = torch.randn(4, 3, 2, dtype=torch.float64)
        train = exact_factors(train_matrices)
        query = exact_factors(query_matrices)

        def batches():
            yield Factors(train.left[:4], train.right[:4])
            yield Factors(train.left[4:7], train.right[4:7])
            yield Factors(train.left[7:], train.right[7:])

        damping = 0.7
        curvature = fit_curvature(
            batches,
            (3, 2),
            num_examples=9,
            rank=6,
            oversample=0,
            num_power_iterations=1,
            damping=damping,
            seed=2,
            dtype=torch.float64,
        )
        actual = influence_scores(query, train, curvature)
        dense_train = train_matrices.flatten(1)
        dense_query = query_matrices.flatten(1)
        hessian = dense_train.T @ dense_train + damping * torch.eye(
            6, dtype=torch.float64
        )
        expected = dense_query @ torch.linalg.solve(hessian, dense_train.T)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)

    def test_truncated_woodbury_matches_explicit_truncated_hessian(self):
        train_matrices = torch.randn(7, 3, 2, dtype=torch.float64)
        query_matrices = torch.randn(2, 3, 2, dtype=torch.float64)
        dense_train = train_matrices.flatten(1)
        _, singular_values, vh = torch.linalg.svd(dense_train, full_matrices=False)
        vectors = vh[:2].T
        damping = 1.3
        curvature = CurvatureModel(
            matrix_shape=(3, 2),
            right_vectors=vectors,
            singular_values=singular_values[:2],
            damping=damping,
        )
        actual = influence_scores(
            exact_factors(query_matrices), exact_factors(train_matrices), curvature
        )
        truncated_hessian = (
            vectors @ torch.diag(singular_values[:2].square()) @ vectors.T
            + damping * torch.eye(6, dtype=torch.float64)
        )
        expected = query_matrices.flatten(1) @ torch.linalg.solve(
            truncated_hessian, dense_train.T
        )
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)

    def test_batch_factory_must_repeat_exact_rows_and_shape(self):
        factors = exact_factors(torch.randn(4, 3, 2))

        def too_short():
            yield Factors(factors.left[:3], factors.right[:3])

        with self.assertRaisesRegex(ValueError, "yielded 3 examples"):
            fit_curvature(
                too_short, (3, 2), num_examples=4, rank=2, damping=0.5
            )

        transposed = exact_factors(torch.randn(4, 2, 3))

        def wrong_shape():
            yield transposed

        with self.assertRaisesRegex(ValueError, "matrix shape changed"):
            fit_curvature(
                wrong_shape, (3, 2), num_examples=4, rank=2, damping=0.5
            )

    def test_stable_formula_preserves_small_low_precision_score(self):
        factors = Factors(
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
        )
        curvature = CurvatureModel(
            matrix_shape=(1, 1),
            right_vectors=torch.ones(1, 1, dtype=torch.bfloat16),
            singular_values=torch.tensor([100.0], dtype=torch.bfloat16),
            damping=1.0,
        )
        actual = float(influence_scores(factors, factors, curvature).item())
        self.assertAlmostEqual(actual, 1 / 10_001, delta=2e-6)

    def test_float16_scoring_is_rejected_before_spectrum_overflow(self):
        factors = Factors(
            torch.ones(1, 1, 1, dtype=torch.float16),
            torch.ones(1, 1, 1, dtype=torch.float16),
        )
        curvature = CurvatureModel(
            matrix_shape=(1, 1),
            right_vectors=torch.ones(1, 1, dtype=torch.float16),
            singular_values=torch.tensor([300.0], dtype=torch.float16),
            damping=1.0,
        )
        with self.assertRaisesRegex(ValueError, "float16"):
            influence_scores(factors, factors, curvature)


if __name__ == "__main__":
    unittest.main()
