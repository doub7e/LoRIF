import unittest

import torch

from lorif.factors import Factors, factor_inner_products, factorize, project_factors


class FactorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_power_factorization_recovers_low_rank_batch(self):
        left = torch.randn(4, 7, 2)
        right = torch.randn(4, 5, 2)
        matrices = left @ right.transpose(1, 2)
        generator = torch.Generator().manual_seed(11)
        factors = factorize(matrices, rank=2, num_iterations=8, generator=generator)
        relative_error = (factors.dense() - matrices).norm() / matrices.norm()
        self.assertLess(relative_error.item(), 1e-5)

    def test_factor_inner_product_includes_cross_rank_terms(self):
        query = Factors(torch.randn(3, 6, 3), torch.randn(3, 4, 3))
        train = Factors(torch.randn(5, 6, 3), torch.randn(5, 4, 3))
        actual = factor_inner_products(query, train)
        expected = query.dense().flatten(1) @ train.dense().flatten(1).T
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_factor_projection_matches_dense_matmul(self):
        factors = Factors(torch.randn(3, 6, 2), torch.randn(3, 4, 2))
        basis = torch.randn(24, 5)
        actual = project_factors(factors, basis)
        expected = factors.dense().flatten(1) @ basis
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
