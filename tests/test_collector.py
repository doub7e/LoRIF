import unittest

import torch
from torch import nn

from lorif.collector import (
    GradientCollector,
    rademacher_projection,
    stable_hash32,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, values):
        return self.linear(values)


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()


class CollectorTests(unittest.TestCase):
    def test_projected_per_example_gradient_matches_autograd(self):
        torch.manual_seed(3)
        model = ToyModel().double()
        values = torch.randn(3, 5, 4, dtype=torch.float64)
        targets = torch.randn(3, 5, 3, dtype=torch.float64)

        exact = []
        for index in range(values.shape[0]):
            output = model(values[index : index + 1])
            loss = (output - targets[index : index + 1]).square().sum() / 2
            gradient = torch.autograd.grad(loss, model.linear.weight)[0]
            exact.append(gradient.transpose(0, 1))  # paper input-by-output layout
        exact = torch.stack(exact)

        with GradientCollector(
            model,
            ["linear"],
            projection_factor=1,
            factor_rank=3,
            factor_iterations=2,
            projection_seed=19,
            compute_dtype=torch.float64,
        ) as collector:
            input_projection, output_projection = collector.projection("linear")
            collector.clear()
            output = model(values.requires_grad_(True))
            loss = (output - targets).square().sum() / 2
            loss.backward()
            factors = collector.pop()["linear"]

        expected = input_projection.T @ exact @ output_projection
        torch.testing.assert_close(factors.dense(), expected, rtol=1e-8, atol=1e-8)

    def test_legacy_projection_has_experiment_scaling(self):
        projection = rademacher_projection(
            9, 3, seed=4, normalization="legacy", dtype=torch.float64
        )
        torch.testing.assert_close(
            projection.abs(), torch.full_like(projection, 1 / 3)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(4)
        expected = torch.randint(
            0, 2, (3, 9), generator=generator, dtype=torch.int8
        ).transpose(0, 1).double()
        expected = expected.mul(2).sub(1).div(3)
        torch.testing.assert_close(projection, expected)

        model = ToyModel()
        with GradientCollector(
            model,
            ["linear"],
            projection_factor=1,
            projection_seed=0,
        ) as collector:
            actual_input = collector.projection("linear")[0]
        prototype_seed = (
            stable_hash32("linear:P_in") ^ 0xA5A5A5A5
        ) & 0x7FFFFFFF
        prototype_generator = torch.Generator(device="cpu")
        prototype_generator.manual_seed(prototype_seed)
        expected_input = torch.randint(
            0,
            2,
            (4, 4),
            generator=prototype_generator,
            dtype=torch.int8,
        ).transpose(0, 1).float()
        expected_input = expected_input.mul(2).sub(1).div(4)
        torch.testing.assert_close(actual_input, expected_input)

        first = GradientCollector(model, ["linear"], projection_seed=0)
        second = GradientCollector(model, ["linear"], projection_seed=0)
        different = GradientCollector(model, ["linear"], projection_seed=1)
        try:
            self.assertEqual(first.projection_sha256(), second.projection_sha256())
            self.assertNotEqual(
                first.projection_sha256(), different.projection_sha256()
            )
        finally:
            first.close()
            second.close()
            different.close()

    def test_constructor_error_does_not_leave_hooks(self):
        model = MixedModel()
        with self.assertRaises(TypeError):
            GradientCollector(model, ["linear", "relu"])
        self.assertEqual(len(model.linear._forward_hooks), 0)
        self.assertEqual(len(model.linear._backward_hooks), 0)

    def test_shared_module_calls_are_summed_before_factorization(self):
        torch.manual_seed(17)
        model = ToyModel().double()
        values = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        with GradientCollector(
            model,
            ["linear"],
            projection_factor=1,
            factor_rank=3,
            factor_iterations=2,
            compute_dtype=torch.float64,
        ) as collector:
            input_projection, output_projection = collector.projection("linear")
            collector.clear()
            loss = (model(values) + model(2 * values)).square().sum() / 2
            loss.backward()
            actual = collector.pop()["linear"].dense()

        exact = []
        detached = values.detach()
        for index in range(detached.shape[0]):
            sample = detached[index : index + 1].requires_grad_(True)
            loss = (model(sample) + model(2 * sample)).square().sum() / 2
            gradient = torch.autograd.grad(loss, model.linear.weight)[0]
            exact.append(gradient.transpose(0, 1))
        expected = input_projection.T @ torch.stack(exact) @ output_projection
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
