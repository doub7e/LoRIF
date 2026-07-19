#!/usr/bin/env python3
"""Small optional CUDA validation; run explicitly on a GPU worker."""

import torch
from torch import nn

from lorif import (
    CurvatureModel,
    Factors,
    GradientCollector,
    fit_curvature,
    influence_scores,
)
from lorif.collector import stable_hash32


def exact_factors(matrices: torch.Tensor) -> Factors:
    identity = torch.eye(
        matrices.shape[2], device=matrices.device, dtype=matrices.dtype
    ).expand(matrices.shape[0], -1, -1)
    return Factors(matrices, identity)


def legacy_projection(
    source_dim: int, target_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed & 0x7FFFFFFF)
    values = torch.randint(
        0,
        2,
        (target_dim, source_dim),
        generator=generator,
        dtype=torch.int8,
        device=device,
    )
    return values.float().mul(2).sub(1).div(target_dim).transpose(0, 1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    device = torch.device("cuda")

    model = nn.Sequential()
    model.add_module("layer", nn.Linear(9, 6, bias=False, device=device))
    with GradientCollector(
        model,
        ["layer"],
        projection_factor=3,
        projection_seed=0,
    ) as collector:
        input_projection, output_projection = collector.projection("layer")
        cuda_projection_digest = collector.projection_sha256()
    input_seed = (stable_hash32("layer:P_in") ^ 0xA5A5A5A5) & 0x7FFFFFFF
    output_seed = (stable_hash32("layer:P_out") ^ 0x5A5A5A5A) & 0x7FFFFFFF
    torch.testing.assert_close(
        input_projection, legacy_projection(9, 3, input_seed, device)
    )
    torch.testing.assert_close(
        output_projection, legacy_projection(6, 2, output_seed, device)
    )
    cpu_model = nn.Sequential()
    cpu_model.add_module("layer", nn.Linear(9, 6, bias=False))
    with GradientCollector(
        cpu_model,
        ["layer"],
        projection_factor=3,
        projection_seed=0,
    ) as cpu_collector:
        if cuda_projection_digest == cpu_collector.projection_sha256():
            raise AssertionError("CPU/CUDA legacy projection digests should differ")

    torch.manual_seed(23)
    train_matrices = torch.randn(7, 2, 2, device=device)
    query_matrices = torch.randn(3, 2, 2, device=device)
    train = exact_factors(train_matrices)
    query = exact_factors(query_matrices)

    def batches():
        yield Factors(train.left[:3], train.right[:3])
        yield Factors(train.left[3:], train.right[3:])

    damping = 0.7
    curvature = fit_curvature(
        batches,
        (2, 2),
        num_examples=7,
        rank=4,
        oversample=0,
        num_power_iterations=1,
        damping=damping,
        device=device,
    )
    actual = influence_scores(query, train, curvature)
    dense_train = train_matrices.flatten(1)
    dense_query = query_matrices.flatten(1)
    hessian = dense_train.T @ dense_train + damping * torch.eye(4, device=device)
    expected = dense_query @ torch.linalg.solve(hessian, dense_train.T)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    factors = Factors(
        torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16),
    )
    low_precision = CurvatureModel(
        matrix_shape=(1, 1),
        right_vectors=torch.ones(1, 1, device=device, dtype=torch.bfloat16),
        singular_values=torch.tensor(
            [100.0], device=device, dtype=torch.bfloat16
        ),
        damping=1.0,
    )
    score = influence_scores(factors, factors, low_precision).float()
    torch.testing.assert_close(
        score, torch.tensor([[1 / 10_001]], device=device), rtol=0.02, atol=1e-6
    )

    print("GPU smoke test passed")


if __name__ == "__main__":
    main()
