#!/usr/bin/env python3
"""A tiny CPU-only LoRIF example with no model or dataset downloads."""

import torch

from lorif import Factors, fit_curvature, influence_scores


def exact_factors(matrices: torch.Tensor) -> Factors:
    identity = torch.eye(matrices.shape[2], dtype=matrices.dtype).expand(
        matrices.shape[0], -1, -1
    )
    return Factors(matrices, identity)


def main() -> None:
    torch.manual_seed(7)
    train = exact_factors(torch.randn(8, 3, 2))
    query = exact_factors(torch.randn(2, 3, 2))

    def train_batches():
        yield Factors(train.left[:4], train.right[:4])
        yield Factors(train.left[4:], train.right[4:])

    curvature = fit_curvature(
        train_batches,
        matrix_shape=(3, 2),
        num_examples=8,
        rank=4,
        damping=0.5,
        seed=0,
    )
    scores = influence_scores(query, train, curvature)
    print("scores shape:", tuple(scores.shape))
    print("top training example per query:", scores.argmax(dim=1).tolist())


if __name__ == "__main__":
    main()
