"""Woodbury influence scoring from LoRIF factors."""

from __future__ import annotations

import torch

from .curvature import CurvatureModel
from .factors import Factors, factor_inner_products, project_factors


@torch.no_grad()
def influence_scores(
    query: Factors,
    train: Factors,
    curvature: CurvatureModel,
) -> torch.Tensor:
    """Compute one layer's query-by-training influence matrix (paper Eq. 9)."""

    if query.shape != curvature.matrix_shape or train.shape != curvature.matrix_shape:
        raise ValueError(
            "Factor and curvature shapes differ: "
            f"query={query.shape}, train={train.shape}, "
            f"curvature={curvature.matrix_shape}"
        )
    device = query.left.device
    if train.left.device != device or curvature.right_vectors.device != device:
        raise ValueError("Query, training, and curvature tensors must share a device")
    dtype = query.left.dtype
    if train.left.dtype != dtype or curvature.right_vectors.dtype != dtype:
        raise ValueError("Query, training, and curvature tensors must share a dtype")
    if dtype == torch.float16:
        raise ValueError("float16 influence scoring is unsafe; use bfloat16 or float32")

    direct = factor_inner_products(query, train)
    query_projected = project_factors(query, curvature.right_vectors)
    train_projected = project_factors(train, curvature.right_vectors)
    singular_sq = curvature.singular_values.square()

    # Split the inverse into the component orthogonal to V and the component
    # within V. Tail correction changes only the orthogonal denominator:
    # top-r directions still use 1 / (lambda + sigma_i^2).
    # This form also avoids s^2 / (lambda+s^2), which can round to exactly one
    # in FP16/BF16 and erase a valid score.
    projected_direct = query_projected @ train_projected.transpose(0, 1)
    orthogonal = (direct - projected_direct) / curvature.background_damping
    within = (query_projected / (curvature.damping + singular_sq)) @ (
        train_projected.transpose(0, 1)
    )
    return orthogonal + within
