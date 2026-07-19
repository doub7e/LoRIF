"""Linear Datamodeling Score (LDS) with an explicit utility sign convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
import torch


@dataclass(frozen=True)
class LDSResult:
    mean: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float
    per_query: np.ndarray
    num_valid_queries: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lds": self.mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_half_width": self.ci_half_width,
            "num_valid_queries": self.num_valid_queries,
        }


def linear_datamodeling_score(
    scores: torch.Tensor,
    subset_membership: torch.Tensor,
    actual_utilities: torch.Tensor | np.ndarray,
    *,
    num_bootstrap: int = 5_000,
    bootstrap_seed: int = 12_345,
    confidence: float = 0.95,
) -> LDSResult:
    """Evaluate whether additive influence predicts subset-trained utilities.

    Args:
        scores: Influence scores with shape ``[queries, training_examples]``.
        subset_membership: Boolean/0-1 matrix ``[subsets, training_examples]``.
        actual_utilities: Matrix ``[subsets, queries]`` where larger is better.
            For cross-entropy evaluation pass *negative* loss. This explicit
            convention fixes the sign inconsistency in the legacy evaluator.
    """

    if scores.ndim != 2 or subset_membership.ndim != 2:
        raise ValueError("scores and subset_membership must both be matrices")
    num_queries, num_train = scores.shape
    num_subsets, membership_width = subset_membership.shape
    if membership_width != num_train:
        raise ValueError(
            f"Membership width {membership_width} != score width {num_train}"
        )
    if subset_membership.dtype != torch.bool:
        binary = torch.all(
            (subset_membership == 0) | (subset_membership == 1)
        ).item()
        if not binary:
            raise ValueError("subset_membership must contain only 0/1 values")
    if isinstance(actual_utilities, torch.Tensor):
        utilities = actual_utilities.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
    else:
        utilities = np.asarray(actual_utilities, dtype=np.float64)
    if utilities.shape != (num_subsets, num_queries):
        raise ValueError(
            f"actual_utilities must have shape {(num_subsets, num_queries)}, "
            f"got {utilities.shape}"
        )
    if num_subsets < 2:
        raise ValueError("LDS requires at least two training subsets")
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")

    membership = subset_membership.to(device=scores.device, dtype=scores.dtype)
    predicted = (scores @ membership.transpose(0, 1)).detach().cpu().numpy()

    correlations = np.full(num_queries, np.nan, dtype=np.float64)
    for query_index in range(num_queries):
        correlation = spearmanr(
            utilities[:, query_index], predicted[query_index], nan_policy="omit"
        ).statistic
        if np.isfinite(correlation):
            correlations[query_index] = float(correlation)
    valid = correlations[np.isfinite(correlations)]
    if valid.size == 0:
        raise ValueError("No valid per-query correlations were produced")

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, valid.size, size=(num_bootstrap, valid.size))
    bootstrap_means = valid[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    mean = float(valid.mean())
    return LDSResult(
        mean=mean,
        ci_lower=float(lower),
        ci_upper=float(upper),
        ci_half_width=float((upper - lower) / 2),
        per_query=correlations,
        num_valid_queries=int(valid.size),
    )
