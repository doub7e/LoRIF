"""Streaming truncated-SVD curvature approximation for LoRIF."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Literal

import torch

from .factors import Factors


DampingPolicy = Literal["paper", "legacy-gpt2"]
BatchFactory = Callable[[], Iterable[Factors]]


@dataclass(frozen=True)
class CurvatureModel:
    """Per-layer right-singular subspace and damping used by Eq. (9)."""

    matrix_shape: tuple[int, int]
    right_vectors: torch.Tensor
    singular_values: torch.Tensor
    damping: float
    damping_policy: str = "paper"

    def __post_init__(self) -> None:
        rows, columns = self.matrix_shape
        if rows <= 0 or columns <= 0:
            raise ValueError("matrix_shape must be positive")
        if self.right_vectors.ndim != 2:
            raise ValueError("right_vectors must have shape [dimension, rank]")
        if self.singular_values.ndim != 1:
            raise ValueError("singular_values must have shape [rank]")
        if self.right_vectors.shape != (rows * columns, self.singular_values.numel()):
            raise ValueError(
                "right_vectors, singular_values, and matrix_shape disagree"
            )
        if self.right_vectors.device != self.singular_values.device:
            raise ValueError("Curvature tensors must be on the same device")
        if self.right_vectors.dtype != self.singular_values.dtype:
            raise ValueError("Curvature tensors must have the same dtype")
        if not self.right_vectors.dtype.is_floating_point:
            raise ValueError("Curvature tensors must use a floating-point dtype")
        if not math.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("damping must be finite and positive")

    @property
    def rank(self) -> int:
        return int(self.singular_values.numel())

    def to(self, *args, **kwargs) -> "CurvatureModel":
        return CurvatureModel(
            matrix_shape=self.matrix_shape,
            right_vectors=self.right_vectors.to(*args, **kwargs),
            singular_values=self.singular_values.to(*args, **kwargs),
            damping=self.damping,
            damping_policy=self.damping_policy,
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format": "lorif-curvature",
                "format_version": 1,
                "matrix_shape": self.matrix_shape,
                "right_vectors": self.right_vectors.detach().cpu(),
                "singular_values": self.singular_values.detach().cpu(),
                "damping": self.damping,
                "damping_policy": self.damping_policy,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "CurvatureModel":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "lorif-curvature"
            or payload.get("format_version") != 1
        ):
            raise ValueError(f"Unsupported curvature file: {path}")
        vectors = payload["right_vectors"].to(device=device, dtype=dtype)
        values = payload["singular_values"].to(device=device, dtype=dtype)
        return cls(
            matrix_shape=tuple(payload["matrix_shape"]),
            right_vectors=vectors,
            singular_values=values,
            damping=float(payload["damping"]),
            damping_policy=str(payload.get("damping_policy", "unknown")),
        )


def damping_from_spectrum(
    singular_values: torch.Tensor,
    *,
    target_rank: int,
    policy: DampingPolicy = "paper",
) -> float:
    """Compute layer damping from the randomized-SVD candidate spectrum."""

    if singular_values.ndim != 1 or singular_values.numel() == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if target_rank <= 0 or target_rank > singular_values.numel():
        raise ValueError("target_rank is outside the available spectrum")
    eigenvalues = singular_values.square()
    if policy == "paper":
        damping = 0.1 * eigenvalues.mean()
    elif policy == "legacy-gpt2":
        damping = 0.01 * math.log2(target_rank) * eigenvalues[:target_rank].mean()
    else:
        raise ValueError(f"Unknown damping policy: {policy}")
    result = float(damping.item())
    if not math.isfinite(result) or result <= 0:
        raise ValueError("The estimated damping is not positive; check the gradients")
    return result


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def _covariance_multiply(
    batches: BatchFactory,
    basis: torch.Tensor,
    *,
    matrix_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    expected_examples: int,
) -> torch.Tensor:
    result = torch.zeros_like(basis)
    seen = 0
    for factors in batches():
        if factors.shape != matrix_shape:
            raise ValueError(
                f"Gradient matrix shape changed: {factors.shape} != {matrix_shape}"
            )
        dense = factors.to(device=device, dtype=dtype).dense().reshape(
            factors.batch_size, -1
        )
        if dense.shape[1] != basis.shape[0]:
            raise ValueError(
                f"Gradient dimension changed: {dense.shape[1]} != {basis.shape[0]}"
            )
        projected = dense @ basis
        result.add_(dense.transpose(0, 1) @ projected)
        seen += factors.batch_size
    if seen != expected_examples:
        raise ValueError(
            f"Batch factory yielded {seen} examples; expected {expected_examples}"
        )
    return result


@torch.no_grad()
def fit_curvature(
    batches: BatchFactory,
    matrix_shape: tuple[int, int],
    *,
    num_examples: int,
    rank: int,
    oversample: int = 10,
    num_power_iterations: int = 3,
    damping: float | None = None,
    damping_policy: DampingPolicy = "paper",
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CurvatureModel:
    """Fit a randomized right-singular subspace without materializing all rows.

    The implementation applies one initial covariance multiply followed by
    ``num_power_iterations`` subspace iterations, matching the paper's three
    power iterations plus initial multiply.
    """

    rows, columns = matrix_shape
    dimension = rows * columns
    if rows <= 0 or columns <= 0 or num_examples <= 0:
        raise ValueError("matrix_shape and num_examples must be positive")
    if rank <= 0 or oversample < 0 or num_power_iterations < 0:
        raise ValueError("rank must be positive; oversample/iterations non-negative")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Curvature fitting requires float32 or float64")
    actual_rank = min(rank, dimension, num_examples)
    candidate_rank = min(actual_rank + oversample, dimension, num_examples)
    device = torch.device(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    basis = torch.randn(
        dimension,
        candidate_rank,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    basis = _orthonormalize(
        _covariance_multiply(
            batches,
            basis,
            matrix_shape=matrix_shape,
            device=device,
            dtype=dtype,
            expected_examples=num_examples,
        )
    )
    for _ in range(num_power_iterations):
        basis = _orthonormalize(
            _covariance_multiply(
                batches,
                basis,
                matrix_shape=matrix_shape,
                device=device,
                dtype=dtype,
                expected_examples=num_examples,
            )
        )

    rayleigh = torch.zeros(
        candidate_rank, candidate_rank, device=device, dtype=dtype
    )
    seen = 0
    for factors in batches():
        if factors.shape != matrix_shape:
            raise ValueError(
                f"Gradient matrix shape changed: {factors.shape} != {matrix_shape}"
            )
        dense = factors.to(device=device, dtype=dtype).dense().reshape(
            factors.batch_size, dimension
        )
        projected = dense @ basis
        rayleigh.add_(projected.transpose(0, 1) @ projected)
        seen += factors.batch_size
    if seen != num_examples:
        raise ValueError(
            f"Batch factory yielded {seen} examples; expected {num_examples}"
        )
    rayleigh = 0.5 * (rayleigh + rayleigh.transpose(0, 1))
    eigenvalues, rotation = torch.linalg.eigh(rayleigh)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    rotation = rotation[:, order]
    candidate_singular_values = eigenvalues.sqrt()

    if damping is None:
        damping_value = damping_from_spectrum(
            candidate_singular_values,
            target_rank=actual_rank,
            policy=damping_policy,
        )
        policy_name = damping_policy
    else:
        damping_value = float(damping)
        policy_name = "explicit"
    vectors = basis @ rotation[:, :actual_rank]
    values = candidate_singular_values[:actual_rank]
    return CurvatureModel(
        matrix_shape=matrix_shape,
        right_vectors=vectors,
        singular_values=values,
        damping=damping_value,
        damping_policy=policy_name,
    )
