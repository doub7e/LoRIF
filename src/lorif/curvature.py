"""Streaming spectral curvature approximation for LoRIF."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Iterable, Literal

import torch

from .factors import Factors


CurvatureMode = Literal["tail-corrected", "truncated"]
BatchFactory = Callable[[], Iterable[Factors]]
CURVATURE_FORMAT_VERSION = 2
DEFAULT_DAMPING_MULTIPLIER = 0.5


@dataclass(frozen=True)
class CurvatureModel:
    """Per-layer spectral approximation used for influence scoring."""

    matrix_shape: tuple[int, int]
    right_vectors: torch.Tensor
    singular_values: torch.Tensor
    damping: float
    mode: CurvatureMode = "truncated"
    tail_variance: float = 0.0
    spectrum_trace: float | None = None
    damping_policy: str = "unspecified"

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
        if self.mode not in {"tail-corrected", "truncated"}:
            raise ValueError(f"Unknown curvature mode: {self.mode}")
        if not math.isfinite(self.tail_variance) or self.tail_variance < 0:
            raise ValueError("tail_variance must be finite and non-negative")
        if self.mode == "truncated" and self.tail_variance != 0:
            raise ValueError("truncated mode requires tail_variance=0")
        if self.spectrum_trace is not None and (
            not math.isfinite(self.spectrum_trace) or self.spectrum_trace < 0
        ):
            raise ValueError("spectrum_trace must be finite and non-negative")

    @property
    def rank(self) -> int:
        return int(self.singular_values.numel())

    @property
    def background_damping(self) -> float:
        """Inverse denominator for directions outside the retained subspace."""

        return self.damping + self.tail_variance

    def to(self, *args, **kwargs) -> "CurvatureModel":
        return CurvatureModel(
            matrix_shape=self.matrix_shape,
            right_vectors=self.right_vectors.to(*args, **kwargs),
            singular_values=self.singular_values.to(*args, **kwargs),
            damping=self.damping,
            mode=self.mode,
            tail_variance=self.tail_variance,
            spectrum_trace=self.spectrum_trace,
            damping_policy=self.damping_policy,
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format": "lorif-curvature",
                "format_version": CURVATURE_FORMAT_VERSION,
                "matrix_shape": self.matrix_shape,
                "right_vectors": self.right_vectors.detach().cpu(),
                "singular_values": self.singular_values.detach().cpu(),
                "damping": self.damping,
                "mode": self.mode,
                "tail_variance": self.tail_variance,
                "spectrum_trace": self.spectrum_trace,
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
        version = payload.get("format_version")
        if payload.get("format") != "lorif-curvature" or version not in {1, 2}:
            raise ValueError(f"Unsupported curvature file: {path}")
        vectors = payload["right_vectors"].to(device=device, dtype=dtype)
        values = payload["singular_values"].to(device=device, dtype=dtype)
        if version == 1:
            mode = "truncated"
            tail_variance = 0.0
            spectrum_trace = None
        else:
            mode = str(payload["mode"])
            tail_variance = float(payload["tail_variance"])
            raw_trace = payload.get("spectrum_trace")
            spectrum_trace = None if raw_trace is None else float(raw_trace)
        return cls(
            matrix_shape=tuple(payload["matrix_shape"]),
            right_vectors=vectors,
            singular_values=values,
            damping=float(payload["damping"]),
            mode=mode,
            tail_variance=tail_variance,
            spectrum_trace=spectrum_trace,
            damping_policy=str(payload.get("damping_policy", "unknown")),
        )


def damping_from_trace(
    spectrum_trace: float,
    dimension: int,
    multiplier: float = DEFAULT_DAMPING_MULTIPLIER,
) -> float:
    """Return ``multiplier * trace(G.T @ G) / D``."""

    if not math.isfinite(spectrum_trace) or spectrum_trace <= 0:
        raise ValueError("spectrum_trace must be finite and positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("damping multiplier must be finite and positive")
    result = float(multiplier * spectrum_trace / dimension)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("The estimated damping is not positive; check the gradients")
    return result


def _squared_frobenius_sum(factors: Factors) -> torch.Tensor:
    """Compute ``sum_n ||L_n R_n.T||_F^2`` without a dense square buffer."""

    left_gram = torch.bmm(factors.left.transpose(1, 2), factors.left)
    right_gram = torch.bmm(factors.right.transpose(1, 2), factors.right)
    return (left_gram * right_gram).sum(dtype=torch.float64)


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
    damping_multiplier: float = DEFAULT_DAMPING_MULTIPLIER,
    mode: CurvatureMode = "truncated",
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
    if mode not in {"tail-corrected", "truncated"}:
        raise ValueError(f"Unknown curvature mode: {mode}")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Curvature fitting requires float32 or float64")
    if damping is not None and (
        not math.isfinite(damping) or damping <= 0
    ):
        raise ValueError("damping must be finite and positive")
    if damping is None and (
        not math.isfinite(damping_multiplier) or damping_multiplier <= 0
    ):
        raise ValueError("damping multiplier must be finite and positive")
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
    trace = torch.zeros((), device=device, dtype=torch.float64)
    seen = 0
    for factors in batches():
        if factors.shape != matrix_shape:
            raise ValueError(
                f"Gradient matrix shape changed: {factors.shape} != {matrix_shape}"
            )
        device_factors = factors.to(device=device, dtype=dtype)
        trace.add_(_squared_frobenius_sum(device_factors))
        dense = device_factors.dense().reshape(factors.batch_size, dimension)
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

    trace_value = float(trace.item())
    if damping is None:
        damping_value = damping_from_trace(
            trace_value,
            dimension,
            damping_multiplier,
        )
        policy_name = "trace-mean"
    else:
        damping_value = float(damping)
        policy_name = "explicit"
    vectors = basis @ rotation[:, :actual_rank]
    values = candidate_singular_values[:actual_rank]
    if mode == "tail-corrected" and actual_rank < dimension:
        top_energy = float(values.square().sum(dtype=torch.float64).item())
        residual = max(trace_value - top_energy, 0.0)
        tail_variance = residual / (dimension - actual_rank)
    else:
        tail_variance = 0.0
    return CurvatureModel(
        matrix_shape=matrix_shape,
        right_vectors=vectors,
        singular_values=values,
        damping=damping_value,
        mode=mode,
        tail_variance=tail_variance,
        spectrum_trace=trace_value,
        damping_policy=policy_name,
    )
