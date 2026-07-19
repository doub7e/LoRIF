"""Pure tensor operations for LoRIF's per-example low-rank factors."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Factors:
    """A batch of matrices represented as ``left @ right.T``.

    ``left`` has shape ``[batch, rows, rank]`` and ``right`` has shape
    ``[batch, columns, rank]``. The matrix convention follows Eq. (4) of the
    paper: rows are projected input dimensions and columns are projected output
    dimensions.
    """

    left: torch.Tensor
    right: torch.Tensor

    def __post_init__(self) -> None:
        if self.left.ndim != 3 or self.right.ndim != 3:
            raise ValueError("Factors must be rank-3 tensors [batch, dimension, rank]")
        if self.left.shape[0] != self.right.shape[0]:
            raise ValueError("Left and right factors must have the same batch size")
        if self.left.shape[2] != self.right.shape[2]:
            raise ValueError("Left and right factors must have the same rank")
        if self.left.device != self.right.device:
            raise ValueError("Left and right factors must be on the same device")
        if self.left.dtype != self.right.dtype:
            raise ValueError("Left and right factors must have the same dtype")
        if not self.left.dtype.is_floating_point:
            raise ValueError("Factors must use a floating-point dtype")

    @property
    def batch_size(self) -> int:
        return int(self.left.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.left.shape[1]), int(self.right.shape[1])

    @property
    def rank(self) -> int:
        return int(self.left.shape[2])

    def dense(self) -> torch.Tensor:
        """Reconstruct the represented matrices as ``[batch, rows, columns]``."""

        return torch.bmm(self.left, self.right.transpose(1, 2))

    def to(self, *args, **kwargs) -> "Factors":
        return Factors(self.left.to(*args, **kwargs), self.right.to(*args, **kwargs))


def _orthonormalize(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    """Batched thin QR with a stable rank-1 fast path."""

    if matrix.shape[-1] == 1:
        return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def factorize(
    matrices: torch.Tensor,
    rank: int = 1,
    num_iterations: int | None = None,
    *,
    generator: torch.Generator | None = None,
    eps: float | None = None,
) -> Factors:
    """Approximate a matrix batch with block power iteration.

    The default iteration counts (8 for rank 1, 16 otherwise) are those used in
    the paper. Singular values are absorbed into the left factor.
    """

    if matrices.ndim != 3:
        raise ValueError("matrices must have shape [batch, rows, columns]")
    if rank <= 0:
        raise ValueError("rank must be positive")
    batch, rows, columns = matrices.shape
    if rank > min(rows, columns):
        raise ValueError(
            f"rank={rank} exceeds the matrix limit min({rows}, {columns})"
        )
    if num_iterations is None:
        num_iterations = 8 if rank == 1 else 16
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative")
    if eps is None:
        eps = torch.finfo(matrices.dtype).eps

    right = torch.randn(
        batch,
        columns,
        rank,
        device=matrices.device,
        dtype=matrices.dtype,
        generator=generator,
    )
    right = _orthonormalize(right, eps)
    transpose = matrices.transpose(1, 2)

    for _ in range(num_iterations):
        left = _orthonormalize(torch.bmm(matrices, right), eps)
        right = _orthonormalize(torch.bmm(transpose, left), eps)

    left = torch.bmm(matrices, right)
    return Factors(left, right)


def factor_inner_products(query: Factors, train: Factors) -> torch.Tensor:
    """Pairwise Frobenius products between two batches of factorized matrices.

    All cross-component pairs are included, which is essential when rank > 1.
    The result has shape ``[num_queries, num_train]``.
    """

    if query.shape != train.shape:
        raise ValueError(f"Factor shapes differ: {query.shape} != {train.shape}")
    if query.left.device != train.left.device:
        raise ValueError("Query and training factors must be on the same device")

    left_grams = torch.einsum("qia,tib->qtab", query.left, train.left)
    right_grams = torch.einsum("qja,tjb->qtab", query.right, train.right)
    return (left_grams * right_grams).sum(dim=(-1, -2))


def project_factors(factors: Factors, basis: torch.Tensor) -> torch.Tensor:
    """Project vectorized factorized matrices onto a dense basis.

    ``basis`` has shape ``[rows * columns, subspace_rank]`` and uses row-major
    vectorization. The dense matrices are never reconstructed.
    """

    rows, columns = factors.shape
    if basis.ndim != 2 or basis.shape[0] != rows * columns:
        raise ValueError(
            f"basis must have shape [{rows * columns}, r], got {tuple(basis.shape)}"
        )
    if basis.device != factors.left.device:
        raise ValueError("Factors and basis must be on the same device")
    basis_view = basis.reshape(rows, columns, basis.shape[1])
    return torch.einsum(
        "qic,qjc,ijr->qr", factors.left, factors.right, basis_view
    )
