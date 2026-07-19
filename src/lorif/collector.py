"""Hook-based collection of projected per-example linear-weight gradients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable, Literal

import torch
from torch import nn

from .factors import Factors, factorize


ProjectionNormalization = Literal["legacy", "jl", "none"]


@dataclass(frozen=True)
class LayerSpec:
    name: str
    input_dim: int
    output_dim: int
    projected_input_dim: int
    projected_output_dim: int

    @property
    def projected_shape(self) -> tuple[int, int]:
        return self.projected_input_dim, self.projected_output_dim

    @property
    def projected_dimension(self) -> int:
        return self.projected_input_dim * self.projected_output_dim

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def stable_hash32(value: str) -> int:
    """Stable FNV-1a hash used to derive module-specific random seeds."""

    result = 2166136261
    for char in value:
        result ^= ord(char)
        result = (result * 16777619) & 0xFFFFFFFF
    return int(result & 0x7FFFFFFF)


def rademacher_projection(
    source_dim: int,
    target_dim: int,
    *,
    seed: int,
    normalization: ProjectionNormalization = "legacy",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a deterministic ``[source_dim, target_dim]`` sign projection.

    ``legacy`` uses entries ``+/-1/target_dim`` and the prototype's transposed,
    destination-device RNG stream. ``jl`` uses conventional
    ``+/-1/sqrt(target_dim)`` scaling and a device-independent CPU stream.
    """

    if source_dim <= 0 or target_dim <= 0:
        raise ValueError("Projection dimensions must be positive")
    if normalization not in {"legacy", "jl", "none"}:
        raise ValueError(f"Unknown projection normalization: {normalization}")

    if normalization == "legacy":
        target_device = torch.device(device)
        generator = torch.Generator(device=target_device)
        generator.manual_seed(int(seed) & 0x7FFFFFFF)
        signs = torch.randint(
            0,
            2,
            (target_dim, source_dim),
            generator=generator,
            dtype=torch.int8,
            device=target_device,
        ).transpose(0, 1).contiguous().to(torch.float32)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) & 0x7FFFFFFF)
        signs = torch.randint(
            0,
            2,
            (source_dim, target_dim),
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        ).to(torch.float32)
    signs.mul_(2).sub_(1)
    if normalization == "legacy":
        signs.div_(target_dim)
    elif normalization == "jl":
        signs.div_(math.sqrt(target_dim))
    return signs.to(device=device, dtype=dtype)


class GradientCollector:
    """Collect LoRIF factors for a fixed set of ``torch.nn.Linear`` modules.

    The caller performs the forward and backward passes. A typical batch is::

        collector.clear()
        loss = model(...).loss_sum
        loss.backward()
        factors = collector.pop()

    The loss must be a *sum over examples* (and, for language models, a token
    sum) so each captured row is batch-size invariant. Model weights may remain
    frozen as long as some forward input requires gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str] | None = None,
        *,
        projection_factor: int = 16,
        factor_rank: int = 1,
        factor_iterations: int | None = None,
        projection_seed: int = 0,
        projection_normalization: ProjectionNormalization = "legacy",
        compute_dtype: torch.dtype = torch.float32,
        offload_to_cpu: bool = True,
        strict: bool = True,
    ) -> None:
        if projection_factor <= 0:
            raise ValueError("projection_factor must be positive")
        if factor_rank <= 0:
            raise ValueError("factor_rank must be positive")
        if compute_dtype not in {torch.float32, torch.float64}:
            raise ValueError("compute_dtype must be float32 or float64")

        self.model = model
        self.projection_factor = int(projection_factor)
        self.factor_rank = int(factor_rank)
        self.factor_iterations = factor_iterations
        self.projection_seed = int(projection_seed)
        self.projection_normalization = projection_normalization
        self.compute_dtype = compute_dtype
        self.offload_to_cpu = bool(offload_to_cpu)
        self.strict = strict

        available = dict(model.named_modules())
        if module_names is None:
            selected = [
                name
                for name, module in available.items()
                if isinstance(module, nn.Linear)
            ]
        else:
            selected = list(module_names)
        if not selected:
            raise ValueError("No modules were selected")
        if len(selected) != len(set(selected)):
            raise ValueError("module_names contains duplicates")

        self.specs: dict[str, LayerSpec] = {}
        self._projections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._activations: dict[str, list[torch.Tensor]] = {}
        self._projected_gradients: dict[str, torch.Tensor] = {}
        self._factorized_gradients: dict[str, Factors] = {}
        self._closed = False

        validated: list[tuple[str, nn.Linear, LayerSpec]] = []
        for name in selected:
            module = available.get(name)
            if module is None:
                raise ValueError(f"Unknown module: {name}")
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Selected module {name!r} is not torch.nn.Linear")
            projected_input = max(1, module.in_features // self.projection_factor)
            projected_output = max(1, module.out_features // self.projection_factor)
            if self.factor_rank > min(projected_input, projected_output):
                raise ValueError(
                    f"factor_rank={self.factor_rank} is too large for {name}: "
                    f"projected shape is ({projected_input}, {projected_output})"
                )

            spec = LayerSpec(
                name=name,
                input_dim=module.in_features,
                output_dim=module.out_features,
                projected_input_dim=projected_input,
                projected_output_dim=projected_output,
            )
            self.specs[name] = spec
            validated.append((name, module, spec))

        # Finish all validation and allocate every projection before installing
        # any hook. A constructor error must never leave a partially instrumented
        # model behind.
        for name, module, spec in validated:
            device = module.weight.device
            if self.projection_normalization == "legacy":
                input_seed = self.projection_seed ^ (
                    stable_hash32(f"{name}:P_in") ^ 0xA5A5A5A5
                )
                output_seed = self.projection_seed ^ (
                    stable_hash32(f"{name}:P_out") ^ 0x5A5A5A5A
                )
            else:
                input_seed = self.projection_seed ^ stable_hash32(f"{name}:input")
                output_seed = self.projection_seed ^ stable_hash32(f"{name}:output")
            input_projection = rademacher_projection(
                spec.input_dim,
                spec.projected_input_dim,
                seed=input_seed,
                normalization=self.projection_normalization,
                device=device,
                dtype=self.compute_dtype,
            )
            output_projection = rademacher_projection(
                spec.output_dim,
                spec.projected_output_dim,
                seed=output_seed,
                normalization=self.projection_normalization,
                device=device,
                dtype=self.compute_dtype,
            )
            self._projections[name] = (input_projection, output_projection)
            self._activations[name] = []

        try:
            for name, module, _ in validated:
                self._handles.append(
                    module.register_forward_hook(self._forward_hook(name))
                )
                self._handles.append(
                    module.register_full_backward_hook(self._backward_hook(name))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            raise

    @property
    def module_names(self) -> list[str]:
        return list(self.specs)

    def projection(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and output projection matrices for a layer."""

        return self._projections[name]

    def projection_sha256(self) -> str:
        """Hash the actual ordered projection tensors used by this collector."""

        digest = hashlib.sha256()
        digest.update(b"lorif-projections-v1\0")
        for name in self.module_names:
            digest.update(name.encode())
            digest.update(b"\0")
            for label, tensor in zip(
                (b"input", b"output"), self._projections[name]
            ):
                value = tensor.detach().cpu().contiguous()
                digest.update(label)
                digest.update(b"\0")
                digest.update(str(value.dtype).encode())
                digest.update(b"\0")
                digest.update(str(tuple(value.shape)).encode())
                digest.update(b"\0")
                digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output):
            del module, output
            if not inputs:
                raise RuntimeError(
                    f"Module {name!r} was called without a positional input"
                )
            values = inputs[0]
            spec = self.specs[name]
            if values.shape[-1] != spec.input_dim:
                raise RuntimeError(
                    f"Unexpected input width for {name}: "
                    f"{values.shape[-1]} != {spec.input_dim}"
                )
            with torch.no_grad():
                flat = values.detach().reshape(values.shape[0], -1, spec.input_dim)
                input_projection = self._projections[name][0]
                projected = flat.to(self.compute_dtype) @ input_projection
                self._activations[name].append(projected)

        return hook

    def _backward_hook(self, name: str):
        def hook(module: nn.Module, grad_input, grad_output):
            del module, grad_input
            if not grad_output or grad_output[0] is None:
                raise RuntimeError(f"No output gradient was produced for {name!r}")
            if not self._activations[name]:
                raise RuntimeError(f"No matching forward activation for {name!r}")

            with torch.no_grad():
                activation = self._activations[name].pop()
                delta = grad_output[0].detach()
                spec = self.specs[name]
                if delta.shape[-1] != spec.output_dim:
                    raise RuntimeError(
                        f"Unexpected output width for {name}: "
                        f"{delta.shape[-1]} != {spec.output_dim}"
                    )
                delta = delta.reshape(delta.shape[0], -1, spec.output_dim)
                if activation.shape[:2] != delta.shape[:2]:
                    raise RuntimeError(
                        f"Activation/output positions differ for {name}: "
                        f"{activation.shape[:2]} != {delta.shape[:2]}"
                    )
                output_projection = self._projections[name][1]
                projected_delta = delta.to(self.compute_dtype) @ output_projection
                projected_gradient = torch.bmm(
                    activation.transpose(1, 2), projected_delta
                )
                if name in self._projected_gradients:
                    self._projected_gradients[name].add_(projected_gradient)
                else:
                    self._projected_gradients[name] = projected_gradient

                # A unique transformer layer reaches this branch immediately.
                # Shared modules retain only their partial dense sum until the
                # final invocation, then release it as low-rank factors.
                if not self._activations[name]:
                    matrix = self._projected_gradients.pop(name)
                    generator = torch.Generator(device=matrix.device)
                    generator.manual_seed(
                        self.projection_seed
                        ^ stable_hash32(f"{name}:factorization")
                    )
                    factors = factorize(
                        matrix,
                        rank=self.factor_rank,
                        num_iterations=self.factor_iterations,
                        generator=generator,
                    )
                    if self.offload_to_cpu:
                        factors = factors.to(device="cpu")
                    self._factorized_gradients[name] = factors

        return hook

    def clear(self) -> None:
        """Discard any incomplete forward pass and previous results."""

        for activations in self._activations.values():
            activations.clear()
        self._projected_gradients.clear()
        self._factorized_gradients.clear()

    def pop(self) -> dict[str, Factors]:
        """Factorize and return the gradients captured by the last backward pass."""

        dangling = [name for name, values in self._activations.items() if values]
        if dangling:
            raise RuntimeError(f"Forward activations were not consumed for: {dangling}")
        if self._projected_gradients:
            raise RuntimeError(
                "Some projected gradients were not finalized: "
                f"{sorted(self._projected_gradients)}"
            )
        missing = [
            name for name in self.specs if name not in self._factorized_gradients
        ]
        if self.strict and missing:
            raise RuntimeError(f"No gradient was captured for: {missing}")

        result = self._factorized_gradients
        self._factorized_gradients = {}
        return result

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()
        self._closed = True

    def __enter__(self) -> "GradientCollector":
        if self._closed:
            raise RuntimeError("A closed collector cannot be reused")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
