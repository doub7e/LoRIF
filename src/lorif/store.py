"""A small, explicit, shard-based format for LoRIF gradient factors."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from .collector import LayerSpec
from .factors import Factors


FORMAT_VERSION = 1


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class FactorWriter:
    """Write aligned per-layer factor shards and a self-describing manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        specs: Mapping[str, LayerSpec],
        *,
        shard_size: int = 10_000,
        storage_dtype: torch.dtype = torch.bfloat16,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if not specs:
            raise ValueError("At least one layer spec is required")
        if not storage_dtype.is_floating_point:
            raise ValueError("storage_dtype must be floating point")
        self.root = Path(output_dir)
        if self.root.exists():
            if not self.root.is_dir():
                raise FileExistsError(f"Output path is not a directory: {self.root}")
            if any(self.root.iterdir()):
                raise FileExistsError(
                    f"Output directory is not empty: {self.root}. Choose a new path."
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.specs = dict(specs)
        self.shard_size = int(shard_size)
        self.storage_dtype = storage_dtype
        self.user_metadata = dict(metadata or {})
        self.layer_paths = {
            name: f"layers/layer-{index:04d}"
            for index, name in enumerate(self.specs)
        }
        for path in self.layer_paths.values():
            (self.root / path).mkdir(parents=True, exist_ok=True)

        self._buffers: dict[str, list[Factors]] = {name: [] for name in self.specs}
        self._buffered_rows = 0
        self._rows_written = 0
        self._shards: list[dict[str, int]] = []
        self._factor_ranks: dict[str, int | None] = {name: None for name in self.specs}
        self._closed = False

    def write(self, factors: Mapping[str, Factors]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed FactorWriter")
        if set(factors) != set(self.specs):
            missing = sorted(set(self.specs) - set(factors))
            extra = sorted(set(factors) - set(self.specs))
            raise ValueError(f"Layer mismatch; missing={missing}, extra={extra}")
        batch_sizes = {value.batch_size for value in factors.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All layers must contain the same number of examples")
        batch_size = batch_sizes.pop()
        if batch_size <= 0:
            raise ValueError("Cannot write an empty factor batch")

        staged: dict[str, Factors] = {}
        staged_ranks: dict[str, int] = {}
        for name in self.specs:
            value = factors[name]
            if value.shape != self.specs[name].projected_shape:
                raise ValueError(
                    f"Projected shape mismatch for {name}: "
                    f"{value.shape} != {self.specs[name].projected_shape}"
                )
            expected_rank = self._factor_ranks[name]
            if expected_rank is not None and value.rank != expected_rank:
                raise ValueError(
                    f"Factor rank changed for {name}: {value.rank} != {expected_rank}"
                )
            staged_ranks[name] = value.rank
            staged[name] = value.to(device="cpu", dtype=self.storage_dtype)

        # Commit only after every layer has passed validation and conversion.
        for name in self.specs:
            self._factor_ranks[name] = staged_ranks[name]
            self._buffers[name].append(staged[name])
        self._buffered_rows += batch_size

        while self._buffered_rows >= self.shard_size:
            self._flush(self.shard_size)

    def _take(self, name: str, rows: int) -> Factors:
        buffered = self._buffers[name]
        left = torch.cat([item.left for item in buffered], dim=0)
        right = torch.cat([item.right for item in buffered], dim=0)
        taken = Factors(left[:rows].contiguous(), right[:rows].contiguous())
        if rows < left.shape[0]:
            self._buffers[name] = [
                Factors(left[rows:].contiguous(), right[rows:].contiguous())
            ]
        else:
            self._buffers[name] = []
        return taken

    def _flush(self, rows: int) -> None:
        shard_index = len(self._shards)
        start = self._rows_written
        end = start + rows
        for name in self.specs:
            value = self._take(name, rows)
            path = self.root / self.layer_paths[name] / f"shard-{shard_index:05d}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save({"left": value.left, "right": value.right}, temporary)
            os.replace(temporary, path)
        self._shards.append({"index": shard_index, "start": start, "end": end})
        self._rows_written = end
        self._buffered_rows -= rows

    def close(self) -> None:
        if self._closed:
            return
        if self._rows_written == 0 and self._buffered_rows == 0:
            raise ValueError("Cannot close an empty FactorWriter")
        if self._buffered_rows:
            self._flush(self._buffered_rows)

        layers = []
        for name, spec in self.specs.items():
            entry = asdict(spec)
            entry["path"] = self.layer_paths[name]
            entry["factor_rank"] = self._factor_ranks[name]
            layers.append(entry)
        manifest = {
            "format": "lorif-factors",
            "format_version": FORMAT_VERSION,
            "num_examples": self._rows_written,
            "storage_dtype": _dtype_name(self.storage_dtype),
            "shards": self._shards,
            "layers": layers,
            "metadata": self.user_metadata,
        }
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.root / "manifest.json")
        self._closed = True

    def __enter__(self) -> "FactorWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


class FactorStore:
    """Read a factor index without loading it fully into memory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing factor manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "lorif-factors":
            raise ValueError(f"Not a LoRIF factor store: {manifest_path}")
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "Unsupported factor format version: "
                f"{self.manifest.get('format_version')}"
            )
        self._layers = {entry["name"]: entry for entry in self.manifest["layers"]}

    @property
    def num_examples(self) -> int:
        return int(self.manifest["num_examples"])

    @property
    def num_shards(self) -> int:
        return len(self.manifest["shards"])

    @property
    def module_names(self) -> list[str]:
        return list(self._layers)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.manifest.get("metadata", {}))

    def spec(self, name: str) -> LayerSpec:
        entry = self._layers[name]
        return LayerSpec(
            name=entry["name"],
            input_dim=int(entry["input_dim"]),
            output_dim=int(entry["output_dim"]),
            projected_input_dim=int(entry["projected_input_dim"]),
            projected_output_dim=int(entry["projected_output_dim"]),
        )

    def shard_range(self, index: int) -> tuple[int, int]:
        shard = self.manifest["shards"][index]
        return int(shard["start"]), int(shard["end"])

    def _shard_path(self, name: str, index: int) -> Path:
        return self.root / self._layers[name]["path"] / f"shard-{index:05d}.pt"

    def load_shard(
        self,
        name: str,
        index: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        mmap: bool = False,
    ) -> Factors:
        path = self._shard_path(name, index)
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
        if not isinstance(payload, dict) or set(payload) != {"left", "right"}:
            raise ValueError(f"Malformed factor shard: {path}")
        factors = Factors(payload["left"], payload["right"])
        spec = self.spec(name)
        if factors.shape != spec.projected_shape:
            raise ValueError(
                f"Projected shape mismatch in {path}: "
                f"{factors.shape} != {spec.projected_shape}"
            )
        expected_rank = self._layers[name].get("factor_rank")
        if expected_rank is not None and factors.rank != int(expected_rank):
            raise ValueError(
                f"Factor rank mismatch in {path}: {factors.rank} != {expected_rank}"
            )
        start, end = self.shard_range(index)
        if factors.batch_size != end - start:
            raise ValueError(
                f"Shard row mismatch in {path}: {factors.batch_size} != {end - start}"
            )
        if dtype is not None:
            return factors.to(device=device, dtype=dtype)
        return factors.to(device)

    def iter_layer(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        mmap: bool = False,
    ) -> Iterator[Factors]:
        for index in range(self.num_shards):
            yield self.load_shard(
                name, index, device=device, dtype=dtype, mmap=mmap
            )

    def load_layer(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> Factors:
        shards = list(self.iter_layer(name, device=device, dtype=dtype))
        if not shards:
            raise ValueError(f"Layer {name!r} has no factor shards")
        return Factors(
            torch.cat([item.left for item in shards], dim=0),
            torch.cat([item.right for item in shards], dim=0),
        )
