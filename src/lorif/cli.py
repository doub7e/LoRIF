"""Command-line pipeline for fitting, scoring, and evaluating LoRIF indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import torch

from .curvature import (
    DEFAULT_DAMPING_MULTIPLIER,
    CurvatureModel,
    fit_curvature,
)
from .influence import influence_scores
from .lds import linear_datamodeling_score
from .store import FactorStore


CURVATURE_INDEX_FORMAT_VERSION = 2
SCORE_FORMAT_VERSION = 2


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _dtype(value: str) -> torch.dtype:
    options = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    return options[value]


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _manifest_hash(store: FactorStore) -> str:
    data = (store.root / "manifest.json").read_bytes()
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _curvature_summary(
    name: str,
    model: CurvatureModel,
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    summary = {
        "name": name,
        "matrix_shape": list(model.matrix_shape),
        "rank": model.rank,
        "damping": model.damping,
        "damping_policy": model.damping_policy,
        "mode": model.mode,
        "tail_variance": model.tail_variance,
        "spectrum_trace": model.spectrum_trace,
        "background_damping": model.background_damping,
    }
    if filename is not None:
        summary["file"] = filename
    return summary


def fit_curvature_command(args: argparse.Namespace) -> None:
    store = FactorStore(args.index)
    output = _prepare_output(args.output)
    device = _device(args.device)
    dtype = _dtype(args.dtype)
    layers = []

    for layer_index, name in enumerate(store.module_names):
        spec = store.spec(name)
        print(
            f"[{layer_index + 1}/{len(store.module_names)}] {name}: "
            f"shape={spec.projected_shape}, rank={args.rank}, mode={args.mode}"
        )

        def batches(layer_name=name):
            return store.iter_layer(
                layer_name, device=device, dtype=dtype, mmap=args.mmap
            )

        model = fit_curvature(
            batches,
            spec.projected_shape,
            num_examples=store.num_examples,
            rank=args.rank,
            oversample=args.oversample,
            num_power_iterations=args.power_iterations,
            damping=args.damping,
            damping_multiplier=args.damping_multiplier,
            mode=args.mode,
            seed=args.seed + layer_index,
            device=device,
            dtype=dtype,
        )
        filename = f"layer-{layer_index:04d}.pt"
        model.save(output / filename)
        layers.append(_curvature_summary(name, model, filename=filename))

    manifest = {
        "format": "lorif-curvature-index",
        "format_version": CURVATURE_INDEX_FORMAT_VERSION,
        "source_factor_manifest_sha256": _manifest_hash(store),
        "layers": layers,
        "config": {
            "requested_rank": args.rank,
            "oversample": args.oversample,
            "power_iterations": args.power_iterations,
            "damping": args.damping,
            "damping_multiplier": (
                None if args.damping is not None else args.damping_multiplier
            ),
            "damping_policy": layers[0]["damping_policy"],
            "mode": args.mode,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
        },
    }
    _write_json_atomic(output / "manifest.json", manifest)
    print(f"Saved curvature index to {output}")


def _load_curvature_index(
    path: str | Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format") != "lorif-curvature-index"
        or manifest.get("format_version") not in {1, 2}
    ):
        raise ValueError(f"Unsupported curvature index: {manifest_path}")
    return root, {entry["name"]: entry for entry in manifest["layers"]}, manifest


def _validate_indexes(train: FactorStore, query: FactorStore) -> None:
    if train.module_names != query.module_names:
        raise ValueError("Training and query indexes have different ordered layer sets")
    for name in train.module_names:
        if train.spec(name) != query.spec(name):
            raise ValueError(f"Training/query layer specs differ for {name}")
    keys = {
        "model",
        "model_revision",
        "model_revision_resolved",
        "model_fingerprint",
        "tokenizer",
        "tokenizer_revision",
        "tokenizer_revision_resolved",
        "pad_token_id",
        "projection_factor",
        "factor_rank",
        "factor_iterations",
        "projection_seed",
        "projection_normalization",
        "projection_rng",
        "projection_sha256",
        "factor_layout",
        "loss_reduction",
    }
    for key in keys:
        if key not in train.metadata or key not in query.metadata:
            raise ValueError(
                f"Training/query indexes are missing required metadata: {key}"
            )
        left = train.metadata[key]
        right = query.metadata[key]
        if left != right:
            raise ValueError(
                f"Training/query metadata differs for {key}: "
                f"{left!r} != {right!r}"
            )


def score_command(args: argparse.Namespace) -> None:
    train = FactorStore(args.train_index)
    query = FactorStore(args.query_index)
    _validate_indexes(train, query)
    curvature_root = None
    curvature_entries: dict[str, dict[str, Any]] = {}
    curvature_manifest: dict[str, Any] | None = None
    if args.curvature is not None:
        (
            curvature_root,
            curvature_entries,
            curvature_manifest,
        ) = _load_curvature_index(args.curvature)
        if set(curvature_entries) != set(train.module_names):
            raise ValueError("Curvature and factor indexes contain different layers")
        expected_source = _manifest_hash(train)
        actual_source = curvature_manifest.get("source_factor_manifest_sha256")
        if actual_source != expected_source:
            raise ValueError(
                "Curvature index was not fitted from this training factor index: "
                f"{actual_source!r} != {expected_source!r}"
            )

    output = _prepare_output(args.output)
    device = _device(args.device)
    dtype = _dtype(args.dtype)
    inline_curvature = curvature_root is None
    fit_dtype = _dtype(args.fit_dtype) if inline_curvature else None
    curvature_summaries = []
    shards = [
        {
            "index": shard_index,
            "start": train.shard_range(shard_index)[0],
            "end": train.shard_range(shard_index)[1],
            "file": f"shard-{shard_index:05d}.pt",
        }
        for shard_index in range(train.num_shards)
    ]
    # Process one layer at a time: a paper-scale D x r curvature matrix can be
    # several GiB. By default it is fitted, consumed, and released here without
    # ever being serialized.
    for layer_index, name in enumerate(train.module_names):
        if inline_curvature:
            spec = train.spec(name)
            print(
                f"[{layer_index + 1}/{len(train.module_names)}] "
                f"fitting and scoring layer {name}: "
                f"shape={spec.projected_shape}, rank={args.rank}, mode={args.mode}"
            )

            def batches(layer_name=name):
                return train.iter_layer(
                    layer_name,
                    device=device,
                    dtype=fit_dtype,
                    mmap=args.mmap,
                )

            curvature = fit_curvature(
                batches,
                spec.projected_shape,
                num_examples=train.num_examples,
                rank=args.rank,
                oversample=args.oversample,
                num_power_iterations=args.power_iterations,
                damping=args.damping,
                damping_multiplier=args.damping_multiplier,
                mode=args.mode,
                seed=args.seed + layer_index,
                device=device,
                dtype=fit_dtype,
            )
            curvature_summaries.append(_curvature_summary(name, curvature))
            curvature = curvature.to(device=device, dtype=dtype)
        else:
            print(
                f"[{layer_index + 1}/{len(train.module_names)}] "
                f"scoring layer {name} from cached curvature"
            )
            curvature = CurvatureModel.load(
                curvature_root / curvature_entries[name]["file"],
                device=device,
                dtype=dtype,
            )

        query_factors = query.load_layer(name, device=device, dtype=dtype)
        for shard in shards:
            shard_index = shard["index"]
            start, end = shard["start"], shard["end"]
            train_factors = train.load_shard(
                name,
                shard_index,
                device=device,
                dtype=dtype,
                mmap=args.mmap,
            )
            contribution = influence_scores(
                query_factors, train_factors, curvature
            ).float().cpu()
            path = output / shard["file"]
            if layer_index:
                contribution.add_(_torch_load(path))
            temporary = path.with_suffix(".pt.tmp")
            torch.save(contribution, temporary)
            os.replace(temporary, path)
            del contribution, train_factors
        del curvature, query_factors

    # The layer-major loop above bounds accelerator memory by one layer. Score
    # shards are atomically accumulated on disk between layers.
    for shard_index, shard in enumerate(shards):
        start, end = shard["start"], shard["end"]
        print(
            f"[{shard_index + 1}/{len(shards)}] "
            f"finalized train rows [{start}, {end})"
        )

    if inline_curvature:
        curvature_source = "inline"
        curvature_mode = args.mode
        curvature_config = {
            "requested_rank": args.rank,
            "oversample": args.oversample,
            "power_iterations": args.power_iterations,
            "damping": args.damping,
            "damping_multiplier": (
                None if args.damping is not None else args.damping_multiplier
            ),
            "damping_policy": (
                "explicit" if args.damping is not None else "trace-mean"
            ),
            "mode": args.mode,
            "seed": args.seed,
            "fit_dtype": args.fit_dtype,
            "device": str(device),
        }
    else:
        curvature_source = "index"
        curvature_mode = curvature_manifest.get("config", {}).get(
            "mode",
            "truncated"
            if curvature_manifest.get("format_version") == 1
            else "unknown",
        )
        curvature_config = curvature_manifest.get("config", {})

    manifest = {
        "format": "lorif-scores",
        "format_version": SCORE_FORMAT_VERSION,
        "num_queries": query.num_examples,
        "num_train": train.num_examples,
        "train_factor_manifest_sha256": _manifest_hash(train),
        "query_factor_manifest_sha256": _manifest_hash(query),
        "curvature_source": curvature_source,
        "curvature_mode": curvature_mode,
        "curvature_config": curvature_config,
        "shards": shards,
        "dtype": "float32",
        "score_dtype": args.dtype,
        "device": str(device),
    }
    if inline_curvature:
        manifest["curvature_layers"] = curvature_summaries
    else:
        manifest["curvature_manifest_sha256"] = _file_hash(
            curvature_root / "manifest.json"
        )
    _write_json_atomic(output / "manifest.json", manifest)
    print(f"Saved score shards to {output}")


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_scores(path: str | Path, device: torch.device) -> torch.Tensor:
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("format") != "lorif-scores":
            raise ValueError(f"Not a LoRIF score directory: {path}")
        pieces = [_torch_load(path / shard["file"]) for shard in manifest["shards"]]
        scores = torch.cat(pieces, dim=1)
    else:
        scores = _torch_load(path)
    if not isinstance(scores, torch.Tensor):
        raise TypeError("Scores must be a torch Tensor")
    if scores.ndim == 3:
        scores = scores.sum(dim=-1)
    if scores.ndim != 2:
        raise ValueError(f"Scores must have shape [queries, train], got {scores.shape}")
    return scores.to(device=device, dtype=torch.float32)


def _numeric_suffix(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else -1, path.name)


def _load_subset_indices(directory: Path) -> list[list[int]]:
    files = sorted(directory.glob("*.json"), key=_numeric_suffix)
    if not files:
        raise FileNotFoundError(f"No subset JSON files found in {directory}")
    subsets = []
    for path in files:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(
            isinstance(item, int) for item in values
        ):
            raise ValueError(f"Expected a JSON integer list: {path}")
        subsets.append(values)
    return subsets


def _load_membership(path: str | Path, num_train: int) -> torch.Tensor:
    path = Path(path)
    if path.is_dir():
        subsets = _load_subset_indices(path)
        membership = torch.zeros(len(subsets), num_train, dtype=torch.bool)
        for row, indices in enumerate(subsets):
            index = torch.tensor(indices, dtype=torch.long)
            if index.numel() and (index.min() < 0 or index.max() >= num_train):
                raise ValueError(f"Subset {row} has an index outside [0, {num_train})")
            membership[row, index] = True
        return membership
    if path.suffix == ".npz":
        archive = np.load(path)
        if "membership" in archive:
            values = archive["membership"].astype(bool, copy=False)
        elif "packed_membership" in archive:
            stored_width = int(np.asarray(archive["num_train"]).item())
            if stored_width != num_train:
                raise ValueError(
                    f"Packed membership width {stored_width} != {num_train}"
                )
            values = np.unpackbits(
                archive["packed_membership"],
                axis=1,
                count=num_train,
                bitorder="little",
            ).astype(bool, copy=False)
        else:
            raise ValueError("NPZ needs membership or packed_membership")
        return torch.from_numpy(values)
    values = _torch_load(path)
    if not isinstance(values, torch.Tensor):
        raise TypeError("Membership must be a torch Tensor")
    return values.bool()


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    elif path.suffix == ".npy":
        value = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "losses" if "losses" in archive else "utilities"
        if key not in archive:
            raise ValueError("NPZ must contain losses or utilities")
        value = archive[key]
    else:
        value = _torch_load(path)
        if isinstance(value, torch.Tensor):
            value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def lds_command(args: argparse.Namespace) -> None:
    device = _device(args.device)
    scores = _load_scores(args.scores, device)
    membership = _load_membership(args.subsets, scores.shape[1]).to(device)
    values = _load_array(args.actual_values)
    utilities = -values if args.values_are == "losses" else values
    result = linear_datamodeling_score(
        scores,
        membership,
        utilities,
        num_bootstrap=args.bootstrap,
        bootstrap_seed=args.seed,
        confidence=args.confidence,
    )
    payload = result.to_dict()
    payload["values_are"] = args.values_are
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


def pack_subsets_command(args: argparse.Namespace) -> None:
    subsets = _load_subset_indices(Path(args.indices))
    membership = np.zeros((len(subsets), args.num_train), dtype=np.uint8)
    for row, indices in enumerate(subsets):
        index = np.asarray(indices, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= args.num_train):
            raise ValueError(f"Subset {row} contains an out-of-range index")
        membership[row, index] = 1
    packed = np.packbits(membership, axis=1, bitorder="little")
    np.savez_compressed(
        args.output,
        packed_membership=packed,
        num_train=np.asarray(args.num_train, dtype=np.int64),
    )
    print(f"Packed {len(subsets)} subsets into {args.output}")


def _add_curvature_fit_options(
    parser: argparse.ArgumentParser,
    *,
    group_title: str | None = None,
):
    target = (
        parser
        if group_title is None
        else parser.add_argument_group(
            group_title,
            "These options are ignored when --curvature supplies a saved index.",
        )
    )
    target.add_argument("--oversample", type=int, default=10)
    target.add_argument("--power-iterations", type=int, default=3)
    target.add_argument(
        "--damping",
        type=float,
        help="Explicit per-layer lambda; overrides --damping-multiplier",
    )
    target.add_argument(
        "--damping-multiplier",
        type=float,
        default=DEFAULT_DAMPING_MULTIPLIER,
        help=(
            "Multiplier in lambda = multiplier * trace(G.T @ G) / D "
            f"(default: {DEFAULT_DAMPING_MULTIPLIER})"
        ),
    )
    target.add_argument(
        "--mode",
        choices=["tail-corrected", "truncated"],
        default="truncated",
        help="Spectral approximation outside the retained rank (default: truncated)",
    )
    target.add_argument("--seed", type=int, default=0)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lorif", description="Low-rank influence functions for data attribution"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit-curvature",
        help="Explicitly fit and save reusable curvature approximations",
    )
    fit.add_argument("index", help="Training factor index")
    fit.add_argument("output", help="New curvature output directory")
    fit.add_argument("--rank", type=int, required=True)
    _add_curvature_fit_options(fit)
    fit.add_argument("--device", default="auto")
    fit.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32"
    )
    fit.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=True)
    fit.set_defaults(func=fit_curvature_command)

    score = subparsers.add_parser(
        "score",
        help="Fit or reuse curvature and score query factors",
        description=(
            "With --rank, fit and consume one layer's curvature at a time without "
            "saving it. Use --curvature to reuse an explicitly saved index."
        ),
    )
    score.add_argument("train_index")
    score.add_argument("query_index")
    score.add_argument("output")
    curvature_source = score.add_mutually_exclusive_group(required=True)
    curvature_source.add_argument(
        "--rank",
        type=int,
        help="Fit this rank in memory; the curvature tensors are not saved",
    )
    curvature_source.add_argument(
        "--curvature",
        help="Reuse an explicitly saved curvature index instead of fitting",
    )
    inline_fit = _add_curvature_fit_options(
        score,
        group_title="inline curvature fitting (with --rank)",
    )
    inline_fit.add_argument(
        "--fit-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dtype used by the in-memory randomized SVD (default: float32)",
    )
    score.add_argument("--device", default="auto")
    score.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32"
    )
    score.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=True)
    score.set_defaults(func=score_command)

    lds = subparsers.add_parser(
        "evaluate-lds", help="Evaluate score additivity on subsets"
    )
    lds.add_argument("scores", help="LoRIF score directory or legacy .pt tensor")
    lds.add_argument("subsets", help="Packed membership NPZ, tensor, or JSON directory")
    lds.add_argument("actual_values", help="[subsets, queries] losses or utilities")
    lds.add_argument("--values-are", choices=["losses", "utilities"], default="losses")
    lds.add_argument("--bootstrap", type=int, default=5_000)
    lds.add_argument("--seed", type=int, default=12_345)
    lds.add_argument("--confidence", type=float, default=0.95)
    lds.add_argument("--device", default="auto")
    lds.add_argument("--output")
    lds.set_defaults(func=lds_command)

    pack = subparsers.add_parser(
        "pack-subsets", help="Pack included-example JSON files into a small NPZ"
    )
    pack.add_argument("indices")
    pack.add_argument("output")
    pack.add_argument("--num-train", type=int, required=True)
    pack.set_defaults(func=pack_subsets_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
