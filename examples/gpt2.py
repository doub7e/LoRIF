#!/usr/bin/env python3
"""GPT-2/WikiText example for building and evaluating LoRIF indexes.

This file intentionally contains model/data glue only. The LoRIF math lives in
the reusable ``lorif`` package. Release-format token tensors are memory-mapped
during indexing, and training factors are subsequently consumed in shards.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from lorif import FactorWriter, GradientCollector


def replace_gpt2_conv1d(model: nn.Module) -> int:
    """Replace Hugging Face GPT-2 Conv1D layers with equivalent Linear layers."""

    from transformers.pytorch_utils import Conv1D

    replaced = 0
    for name, child in list(model.named_children()):
        replaced += replace_gpt2_conv1d(child)
        if isinstance(child, Conv1D):
            linear = nn.Linear(
                child.weight.shape[0],
                child.weight.shape[1],
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(child.weight.transpose(0, 1))
                if child.bias is not None:
                    linear.bias.copy_(child.bias)
            setattr(model, name, linear)
            replaced += 1
    return replaced


def _load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


TokenBlocks = torch.Tensor | list[list[int]]
TokenBlock = torch.Tensor | list[int]


def load_blocks(path: str | Path) -> TokenBlocks:
    """Memory-map release blocks or load a trusted legacy pickle cache."""

    path = Path(path)
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    else:
        value = _load_torch(path)
    if isinstance(value, dict):
        try:
            value = value["input_ids"]
        except KeyError as error:
            raise ValueError(f"Missing input_ids in {path}") from error
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError(
                f"Expected a rank-2 token tensor in {path}, got {tuple(value.shape)}"
            )
        return value
    if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
        raise ValueError(f"Expected a list/tensor of token blocks in {path}")
    return value


class BlockDataset(Dataset):
    def __init__(self, blocks: TokenBlocks) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> TokenBlock:
        return self.blocks[index]


def collate_blocks(
    examples: list[TokenBlock], pad_token_id: int
) -> dict[str, torch.Tensor]:
    sequences = [torch.as_tensor(row, dtype=torch.long) for row in examples]
    if any(row.ndim != 1 for row in sequences):
        raise ValueError("Each token block must be one-dimensional")
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(
        [torch.ones_like(row) for row in sequences], batch_first=True, padding_value=0
    )
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PAPER_BLOCK_REGRESSION = {
    "train": {
        "count": 232_585,
        "digests": (
            "4a2b4de41e8f6afb6feca8b76e54e76cadfac0943a7aabd1b60448a74eea46c5",
            "89cf484f19f752bea3ec8f1f2ecd5aa3302da727bf01a0b47aea4ec72fbc21e4",
            "2a57437dd8955adc30b0b738fbb3cf69d2ce42d777b1d38d7b29b732bf5f969c",
        ),
    },
    "validation": {
        "count": 487,
        "digests": (
            "356ab41c83cfbbf8cfc520339cd145bdb6c4b40a5223c85c1e2c82090138f461",
            "59cf51fa0acd9521a03e941f727aed39a54d17a1919f94a72d54924345788979",
            "811220d71a77a2c7640572c993f8fc80681f43b32c98e626640a242fe7e279ab",
        ),
    },
}


def paper_blocks(
    texts: list[str], tokenizer, *, block_size: int, text_batch_size: int = 1_000
) -> list[list[int]]:
    """Reproduce the exact text concatenation used for the paper checkpoints."""

    if block_size <= 0 or text_batch_size <= 0:
        raise ValueError("block_size and text_batch_size must be positive")
    nonempty = [text for text in texts if text.strip()]
    blocks: list[list[int]] = []
    carry: list[int] = []
    for start in range(0, len(nonempty), text_batch_size):
        joined = "\n\n".join(nonempty[start : start + text_batch_size])
        token_ids = tokenizer(
            joined, truncation=False, padding=False, return_tensors=None
        )["input_ids"]
        combined = carry + list(token_ids)
        full_length = (len(combined) // block_size) * block_size
        blocks.extend(
            combined[offset : offset + block_size]
            for offset in range(0, full_length, block_size)
        )
        carry = combined[full_length:]
    return blocks


def _block_digest(block: list[int]) -> str:
    encoded = json.dumps(block, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_paper_blocks(blocks: list[list[int]], split: str) -> None:
    expected = PAPER_BLOCK_REGRESSION.get(split)
    if expected is None:
        return
    if len(blocks) != expected["count"]:
        raise RuntimeError(
            f"Prepared {len(blocks)} {split} blocks; expected {expected['count']} "
            "for the paper cache. Check dataset/tokenizer revisions."
        )
    positions = (0, len(blocks) // 2, len(blocks) - 1)
    actual_digests = tuple(_block_digest(blocks[index]) for index in positions)
    if actual_digests != expected["digests"]:
        raise RuntimeError(
            "Prepared blocks do not match the paper cache. Check the dataset and "
            "tokenizer revisions; refusing to create an incompatible artifact."
        )


def prepare_command(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.cache_dir, revision=args.revision
    )
    raw = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=args.split,
        cache_dir=args.cache_dir,
        revision=args.dataset_revision,
    )
    blocks = paper_blocks(
        list(raw["text"]),
        tokenizer,
        block_size=args.block_size,
        text_batch_size=args.text_batch_size,
    )
    if (
        args.tokenizer == "openai-community/gpt2"
        and args.block_size == 512
        and args.text_batch_size == 1_000
    ):
        _verify_paper_blocks(blocks, args.split)
    input_ids = torch.tensor(blocks, dtype=torch.long)
    torch.save(
        {
            "input_ids": input_ids,
            "metadata": {
                "dataset": "Salesforce/wikitext:wikitext-103-raw-v1",
                "dataset_revision": args.dataset_revision,
                "split": args.split,
                "tokenizer": args.tokenizer,
                "revision": args.revision,
                "block_size": args.block_size,
                "text_batch_size": args.text_batch_size,
                "preprocessing": "paper-v1-filter-join1000-global-blocks",
            },
        },
        output,
    )
    print(f"Saved {len(input_ids):,} blocks to {output}")


def model_fingerprint(
    model_reference: str, *, requested_revision: str, resolved_revision: str
) -> str:
    """Identify a Hub revision or hash local config and weight files."""

    source = Path(model_reference)
    if not source.is_dir():
        return f"hf:{model_reference}@{resolved_revision or requested_revision}"
    candidates = sorted(
        path
        for path in source.iterdir()
        if path.name in {"config.json", "generation_config.json"}
        or path.suffix == ".safetensors"
        or (path.suffix == ".bin" and path.name.startswith("pytorch_model"))
    )
    if not candidates:
        raise ValueError(f"No model config or weight files found in {source}")
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def index_command(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    blocks = load_blocks(args.data)
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, cache_dir=args.cache_dir, revision=args.tokenizer_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
        model_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(_resolve_device(args.device))
    replaced = replace_gpt2_conv1d(model)
    model.requires_grad_(False)
    model.eval()
    print(f"Replaced {replaced} GPT-2 Conv1D modules")

    import re

    pattern = re.compile(args.module_regex)
    module_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.search(name)
    ]
    if not module_names:
        raise ValueError(f"No Linear modules match {args.module_regex!r}")
    print(f"Tracking {len(module_names)} layers")

    loader = DataLoader(
        BlockDataset(blocks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate_blocks, pad_token_id=tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
    factor_iterations = args.factor_iterations
    if factor_iterations is None:
        factor_iterations = 8 if args.factor_rank == 1 else 16
    storage_dtype = torch.float32 if args.storage_dtype == "float32" else torch.bfloat16
    resolved_revision = str(
        getattr(model.config, "_commit_hash", None) or args.revision
    )
    resolved_tokenizer_revision = str(
        tokenizer.init_kwargs.get("_commit_hash") or args.tokenizer_revision
    )
    metadata = {
        "model": args.model,
        "model_revision": args.revision,
        "model_revision_resolved": resolved_revision,
        "model_fingerprint": model_fingerprint(
            args.model,
            requested_revision=args.revision,
            resolved_revision=resolved_revision,
        ),
        "tokenizer": tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_revision_resolved": resolved_tokenizer_revision,
        "pad_token_id": tokenizer.pad_token_id,
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": file_sha256(args.data),
        "num_examples": len(blocks),
        "projection_factor": args.projection_factor,
        "factor_rank": args.factor_rank,
        "factor_iterations": factor_iterations,
        "projection_seed": args.seed,
        "projection_normalization": args.projection_normalization,
        "projection_rng": (
            "legacy-gpt2-device"
            if args.projection_normalization == "legacy"
            else "portable-cpu"
        ),
        "factor_layout": "input-by-output",
        "loss": "causal_cross_entropy",
        "loss_reduction": "token-sum-per-example",
        "module_regex": args.module_regex,
    }

    with GradientCollector(
        model,
        module_names,
        projection_factor=args.projection_factor,
        factor_rank=args.factor_rank,
        factor_iterations=factor_iterations,
        projection_seed=args.seed,
        projection_normalization=args.projection_normalization,
    ) as collector, FactorWriter(
        args.output,
        collector.specs,
        shard_size=args.shard_size,
        storage_dtype=storage_dtype,
        metadata=metadata,
    ) as writer:
        writer.user_metadata["projection_sha256"] = collector.projection_sha256()
        for batch_index, batch in enumerate(loader):
            collector.clear()
            embedding = model.get_input_embeddings()
            embedding_device = embedding.weight.device
            input_ids = batch["input_ids"].to(embedding_device)
            attention_mask = batch["attention_mask"].to(embedding_device)
            labels = batch["labels"].to(embedding_device)
            inputs_embeds = embedding(input_ids).detach().requires_grad_(True)
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            labels = labels.to(logits.device)
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss.backward()
            writer.write(collector.pop())
            model.zero_grad(set_to_none=True)
            if batch_index % args.log_every == 0:
                seen = min((batch_index + 1) * args.batch_size, len(blocks))
                print(f"Indexed {seen:,}/{len(blocks):,} examples")
    print(f"Saved factor index to {args.output}")


@torch.no_grad()
def _query_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        token_losses = F.cross_entropy(
            logits[..., :-1, :].transpose(1, 2),
            labels[..., 1:],
            ignore_index=-100,
            reduction="none",
        )
        valid_tokens = labels[..., 1:].ne(-100).sum(dim=1).clamp_min(1)
        values.append((token_losses.sum(dim=1) / valid_tokens).cpu())
    return torch.cat(values)


def subset_losses_command(args: argparse.Namespace) -> None:
    """Convert external subset checkpoints into the small LDS loss artifact."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if any("{subset}" not in pattern for pattern in args.checkpoint_pattern):
        raise ValueError("Every --checkpoint-pattern must contain {subset}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    blocks = load_blocks(args.queries)
    tokenizer_name = args.tokenizer or args.checkpoint_pattern[0].format(subset=0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(
        BlockDataset(blocks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_blocks, pad_token_id=tokenizer.pad_token_id),
    )
    device = _resolve_device(args.device)
    all_losses = torch.empty(args.num_subsets, len(blocks), dtype=torch.float64)

    for subset in range(args.num_subsets):
        accumulated = torch.zeros(len(blocks), dtype=torch.float64)
        for pattern in args.checkpoint_pattern:
            checkpoint = Path(pattern.format(subset=subset))
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing subset checkpoint: {checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint, cache_dir=args.cache_dir
            ).to(device)
            model.eval()
            accumulated.add_(_query_losses(model, loader, device).double())
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_losses[subset] = accumulated / len(args.checkpoint_pattern)
        print(f"Computed losses for subset {subset + 1}/{args.num_subsets}")
    np.savez_compressed(output, losses=all_losses.numpy())
    print(f"Saved averaged loss matrix {tuple(all_losses.shape)} to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GPT-2/WikiText data, build LoRIF indexes, or export LDS losses."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-wikitext",
        help="Create the paper-compatible token-block artifact",
    )
    prepare.add_argument("output", help="New release-format .pt artifact")
    prepare.add_argument("--tokenizer", default="openai-community/gpt2")
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--dataset-revision", default="main")
    prepare.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    prepare.add_argument("--block-size", type=int, default=512)
    prepare.add_argument("--text-batch-size", type=int, default=1_000)
    prepare.add_argument("--cache-dir")
    prepare.set_defaults(func=prepare_command)

    index = commands.add_parser(
        "index",
        help="Build a sharded low-rank gradient-factor index",
    )
    index.add_argument("model", help="Hugging Face model ID or local checkpoint")
    index.add_argument(
        "data",
        help="Memory-mapped release .pt artifact or trusted legacy pickle cache",
    )
    index.add_argument("output", help="New factor-index directory")
    index.add_argument("--revision", default="main")
    index.add_argument("--tokenizer")
    index.add_argument("--tokenizer-revision", default="main")
    index.add_argument("--cache-dir")
    index.add_argument("--device", default="auto")
    index.add_argument("--device-map")
    index.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    index.add_argument("--module-regex", default=r"(?:attn|mlp)")
    index.add_argument("--projection-factor", type=int, default=16)
    index.add_argument(
        "--projection-normalization",
        choices=["legacy", "jl", "none"],
        default="legacy",
    )
    index.add_argument("--factor-rank", type=int, default=1)
    index.add_argument("--factor-iterations", type=int)
    index.add_argument("--seed", type=int, default=0)
    index.add_argument("--batch-size", type=int, default=8)
    index.add_argument("--shard-size", type=int, default=10_000)
    index.add_argument("--workers", type=int, default=0)
    index.add_argument(
        "--storage-dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    index.add_argument("--log-every", type=int, default=10)
    index.set_defaults(func=index_command)

    losses = commands.add_parser(
        "subset-losses",
        help="Export mean next-token cross-entropy losses for LDS",
    )
    losses.add_argument("queries", help="Query token-block artifact")
    losses.add_argument("output", help="New compressed .npz loss matrix")
    losses.add_argument(
        "--checkpoint-pattern",
        action="append",
        required=True,
        help="Repeat for each training seed; include a {subset} placeholder",
    )
    losses.add_argument("--num-subsets", type=int, default=100)
    losses.add_argument("--tokenizer")
    losses.add_argument("--cache-dir")
    losses.add_argument("--batch-size", type=int, default=8)
    losses.add_argument("--device", default="auto")
    losses.set_defaults(func=subset_losses_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
