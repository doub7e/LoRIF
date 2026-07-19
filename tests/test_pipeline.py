import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from lorif.cli import fit_curvature_command, score_command
from lorif.collector import LayerSpec
from lorif.factors import Factors
from lorif.store import FactorWriter


def exact_factors(matrices: torch.Tensor) -> Factors:
    columns = matrices.shape[2]
    identity = torch.eye(columns, dtype=matrices.dtype).expand(
        matrices.shape[0], -1, -1
    )
    return Factors(matrices, identity)


class PipelineTests(unittest.TestCase):
    def test_factor_store_to_curvature_to_score(self):
        torch.manual_seed(29)
        train_matrices = torch.randn(8, 3, 2)
        query_matrices = torch.randn(3, 3, 2)
        spec = {"linear": LayerSpec("linear", 3, 2, 3, 2)}
        metadata = {
            "model": "toy",
            "model_revision": "test",
            "model_revision_resolved": "test-commit",
            "model_fingerprint": "toy:test-weights",
            "tokenizer": "toy",
            "tokenizer_revision": "test",
            "tokenizer_revision_resolved": "test-tokenizer-commit",
            "pad_token_id": 0,
            "projection_factor": 1,
            "factor_rank": 2,
            "factor_iterations": 1,
            "projection_seed": 0,
            "projection_normalization": "legacy",
            "projection_rng": "legacy-gpt2-device",
            "projection_sha256": "toy-projections",
            "factor_layout": "input-by-output",
            "loss_reduction": "sum",
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with FactorWriter(
                root / "train",
                spec,
                shard_size=3,
                storage_dtype=torch.float32,
                metadata=metadata,
            ) as writer:
                writer.write({"linear": exact_factors(train_matrices[:5])})
                writer.write({"linear": exact_factors(train_matrices[5:])})
            with FactorWriter(
                root / "query",
                spec,
                shard_size=2,
                storage_dtype=torch.float32,
                metadata=metadata,
            ) as writer:
                writer.write({"linear": exact_factors(query_matrices)})

            damping = 0.6
            fit_curvature_command(
                argparse.Namespace(
                    index=str(root / "train"),
                    output=str(root / "curvature"),
                    rank=6,
                    oversample=0,
                    power_iterations=1,
                    damping=damping,
                    damping_policy="paper",
                    seed=3,
                    device="cpu",
                    dtype="float32",
                    mmap=False,
                )
            )
            score_command(
                argparse.Namespace(
                    train_index=str(root / "train"),
                    query_index=str(root / "query"),
                    curvature=str(root / "curvature"),
                    output=str(root / "scores"),
                    device="cpu",
                    dtype="float32",
                    mmap=False,
                )
            )

            manifest = json.loads((root / "scores" / "manifest.json").read_text())
            actual = torch.cat(
                [
                    torch.load(
                        root / "scores" / shard["file"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    for shard in manifest["shards"]
                ],
                dim=1,
            )
            train_dense = train_matrices.flatten(1)
            query_dense = query_matrices.flatten(1)
            hessian = train_dense.T @ train_dense + damping * torch.eye(6)
            expected = query_dense @ torch.linalg.solve(hessian, train_dense.T)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

            query_manifest_path = root / "query" / "manifest.json"
            original_query_manifest = query_manifest_path.read_text()
            query_manifest = json.loads(original_query_manifest)
            query_manifest["metadata"]["projection_sha256"] = "other-projections"
            query_manifest_path.write_text(json.dumps(query_manifest))
            with self.assertRaisesRegex(ValueError, "projection_sha256"):
                score_command(
                    argparse.Namespace(
                        train_index=str(root / "train"),
                        query_index=str(root / "query"),
                        curvature=str(root / "curvature"),
                        output=str(root / "projection-rejected-scores"),
                        device="cpu",
                        dtype="float32",
                        mmap=False,
                    )
                )
            query_manifest_path.write_text(original_query_manifest)

            curvature_manifest_path = root / "curvature" / "manifest.json"
            curvature_manifest = json.loads(curvature_manifest_path.read_text())
            curvature_manifest["source_factor_manifest_sha256"] = "0" * 64
            curvature_manifest_path.write_text(json.dumps(curvature_manifest))
            with self.assertRaisesRegex(ValueError, "not fitted from this training"):
                score_command(
                    argparse.Namespace(
                        train_index=str(root / "train"),
                        query_index=str(root / "query"),
                        curvature=str(root / "curvature"),
                        output=str(root / "rejected-scores"),
                        device="cpu",
                        dtype="float32",
                        mmap=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
