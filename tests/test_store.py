import tempfile
import unittest
from pathlib import Path

import torch

from lorif.collector import LayerSpec
from lorif.factors import Factors
from lorif.store import FactorStore, FactorWriter


class StoreTests(unittest.TestCase):
    def test_shards_cover_rows_exactly_and_round_trip(self):
        torch.manual_seed(5)
        specs = {
            "a": LayerSpec("a", 4, 3, 4, 3),
            "b": LayerSpec("b", 2, 5, 2, 5),
        }
        batches = [3, 2, 2]
        originals = {name: [] for name in specs}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index"
            with FactorWriter(
                path, specs, shard_size=4, storage_dtype=torch.float32
            ) as writer:
                for size in batches:
                    payload = {}
                    for name, spec in specs.items():
                        value = Factors(
                            torch.randn(size, spec.projected_input_dim, 2),
                            torch.randn(size, spec.projected_output_dim, 2),
                        )
                        originals[name].append(value)
                        payload[name] = value
                    writer.write(payload)

            store = FactorStore(path)
            self.assertEqual(store.num_examples, 7)
            self.assertEqual(store.num_shards, 2)
            self.assertEqual(store.shard_range(0), (0, 4))
            self.assertEqual(store.shard_range(1), (4, 7))
            for name in specs:
                loaded = store.load_layer(name)
                expected_left = torch.cat([item.left for item in originals[name]])
                expected_right = torch.cat([item.right for item in originals[name]])
                torch.testing.assert_close(loaded.left, expected_left)
                torch.testing.assert_close(loaded.right, expected_right)

    def test_rejected_write_is_atomic_and_rank_is_enforced(self):
        specs = {
            "a": LayerSpec("a", 2, 2, 2, 2),
            "b": LayerSpec("b", 2, 2, 2, 2),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index"
            writer = FactorWriter(
                path, specs, shard_size=10, storage_dtype=torch.float32
            )
            bad = {
                "a": Factors(torch.full((1, 2, 1), 111.0), torch.ones(1, 2, 1)),
                "b": Factors(torch.ones(1, 3, 1), torch.ones(1, 2, 1)),
            }
            with self.assertRaisesRegex(ValueError, "Projected shape mismatch"):
                writer.write(bad)
            good = {
                "a": Factors(torch.ones(1, 2, 1), torch.ones(1, 2, 1)),
                "b": Factors(torch.full((1, 2, 1), 2.0), torch.ones(1, 2, 1)),
            }
            writer.write(good)
            changed_rank = {
                name: Factors(torch.ones(1, 2, 2), torch.ones(1, 2, 2))
                for name in specs
            }
            with self.assertRaisesRegex(ValueError, "Factor rank changed"):
                writer.write(changed_rank)
            writer.close()

            store = FactorStore(path)
            self.assertEqual(store.manifest["layers"][0]["factor_rank"], 1)
            torch.testing.assert_close(store.load_layer("a").left, torch.ones(1, 2, 1))


if __name__ == "__main__":
    unittest.main()
