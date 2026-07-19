import importlib.util
from pathlib import Path
import unittest

import torch


def load_example_module():
    example_path = Path(__file__).parents[1] / "examples" / "gpt2.py"
    spec = importlib.util.spec_from_file_location("lorif_gpt2_example", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GPT2CompatibilityTests(unittest.TestCase):
    def test_paper_blocking_matches_legacy_batch_concatenation(self):
        module = load_example_module()

        class CharacterTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(character) for character in text]}

        texts = ["alpha", "", " beta ", "gamma", "delta"]
        blocks = module.paper_blocks(
            texts, CharacterTokenizer(), block_size=7, text_batch_size=2
        )
        stream = [
            ord(character)
            for character in "alpha\n\n beta " + "gamma\n\ndelta"
        ]
        full_length = (len(stream) // 7) * 7
        expected = [
            stream[index : index + 7] for index in range(0, full_length, 7)
        ]
        self.assertEqual(blocks, expected)

    def test_conv1d_replacement_preserves_logits(self):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except Exception as error:
            self.skipTest(f"transformers GPT-2 is unavailable: {error}")

        module = load_example_module()

        torch.manual_seed(17)
        config = GPT2Config(
            vocab_size=23,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_inner=16,
        )
        model = GPT2LMHeadModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        with torch.no_grad():
            before = model(input_ids).logits
            count = module.replace_gpt2_conv1d(model)
            after = model(input_ids).logits
        self.assertEqual(count, 4)
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
