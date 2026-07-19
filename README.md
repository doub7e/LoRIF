# LoRIF

Official implementation of **LoRIF: Low-Rank Influence Functions for Scalable
Training Data Attribution**.

[[Paper](https://arxiv.org/abs/2601.21929)]

LoRIF combines two low-rank approximations:

1. It stores each two-sided projected per-example gradient as rank-`c` factors.
2. It represents each layer's Gauss–Newton curvature with a rank-`r` truncated
   SVD and applies its inverse through the Woodbury identity.

This repository is deliberately small: the method, one GPT-2 example, and
focused correctness tests. It contains no model checkpoints, datasets, gradient
indexes, or score matrices.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For the GPT-2/WikiText example:

```bash
python -m pip install -e ".[gpt2]"
```

Python 3.10+ and PyTorch 2.1+ are required.

## 10-second CPU example

This runs end to end without downloading a model or dataset:

```bash
PYTHONPATH=src python examples/minimal.py
```

It creates synthetic low-rank gradients, fits the curvature approximation, and
attributes two queries to eight training examples.

## GPT-2 on WikiText-103

The full pipeline prepares token blocks, indexes train/query gradients, fits
curvature, and scores influence.

### 1. Prepare the paper's token blocks

```bash
python examples/gpt2.py prepare-wikitext artifacts/train.pt --split train
python examples/gpt2.py prepare-wikitext artifacts/query.pt --split validation
```

The default recipe exactly follows the paper training code: remove blank rows,
join groups of 1,000 rows with two newlines, concatenate the resulting token
streams, then create fixed 512-token blocks. The commands verify the expected
counts (232,585 train and 487 validation) and selected block hashes, and fail if
the dataset/tokenizer snapshot is incompatible.

The `index` and LDS commands also accept legacy trusted pickle caches directly.
Do not load pickle files from untrusted sources.

### 2. Store projected gradient factors

Set `MODEL` to the same final GPT-2 checkpoint for both indexes:

```bash
MODEL=/external/path/to/final-gpt2-checkpoint

python examples/gpt2.py index "$MODEL" \
  artifacts/train.pt artifacts/train-f16-c1 \
  --tokenizer openai-community/gpt2 \
  --projection-factor 16 --factor-rank 1 --batch-size 64

python examples/gpt2.py index "$MODEL" \
  artifacts/query.pt artifacts/query-f16-c1 \
  --tokenizer openai-community/gpt2 \
  --projection-factor 16 --factor-rank 1 --batch-size 64
```

The paper checkpoint is not committed because of its size. A Hugging Face GPT-2
model ID may be used for a method demo, but it will not reproduce paper scores.
For local checkpoints, the indexer hashes model config and weight files; for Hub
models it records resolved model/tokenizer revisions. Scoring rejects mismatched
checkpoints or projection settings.

The example tracks attention/MLP matrices, uses token-summed causal
cross-entropy, computes factors in FP32, and stores them in BF16. Low-rank
factorization happens during each layer's backward hook and is offloaded to CPU,
so dense projected gradients do not accumulate across layers.

### 3. Fit curvature

```bash
lorif fit-curvature artifacts/train-f16-c1 artifacts/curvature-r2048 \
  --rank 2048 --device cuda
```

The index is streamed through randomized SVD. Defaults match the paper:
oversampling `p=10`, three power iterations after the initial covariance
multiply, and

```text
lambda = 0.1 * mean(sigma[:r+p] ** 2)
```

Every randomized-SVD pass checks the exact row count and matrix layout.

### 4. Score influence

```bash
lorif score \
  artifacts/train-f16-c1 artifacts/query-f16-c1 \
  artifacts/curvature-r2048 artifacts/scores \
  --device cuda
```

Scores are summed over layers and written in training-order shards. Larger
scores are proponents under the paper's convention. Only one layer's query
factors and curvature are placed on the accelerator at once. Scoring supports
FP32 (default) and BF16; FP16 is rejected because squared singular values can
overflow its limited exponent range.

## Paper configurations

`f` is the projection factor, `c` the per-example factor rank, and `r` the
curvature rank.

| GPT-2 regime | `f` | `c` | `r` |
|---|---:|---:|---:|
| Low storage | 16 | 1 | 2,048 |
| Medium | 4 | 1 | 4,096 |
| High quality | 4 | 32 | 16,384 |

Rank `c=1` is the recommended cost-efficient default. Choose the smallest `f`
that fits storage, then increase `r`; increase `c` only when additional
per-example fidelity is worth the storage.

## LDS evaluation without committing 500 checkpoints

The GPT-2 LDS experiment uses 100 independently sampled 50% training subsets
and averages validation losses over five runs per subset. The 500 checkpoints
occupy hundreds of GiB and are intentionally external.

Convert them once into:

- an averaged loss matrix `[100, 487]`;
- a packed included-example matrix `[100, 232585]`.

Compute losses from checkpoints stored elsewhere:

```bash
python examples/gpt2.py subset-losses \
  artifacts/query.pt artifacts/subset-losses.npz \
  --checkpoint-pattern '/external/checkpoints/seed1/subset{subset}' \
  --checkpoint-pattern '/external/checkpoints/seed2/subset{subset}' \
  --checkpoint-pattern '/external/checkpoints/seed3/subset{subset}' \
  --checkpoint-pattern '/external/checkpoints/seed4/subset{subset}' \
  --checkpoint-pattern '/external/checkpoints/seed5/subset{subset}' \
  --device cuda
```

Pack the JSON index files. Each file contains examples **included** in its
training subset:

```bash
lorif pack-subsets /external/checkpoints/indices artifacts/subsets.npz \
  --num-train 232585
```

Evaluate:

```bash
lorif evaluate-lds \
  artifacts/scores artifacts/subsets.npz artifacts/subset-losses.npz \
  --values-are losses --device cuda
```

The evaluator computes, for every query,

```text
Spearman(actual utility, sum of included-example influence)
```

where utility is negative cross-entropy. It reports the mean and a query-level
percentile-bootstrap interval (5,000 resamples, seed 12,345). This removes the
legacy evaluator's inconsistent output sign. See [VALIDATION.md](VALIDATION.md)
for the checks performed against the available legacy artifacts.

## Compatibility and artifact safety

- `--projection-normalization legacy` reproduces the GPT-2 prototype's
  Rademacher seed/layout stream and `+/-1/d` scaling (with the default seed 0).
  `jl` selects conventional `+/-1/sqrt(d)` scaling and a portable CPU stream.
- `fit-curvature` uses the paper damping rule by default. Pass
  `--damping-policy legacy-gpt2` only for the older notebook rule
  `0.01 * log2(r) * mean(sigma[:r] ** 2)`.
- New factors use the paper's input-by-output matrix layout. The prototype used
  its transpose, which is mathematically equivalent when applied consistently;
  raw prototype factor shards are not silently accepted.
- Factor and curvature tensors are loaded with PyTorch's restricted
  `weights_only=True` loader. Manifests bind model identity, ordered layers,
  projection settings, factor rank/layout, loss reduction, dtype, and shard
  ranges. The actual ordered projection tensors are hashed, so different device
  RNG streams cannot be mixed. Curvature is cryptographically bound to its
  training-factor manifest.
- Commands refuse to overwrite non-empty artifact directories.

## Python API

The CLI is a thin wrapper over pure tensor operations:

```python
from lorif import Factors, fit_curvature, influence_scores
```

`Factors.left` is `[batch, projected_input, c]`, `Factors.right` is
`[batch, projected_output, c]`, and the dense gradient is
`left @ right.transpose(1, 2)`. See [examples/minimal.py](examples/minimal.py)
for a complete runnable use of the API.

`GradientCollector` can attach to explicitly selected `nn.Linear` layers. The
caller owns the forward pass and loss, making custom loss masks and
non-language applications straightforward.

## Validation

Run all CPU correctness tests with no downloads:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite checks projected gradients against per-example autograd, rank-`c`
factor algebra, full/truncated Woodbury scores against dense solves, streaming
and shard invariants, provenance rejection, GPT-2 conversion/preprocessing, and
LDS sign/bootstrap behavior.

An optional CUDA check is available for a GPU worker:

```bash
PYTHONPATH=src python tests/gpu_smoke.py
```

## Citation

```bibtex
@article{li2026lorif,
  title   = {LoRIF: Low-Rank Influence Functions for Scalable Training Data Attribution},
  author  = {Li, Shuangqi and Le, Hieu and Xu, Jingyi and Salzmann, Mathieu},
  journal = {arXiv preprint arXiv:2601.21929},
  year    = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
