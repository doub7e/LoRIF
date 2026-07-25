# LoRIF

Official PyTorch implementation of
**[LoRIF: Low-Rank Influence Functions for Scalable Training Data Attribution](https://arxiv.org/abs/2601.21929)**.

LoRIF makes influence functions practical for large models by combining two
low-rank approximations:

1. each projected per-example gradient is stored as rank-`c` matrix factors;
2. each layer's curvature is approximated with a streaming rank-`r` randomized
   SVD.

The implementation is deliberately small. It contains the reusable LoRIF
library and one end-to-end GPT-2/WikiText-103 workflow. Model checkpoints,
datasets, factor indexes, and LDS subset checkpoints are not included.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[gpt2]"
```

LoRIF requires Python 3.10+ and PyTorch 2.1+.

## GPT-2 on WikiText-103

The commands below use the paper's low-storage configuration
`f=16, c=1, r=2048`.

### 1. Prepare the paper data

```bash
python examples/gpt2.py prepare-wikitext artifacts/train.pt --split train
python examples/gpt2.py prepare-wikitext artifacts/query.pt --split validation
```

This removes blank WikiText-103 rows, joins groups of 1,000 rows with two
newlines, and slices the global GPT-2 token stream into fixed 512-token blocks.
The script checks the expected paper data: 232,585 training blocks and 487
validation blocks.

### 2. Index projected gradients

Use the same final full-data GPT-2 checkpoint for the training and query
indexes:

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

Reduce `--batch-size` if needed; it controls the indexing working set. The
indexer tracks the 48 GPT-2 attention/MLP weight matrices and uses token-summed
causal cross-entropy. It computes factors in FP32, writes BF16 shards, and
records model, tokenizer, data, projection, and loss metadata.

The paper checkpoint is not committed because of its size. A Hugging Face GPT-2
model ID can be used to try the pipeline, but it will not reproduce paper
results.

### 3. Fit curvature and score influence

```bash
lorif score \
  artifacts/train-f16-c1 artifacts/query-f16-c1 artifacts/scores \
  --rank 2048 --device cuda
```

This is the default, disk-efficient path. For each layer, LoRIF streams the
training factor shards through randomized SVD, scores that layer, and releases
its curvature before moving to the next layer. Curvature vectors and singular
values are not saved; only score shards and small reproducibility metadata are
written.

Larger scores indicate stronger proponents under the paper's convention.

### Optional: reuse a curvature index

Save curvature only when several query indexes will reuse it:

```bash
lorif fit-curvature artifacts/train-f16-c1 artifacts/curvature-r2048 \
  --rank 2048 --device cuda

lorif score \
  artifacts/train-f16-c1 artifacts/query-f16-c1 artifacts/scores \
  --curvature artifacts/curvature-r2048 --device cuda
```

## Defaults and paper configurations

The release defaults are:

| Setting | Default |
|---|---:|
| per-example factor iterations | 8 for `c=1`; 16 for `c>1` |
| randomized-SVD oversampling | `p=10` |
| randomized-SVD power iterations | 3 |
| curvature mode | `truncated` |
| damping | `0.5 * trace(G.T @ G) / D` |
| curvature persistence | off |

Here `G` is the per-layer matrix of projected training gradients and `D` is its
feature dimension. Truncated mode retains curvature in the top-`r` subspace;
the complementary subspace keeps inverse weight `1 / lambda`. Tail correction
is a post-paper ablation and remains opt-in with `--mode tail-corrected`.

The GPT-2 settings reported in Table 1 of the paper are:

| Storage regime | `f` | `c` | `r` | Paper LDS |
|---|---:|---:|---:|---:|
| Low | 16 | 1 | 2,048 | 0.1392 |
| Medium | 4 | 1 | 4,096 | 0.2073 |
| High | 4 | 32 | 16,384 | 0.3428 |

Choose the smallest projection factor `f` that fits storage, keep `c=1` unless
more per-example fidelity is needed, and then increase `r` as the curvature
budget allows.

### Reproducibility note

The paper experiments used
`lambda = 0.1 * mean(sigma[:r+p] ** 2)`. Post-paper validation found the
full-trace rule above to be a more stable release default:

```text
lambda = 0.5 * trace(G.T @ G) / D
```

Thus the workflow and the `f/c/r` configurations align with the paper, while
exact Table 1 values additionally require the authors' original final checkpoint,
subset checkpoints, and historical damping artifacts. The historical automatic
policy is intentionally not exposed by this minimal release. Tune the release
rule with `--damping-multiplier`, or pass an explicit `--damping`.

## Memory and artifact behavior

- Prepared `.pt` token blocks are memory-mapped and consumed by a `DataLoader`;
  indexing does not convert the full corpus into Python objects.
- Gradient factors are written in aligned shards (`--shard-size`, default
  10,000 examples).
- Every randomized-SVD pass reads one factor shard at a time. It never
  materializes the full training gradient matrix `G`.
- Scoring loads the small query set for one layer and streams training shards.
- The default `lorif score --rank ...` path never saves curvature tensors.
- Commands refuse to overwrite non-empty artifact directories.

The GPT-2 helper and LDS loader also accept legacy pickle artifacts. Pickle files
are loaded eagerly and must never come from an untrusted source.

## LDS evaluation

The paper evaluates 100 random 50% training subsets and averages validation loss
over five training runs per subset. This release stores mean next-token loss;
for the fixed 512-token paper blocks, the historical token-sum cache differs
only by the constant factor 511 and gives identical Spearman LDS. The 500 subset
checkpoints are intentionally kept outside this repository.

Convert them once into a small loss matrix:

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

Pack each subset's included-example JSON file and evaluate:

```bash
lorif pack-subsets /external/checkpoints/indices artifacts/subsets.npz \
  --num-train 232585

lorif evaluate-lds \
  artifacts/scores artifacts/subsets.npz artifacts/subset-losses.npz \
  --values-are losses --device cuda
```

The evaluator reports the mean per-query Spearman correlation and a
query-bootstrap confidence interval. It converts losses to utility internally,
so higher LoRIF scores consistently mean stronger proponents.

## Repository layout

```text
src/lorif/
  collector.py   projected per-example gradients
  factors.py     rank-c factor algebra
  store.py       sharded factor storage
  curvature.py   streaming randomized SVD
  influence.py   low-rank inverse scoring
  lds.py         LDS metric
  cli.py         command-line workflow
examples/
  gpt2.py        WikiText-103 preparation, indexing, and subset losses
```

The default `legacy` projection normalization reproduces the original GPT-2
prototype's Rademacher layout, seed stream, and `+/-1/d` scaling. Factor
manifests hash the actual projections and bind model/data metadata so
incompatible training and query indexes are rejected.

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
