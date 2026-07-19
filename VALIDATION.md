# Validation record

Validation performed on 2026-07-19 before public release.

## Self-contained checks

```text
PYTHONPATH=src python -m unittest discover -s tests -v
20 tests passed
```

The tests include dense-reference comparisons for projected per-example
gradients and Woodbury influence, an end-to-end factor-store to score pipeline,
and regressions for low-precision cancellation, multi-pass row drift, partial
write corruption, checkpoint/curvature/projection provenance, paper-style
WikiText blocking, and LDS sign handling.

The no-download example also completed on CPU:

```text
scores shape: (2, 8)
top training example per query: [2, 4]
```

The embedded WikiText regression guards were checked directly against the
trusted legacy caches: 232,585 train blocks and 487 validation blocks, including
all six selected token-block hashes.

An explicit H200 run of `tests/gpu_smoke.py` also passed. It checks the legacy
CUDA projection RNG stream element by element, verifies that CPU/CUDA projection
digests differ, compares GPU influence with a dense solve, and exercises BF16
scoring. FP16 scoring has a separate rejection regression for spectrum overflow.

## Legacy GPT-2 LDS artifact check

A checkpoint-free H200 run evaluated the available rank-32, projection-factor-4,
rank-4096 score snapshot with all 100 subset memberships and the available
averaged loss cache:

```text
scores shape:       (487, 232585, 1)
membership shape:   (100, 232585)
losses shape:       (100, 487)
LDS:                0.3028154929
95% interval:       [0.2908455380, 0.3152111188]
valid queries:      487
```

Artifact SHA-256 values:

```text
scores:      8fc7e4a759d5644b60da7f0bc19598f22a588a944c4fd2900ba5ce84bf638be9
losses:      0b478a836ef2b2b099fc574eaf6263ba34c685db931d223752ed1d1949fbffac
membership:  8c3a34119d080b04d467e0a3939deca0e83e19c6868ebf5924fdd8decd52b1e7
```

The packed membership was checked exactly against all 100 JSON files; every
subset contains 116,292 included examples. The public utility convention is
algebraically identical to the value printed by the legacy evaluator.

This snapshot does not reproduce an older notebook comment of `0.3162`. That
notebook used an unseeded randomized SVD and overwriteable artifact filenames,
and no hash was recorded for the commented result. Therefore the older number
cannot be treated as a regression target without recovering its exact artifacts.
The validation above establishes the result for the explicitly hashed snapshot.

Large validation artifacts and subset checkpoints are intentionally excluded
from Git.
