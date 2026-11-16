# Unassigned-symbol grammar study

The [report](spontaneous-grammar-report.pdf) gives the self-contained setup,
objectives, complete five-seed results, syntax evidence and limitations.
[Source and reproduction](../../experiments/spontaneous_grammar/README.md).

Baseline `8152d91` is preserved. New source/protocol commit: `f922f1c`.
65 final training runs, no admission filtering or test-based checkpoint selection.
The CSV, run evaluations, codebooks, curves, development failures, source hashes,
independent audits and exact inference-weight manifest are retained here.
Fifteen strongest-method inference checkpoints are in the experiment package;
all 65 full and inference checkpoints are in the downloadable study archive.
The archive also includes raw MuJoCo episode/scene records and source snapshots.

The grid is a neural referential game with supplied movement. Invariance is an
explicit inductive bias; order-sensitive finite codes are not human language.
The new arena check uses exact state, public routes and frozen CUDA PPO motors.
