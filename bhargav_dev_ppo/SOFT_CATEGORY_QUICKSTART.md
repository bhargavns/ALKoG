# ALKoG soft-category experiment

This experiment tests the smallest useful perception change before fine-tuning
SAM itself. SAM and ResNet-18 stay frozen. A supervised category head learns
six proposal classes:

| ID | Category | Environment role |
|---:|---|---|
| 0 | food | terminal reward `+10` |
| 1 | lion | terminal penalty `-50` when loose |
| 2 | cage | enclosure used for the lion/cage relation |
| 3 | water | terminal reward `+6` |
| 4 | poison | terminal penalty `-20` |
| 5 | background | rejected/background SAM proposals |

The simulator's segmentation-ID render is used only to generate labels and
evaluate predictions. Training and inference receive RGB frames.

## 1. Create an environment

From `troy_dev_ppo/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-softcat.txt
python scripts/download_sam_checkpoint.py
```

ResNet-18 ImageNet weights are downloaded by `torchvision` the first time the
perception pipeline starts. On a headless Linux server, run
`export MUJOCO_GL=egl`. Use `--device cpu` without CUDA; SAM ViT-B is much
faster on a CUDA GPU.

## 2. Run a small smoke test

```bash
PYTHONPATH=. pytest -q tests/test_soft_categories.py
python scripts/generate_category_dataset.py \
  --scenes 4 --views-per-scene 1 --candidate-images 8 --device cuda \
  --output output/smoke_dataset.pt \
  --candidate-dir output/smoke_candidates
```

Inspect `output/smoke_candidates/`. Labels beginning with `oracle:` show which
simulator object occupies each accepted SAM proposal. Several cage fragments
are allowed at this stage; the category head should assign them consistently.

## 3. Generate the training proposals

```bash
python scripts/generate_category_dataset.py \
  --scenes 300 --views-per-scene 2 --candidate-images 80 --device cuda \
  --output output/category_dataset.pt \
  --candidate-dir output/candidates_oracle
```

Generated outputs:

- `category_dataset.pt`: proposal embeddings, labels, frame groups, and diagnostics;
- `category_dataset.summary.json`: class counts and per-object proposal recall;
- `candidates_oracle/*.png`: candidate proposals with oracle labels.

If one category has low proposal recall, adjust SAM thresholds before training
the classifier. Classification cannot repair an object SAM never proposes.

## 4. Train the category head

```bash
python scripts/train_category_model.py \
  --dataset output/category_dataset.pt \
  --output output/category_model.pt \
  --epochs 30 --batch-size 256 --device cuda
```

The split is grouped by rendered frame, preventing proposals from one image
from appearing in both partitions. The best validation macro-F1 checkpoint is
saved to `output/category_model.pt`.

## 5. Evaluate and generate predicted candidate images

```bash
python scripts/evaluate_category_model.py \
  --dataset output/category_dataset.pt \
  --model output/category_model.pt \
  --live-scenes 40 --prediction-images 80 \
  --output-dir output/category_evaluation --device cuda
```

Inspect `metrics.json`, `confusion_matrix.png`, and the `prediction_*.png`
files. Track macro-F1 together with proposal recall and cage fragmentation.

## 6. Train PPO from soft category slots

Short integration check:

```bash
python scripts/train_soft_category_ppo.py \
  --category-model output/category_model.pt \
  --iterations 2 --horizon 256 --epochs 2 --minibatch-size 64 \
  --perceive-every 25 --device cuda
```

Main experiment:

```bash
python scripts/train_soft_category_ppo.py \
  --category-model output/category_model.pt \
  --iterations 150 --horizon 2048 --perceive-every 25 --device cuda
```

The run directory contains `metrics.jsonl`, periodic checkpoints, and
`policy_final.pt`. Compare food/water rates with poison/death rates rather than
using mean return alone.

## 7. Run inference and inspect policy perception

```bash
python scripts/run_soft_category_policy.py \
  --policy output/soft_category_ppo/run_YYYYMMDD_HHMMSS/policy_final.pt \
  --category-model output/category_model.pt \
  --episodes 40 --output-dir output/policy_inference --device cuda
```

The output contains predicted perception images and `summary.json` with food,
water, poison, death, and timeout rates.

## Scope and genericity

This version tests stable supervised categories against the original hard
cosine-threshold KG. Category presence, confidence, freshness, age, and
egocentric location are continuous policy inputs. PPO learns category symbols,
while the proposal classifier remains frozen during PPO.

Additional distinct categories can be added in `lib/ObjectCategories.py`, with
matching MuJoCo bodies and regenerated training data. The current fixed slots
collapse multiple simultaneous instances of the same category. A generic
multi-instance world should replace them with tracked instance slots while
retaining the same soft category distribution per slot.

SAM mask-decoder fine-tuning is deliberately deferred. Start it after the
reports identify whether the remaining bottleneck is proposal recall, category
consistency, cage extent, or relation inference.
