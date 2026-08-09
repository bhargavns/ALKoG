# Five-object soft-category change summary

## Environment

- Added `water` and `poison` MuJoCo bodies.
- Randomized all five object locations with minimum separation constraints.
- Added terminal water reward (`+6`) and poison penalty (`-20`).
- Added distance/outcome metadata and positive-resource potential shaping.
- Added aligned simulator category-mask rendering for training diagnostics.

## Perception

- Preserved the original frozen SAM + ResNet KG baseline.
- Added a stable vocabulary: food, lion, cage, water, poison, background.
- Added full proposal output with masks and SAM quality scores.
- Added oracle proposal labeling, grouped dataset splits, class-balanced
  training, confusion metrics, fragmentation metrics, and live holdout images.

## Reinforcement learning

- Added fixed soft category slots containing presence, confidence, freshness,
  age, and egocentric position.
- Added a short-term category anchor memory and lion/cage relation features.
- Added a dedicated PPO actor-critic, trainer, training CLI, and inference CLI.
- Extended legacy PPO outcome logging for water and poison.

## Compatibility fixes

- Removed the external `CORE` environment-variable requirement from packaged
  scripts.
- Replaced missing `lib.python.*` imports with local package imports.
- Added explicit runtime requirements, checkpoint downloader, tests, and a
  complete quick-start guide.

## Validation performed in the build workspace

- Parsed every Python file with `ast.parse` and `compileall`.
- Parsed every XML asset with Python's XML parser.
- Checked the tree for the removed external imports and stale 13-dimensional
  observation declarations.

The build workspace did not contain PyTorch, MuJoCo, OpenCV, or SAM, so the GPU
rollout test must be run after installing `requirements-softcat.txt`. The
recommended four-scene dataset command in `SOFT_CATEGORY_QUICKSTART.md` is the
fastest full-stack smoke test.
