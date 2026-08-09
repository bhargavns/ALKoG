#!/usr/bin/env python3
"""Train PPO using five fixed soft semantic category slots."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lib.CategoryModel import load_category_checkpoint
from lib.KGWorldEnv import KinematicKGWorldEnv
from lib.ObjectCategories import NUM_OBJECT_CATEGORIES
from lib.Perception import PerceptionPipeline
from lib.SoftCategoryGrounding import SLOT_FEATURE_DIM, detect_soft_categories_geo
from lib.SoftCategoryPolicy import SoftCategoryActorCritic, save_policy_checkpoint
from lib.SoftCategoryPPOTrainer import SoftCategoryPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-model", type=Path, default=ROOT / "output" / "category_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "soft_category_ppo")
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--perceive-every", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = args.output_dir / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.jsonl"

    env = KinematicKGWorldEnv(seed=args.seed)
    pipeline = PerceptionPipeline(device=args.device)
    category_model, _ = load_category_checkpoint(args.category_model, device=args.device)
    category_model.requires_grad_(False)

    def perception_fn():
        return detect_soft_categories_geo(
            env, pipeline, category_model, min_confidence=args.min_confidence
        )

    policy_config = {
        "obs_dim": int(env.observation_space.shape[0]),
        "n_actions": int(env.action_space.n),
        "num_categories": NUM_OBJECT_CATEGORIES,
        "slot_dim": SLOT_FEATURE_DIM,
        "symbol_dim": 8,
        "hidden_dim": 128,
    }
    model = SoftCategoryActorCritic(**policy_config)
    trainer = SoftCategoryPPOTrainer(
        model,
        env,
        perception_fn,
        device=args.device,
        horizon=args.horizon,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        perceive_every_steps=args.perceive_every,
    )

    try:
        for iteration in range(1, args.iterations + 1):
            stats = trainer.collect_and_update()
            stats["iteration"] = iteration
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats, sort_keys=True) + "\n")
            print(
                f"iter={iteration:04d} return={stats['mean_return']:+.3f} "
                f"food={stats['food_rate']:.3f} water={stats['water_rate']:.3f} "
                f"poison={stats['poison_rate']:.3f} death={stats['death_rate']:.3f} "
                f"timeout={stats['timeout_rate']:.3f} kl={stats['approx_kl']:.5f}"
            )
            if iteration % 25 == 0:
                save_policy_checkpoint(
                    run_dir / f"policy_iter{iteration:04d}.pt",
                    model,
                    iteration=iteration,
                    category_checkpoint=args.category_model,
                    config=policy_config,
                )
    finally:
        env.close()

    final_path = run_dir / "policy_final.pt"
    save_policy_checkpoint(
        final_path,
        model,
        iteration=args.iterations,
        category_checkpoint=args.category_model,
        config=policy_config,
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    print(f"Saved policy to {final_path}")


if __name__ == "__main__":
    main()
