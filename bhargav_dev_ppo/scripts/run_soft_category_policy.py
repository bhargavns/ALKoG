#!/usr/bin/env python3
"""Run a trained policy, save perception images, and summarize outcomes."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from lib.CategoryData import draw_category_boxes, save_json
from lib.CategoryModel import load_category_checkpoint
from lib.KGWorldEnv import CAMERA_NAMES, KinematicKGWorldEnv
from lib.Perception import PerceptionPipeline
from lib.SoftCategoryGrounding import SoftCategoryMemory, detect_soft_categories_geo
from lib.SoftCategoryPolicy import load_policy_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--category-model", type=Path, default=ROOT / "output" / "category_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "policy_inference")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--perceive-every", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def outcome_name(info, truncated):
    if info.get("food_reached"):
        return "food"
    if info.get("water_reached"):
        return "water"
    if info.get("poison_touched"):
        return "poison"
    if info.get("lion_caught"):
        return "death"
    return "timeout" if truncated else "end"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy, _ = load_policy_checkpoint(args.policy, device=args.device)
    category_model, _ = load_category_checkpoint(args.category_model, device=args.device)
    pipeline = PerceptionPipeline(device=args.device)
    env = KinematicKGWorldEnv(seed=args.seed)
    memory = SoftCategoryMemory(max_age=max(2 * args.perceive_every, 50))
    outcomes = {"food": 0, "water": 0, "poison": 0, "death": 0, "timeout": 0, "end": 0}
    episode_rows = []
    images_written = 0

    def refresh(obs, episode, step):
        nonlocal images_written
        detections, relations, debug_frames = detect_soft_categories_geo(
            env, pipeline, category_model, min_confidence=args.min_confidence
        )
        memory.update(detections, relations, obs[:2])
        for camera_name, (frame, boxes, ids, confidences) in zip(CAMERA_NAMES, debug_frames):
            if not boxes or images_written >= args.max_images:
                continue
            annotated = draw_category_boxes(frame, boxes, ids, confidences, prefix="pred:")
            path = args.output_dir / f"episode{episode:03d}_step{step:03d}_{camera_name}.png"
            if not cv2.imwrite(str(path), annotated):
                raise RuntimeError(f"Failed to write {path}")
            images_written += 1

    try:
        for episode in range(args.episodes):
            env.fixed_cage_state = bool(episode % 2 == 0)
            obs, _ = env.reset(seed=args.seed + episode)
            memory.reset()
            refresh(obs, episode, 0)
            episode_return = 0.0
            step = 0
            while True:
                slots, relations = memory.features(obs)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                slot_tensor = torch.as_tensor(slots, dtype=torch.float32, device=args.device).unsqueeze(0)
                relation_tensor = torch.as_tensor(
                    relations, dtype=torch.float32, device=args.device
                ).unsqueeze(0)
                action, _logp, _value = policy.act(
                    obs_tensor,
                    slot_tensor,
                    relation_tensor,
                    deterministic=not args.stochastic,
                )
                obs, reward, terminated, truncated, info = env.step(int(action.item()))
                episode_return += reward
                step += 1
                memory.advance()
                if terminated or truncated:
                    name = outcome_name(info, truncated)
                    outcomes[name] += 1
                    episode_rows.append(
                        {
                            "episode": episode,
                            "return": float(episode_return),
                            "steps": step,
                            "outcome": name,
                            "lion_caged": bool(info["lion_caged"]),
                        }
                    )
                    break
                if args.perceive_every and step % args.perceive_every == 0:
                    refresh(obs, episode, step)
    finally:
        env.close()

    summary = {
        "episodes": args.episodes,
        "outcomes": outcomes,
        "rates": {name: count / max(1, args.episodes) for name, count in outcomes.items()},
        "mean_return": float(np.mean([row["return"] for row in episode_rows])),
        "prediction_images": images_written,
        "episodes_detail": episode_rows,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
