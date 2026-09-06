#!/bin/python3
import sys, os
if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.Logger import Logger
from lib.ArgumentParser import ArgumentParser
from lib.HumanoidWithObjectsEnv import HumanoidWithObjectsEnv


g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

required_arguments = []
optional_arguments = {}

USAGE = """
gym_test.py
    Required:

    Optional:

    Example Usage:
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def run_gym():
    # --- recorded run (rgb_array) ---
    env = HumanoidWithObjectsEnv(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="./videos",
        name_prefix="humanoid-test",
        episode_trigger=lambda x: True,
    )

    obs, info = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    # --- interactive run (requires a display) ---
    if not os.environ.get("DISPLAY"):
        print("No DISPLAY set — skipping interactive window")
        return

    env = HumanoidWithObjectsEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def main(inputArguments):
    initialize(inputArguments)
    run_gym()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
