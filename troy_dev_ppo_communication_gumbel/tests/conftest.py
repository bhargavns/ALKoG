"""Keep imports and test artifacts inside this standalone project."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'output' / 'matplotlib'))
import torch
torch.set_num_threads(1)


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / 'output' / 'pytest_tmp')
