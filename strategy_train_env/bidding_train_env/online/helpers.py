"""Small utilities: safe reductions and checkpoint discovery/resume."""

import glob
import os
import re
from pathlib import Path

import numpy as np

MODEL_PATTERN = "rl_model_*_steps.zip"
VECNORM_PATTERN = "rl_model_vecnormalize_{n}_steps.pkl"


def safe_mean(arr):
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def safe_max(arr):
    if len(arr) == 0:
        return 0.0
    return float(np.max(arr))


def _get_step_number(filename: str) -> int:
    m = re.search(r"rl_model_(\d+)_steps\.zip$", filename)
    return int(m.group(1)) if m else -1


def get_last_checkpoint(path):
    path = str(path)
    if not os.path.isdir(path):
        return None
    files = sorted(glob.glob(os.path.join(path, MODEL_PATTERN)), key=_get_step_number)
    nums = [_get_step_number(f) for f in files if _get_step_number(f) >= 0]
    return max(nums) if nums else None


def get_model_and_env_path(log_dir, load_path, checkpoint_num):
    """Resolve (model_path, vecnormalize_path) for training start.

    1. If `log_dir` already has checkpoints, resume from the latest (ignoring load_path).
    2. Otherwise, if `load_path` is given, load `checkpoint_num` (or its latest).
    3. Otherwise, return (None, None) to start fresh.
    """
    log_dir = str(log_dir) if log_dir else None
    load_path = str(load_path) if load_path else None

    if log_dir and os.path.isdir(log_dir):
        n = get_last_checkpoint(log_dir)
        if n is not None:
            load_path = log_dir
            checkpoint_num = n
            print(f"Resuming from checkpoint {n} in {log_dir}")

    if load_path is None:
        return None, None

    if checkpoint_num is None:
        checkpoint_num = get_last_checkpoint(load_path)
    if checkpoint_num is None:
        print(f"No checkpoints found at {load_path}, starting fresh.")
        return None, None

    model_path = os.path.join(load_path, f"rl_model_{checkpoint_num}_steps.zip")
    env_path = os.path.join(load_path, VECNORM_PATTERN.format(n=checkpoint_num))
    return model_path, env_path
