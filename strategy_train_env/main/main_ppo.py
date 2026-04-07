import numpy as np
import torch
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ENV = os.path.dirname(_THIS)            # strategy_train_env/
_REPO_ROOT = os.path.dirname(_TRAIN_ENV)       # AuctionNet/
sys.path.append(_TRAIN_ENV)
sys.path.append(_REPO_ROOT)

from run.run_ppo import run_ppo

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_ppo()
