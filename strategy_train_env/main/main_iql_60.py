import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run.run_iql_60 import run_iql_60

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    run_iql_60()
