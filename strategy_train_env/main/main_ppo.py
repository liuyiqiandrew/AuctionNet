"""Entry point for PPO training, evaluation, and plotting.

Usage:
    cd strategy_train_env
    python main/main_ppo.py                          # train + eval (default)
    python main/main_ppo.py --mode train             # train only
    python main/main_ppo.py --mode eval              # eval saved model
    python main/main_ppo.py --mode plot              # regenerate plots
    python main/main_ppo.py --num-iterations 5       # quick test run
"""

import argparse
import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ENV = os.path.dirname(_THIS)
_REPO_ROOT = os.path.dirname(_TRAIN_ENV)
sys.path.append(_TRAIN_ENV)
sys.path.append(_REPO_ROOT)

from run.run_ppo import run_ppo, TrainConfig
from run.run_eval_ppo import run_eval_ppo, EvalConfig
from bidding_train_env.baseline.ppo.metrics import (
    MetricsTracker, plot_training_curves, plot_eval_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="PPO training and evaluation for AuctionNet")
    p.add_argument("--mode", choices=["train", "eval", "all", "plot"], default="all")
    p.add_argument("--save-dir", default="saved_model/PPOtest")
    p.add_argument("--num-iterations", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="Episodes per player index during evaluation")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tracker = None

    if args.mode in ("train", "all"):
        train_config = TrainConfig(
            num_iterations=args.num_iterations,
            episodes_per_iter=args.episodes_per_iter,
            save_dir=args.save_dir,
            seed=args.seed,
        )
        tracker = run_ppo(train_config)

    if args.mode in ("eval", "all"):
        eval_config = EvalConfig(
            model_dir=args.save_dir,
            num_episodes_per_player=args.eval_episodes,
            seed=args.seed + 10000,
        )
        tracker = run_eval_ppo(eval_config, tracker)

    if args.mode == "plot":
        metrics_path = os.path.join(args.save_dir, "metrics.json")
        tracker = MetricsTracker.load(metrics_path)
        plot_training_curves(tracker, args.save_dir)
        plot_eval_summary(tracker, args.save_dir)
        print(f"Plots regenerated in {args.save_dir}")


if __name__ == "__main__":
    main()
