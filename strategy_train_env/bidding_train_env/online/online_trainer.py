"""stable_baselines3-compatible RL trainer for the AuctionNet online pipeline."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from bidding_train_env.online.definitions import ALGO_CLASS_DICT


@dataclass
class OnlineTrainer:
    algo: str
    envs: Any  # VecNormalize-wrapped vectorized env
    load_model_path: Optional[str] = None
    log_dir: str = "."
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 1_000_000

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._dump_configs()
        self.agent = self._init_agent()

    def _get_algo_class(self):
        if self.algo not in ALGO_CLASS_DICT:
            raise ValueError(
                f"Unknown algo '{self.algo}'. Registered: {list(ALGO_CLASS_DICT)}"
            )
        return ALGO_CLASS_DICT[self.algo]

    def _init_agent(self):
        cls = self._get_algo_class()
        if self.load_model_path is not None:
            print(f"Loading model from {self.load_model_path}")
            return cls.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=None,
                custom_objects=self.model_config,
            )
        print("No model path provided. Initializing new model.")
        return cls(
            env=self.envs,
            verbose=1,
            tensorboard_log=None,
            **self.model_config,
        )

    def _dump_configs(self):
        with open(os.path.join(self.log_dir, "model_config.json"), "w") as f:
            json.dump(
                self.model_config, f, indent=2, default=lambda _: "<not serializable>"
            )

    def train(self):
        self.agent.learn(
            total_timesteps=self.timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

    def save(self):
        self.agent.save(os.path.join(self.log_dir, "final_model.zip"))
        if hasattr(self.envs, "save"):
            self.envs.save(os.path.join(self.log_dir, "final_vecnormalize.pkl"))
