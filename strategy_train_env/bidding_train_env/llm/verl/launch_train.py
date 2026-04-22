"""Launcher that registers local AuctionNet VeRL extensions before training.

Ray quirk: `@register("bidding_agent")` in `bidding_agent_loop` populates a
process-local dict. Importing the module here only registers it in the
driver process; the `TaskRunner` Ray actor and rollout workers start in
separate Python interpreters with empty registries. To fix, we pre-initialize
Ray with `worker_process_setup_hook` pointing at a tiny setup function that
re-imports the module inside every worker.
"""

from __future__ import annotations

import os
import sys

import ray

# Driver-side registration (for any ad-hoc use before main()).
import bidding_train_env.llm.verl.bidding_agent_loop  # noqa: F401
from verl.trainer.main_ppo import main


def _init_ray_with_setup_hook() -> None:
    if ray.is_initialized():
        return
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                # Forward PYTHONPATH so worker processes can import
                # `bidding_train_env.llm.verl.*` (the SLURM script exports this).
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            },
            # Fires once per Ray worker process; registers "bidding_agent"
            # AgentLoop in the worker's local registry.
            "worker_process_setup_hook": "bidding_train_env.llm.verl._worker_setup.setup",
        },
    )


if __name__ == "__main__":
    _init_ray_with_setup_hook()
    # Our local config composes against VeRL's bundled `ppo_trainer.yaml`.
    if not any(arg.startswith("hydra.searchpath=") for arg in sys.argv[1:]):
        sys.argv.append("hydra.searchpath=[pkg://verl.trainer.config]")
    main()
