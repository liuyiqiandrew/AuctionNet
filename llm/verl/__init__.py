"""VeRL integration for the AuctionNet LLM bidding agent.

This package uses VeRL 0.4.1's supported async VLLM multi-turn path:
- `auction_completion_callback.py` drives the 48-tick AuctionNet env
- `auction_reward_manager.py` converts terminal env metrics into rewards
- `launch_train.py` registers local extensions before delegating to
  `verl.trainer.main_ppo`
"""
