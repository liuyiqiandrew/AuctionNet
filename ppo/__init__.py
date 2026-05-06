from gymnasium.envs.registration import register, registry

if "BiddingEnv-v0" not in registry:
    register(
        id="BiddingEnv-v0",
        entry_point="ppo.online_env:BiddingEnv",
    )
