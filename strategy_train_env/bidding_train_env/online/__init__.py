from gymnasium.envs.registration import register

register(
    id="BiddingEnv-v0",
    entry_point="bidding_train_env.online.online_env:BiddingEnv",
)
