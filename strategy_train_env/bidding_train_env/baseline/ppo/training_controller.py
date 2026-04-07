import numpy as np
from simul_bidding_env.Controller.Controller import Controller
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy


class TrainingController(Controller):
    """Controller with a checkpoint-free opponent roster so PPO rollouts don't
    require IQL/BC/CQL/... to be pre-trained.

    Reuses the same budget / CPA / category tables as the eval-time Controller,
    so the player slot is unchanged. Only the competitor roster is swapped for
    PidBiddingStrategy + OnlineLpBiddingStrategy, which are the two simul-side
    strategies that do not call torch.jit.load at construction time.
    """

    def initialize_agents(self):
        agents = []
        for i in range(self.num_category):
            agents.extend([
                PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                OnlineLpBiddingStrategy(episode=i % 2),
                PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                OnlineLpBiddingStrategy(episode=(i + 1) % 2),
                PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                OnlineLpBiddingStrategy(episode=i % 2),
                PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
                OnlineLpBiddingStrategy(episode=(i + 1) % 2),
            ])
        # Trim/pad to exactly num_agent (= num_category * num_agent_category).
        return agents[: self.num_agent]
