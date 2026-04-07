import os
import sys
import ast
import math
import logging
import numpy as np
import pandas as pd
import torch

from bidding_train_env.common.utils import normalize_state, save_normalize_dict
from bidding_train_env.baseline.ppo.ppo import PPO
from bidding_train_env.baseline.ppo.rollout_buffer import RolloutBuffer
from bidding_train_env.baseline.ppo.state_builder import (
    build_state, apply_normalize, STATE_DIM, NUM_TICK,
)
from bidding_train_env.baseline.ppo.training_controller import TrainingController
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_DIR = "saved_model/PPOtest"


def _safe_literal(x):
    if pd.isna(x):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return eval(x, {"__builtins__": {}}, {"np": np})
        except Exception:
            return x


def load_offline_for_warmstart():
    path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    df = pd.read_csv(path)
    df["state"] = df["state"].apply(_safe_literal)
    df["next_state"] = df["next_state"].apply(_safe_literal)
    normalize_dict = normalize_state(df, STATE_DIM, normalize_indices=[13, 14, 15])
    states = np.stack(df["normalize_state"].values).astype(np.float32)
    actions = df["action"].values.astype(np.float32)
    return states, actions, normalize_dict


class PpoRolloutAgent(BaseBiddingStrategy):
    """Training-mode player wrapper: builds the 16-dim state, samples alpha
    via PPO.act(), and pushes the transition into the shared rollout buffer."""

    def __init__(self, ppo: PPO, normalize_dict: dict, buffer: RolloutBuffer,
                 budget: float = 100, name: str = "Ppo-Rollout",
                 cpa: float = 2, category: int = 1):
        super().__init__(budget, name, cpa, category)
        self.ppo = ppo
        self.normalize_dict = normalize_dict
        self.buffer = buffer

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        raw = build_state(
            timeStepIndex, pValues, historyPValueInfo, historyBid,
            historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
            self.budget, self.remaining_budget,
        )
        obs = apply_normalize(raw, self.normalize_dict)
        alpha, logp, value, log_alpha = self.ppo.act(obs)
        self.buffer.add(obs, logp, value, log_alpha)
        return float(alpha) * np.asarray(pValues)


def _adjust_over_cost(bids, over_cost_ratio, slot_coefficients, winner_pit):
    """Mirror of run/run_test.py::adjust_over_cost — keeps the rollout
    loop budget-safe in the same way the eval loop does."""
    overcost = np.where(over_cost_ratio > 0)[0]
    for ai in overcost:
        for si, _ in enumerate(slot_coefficients):
            pv_idx = np.where(winner_pit[:, si] == ai)[0]
            rng = np.random.default_rng(seed=1)
            n = math.ceil(pv_idx.size * over_cost_ratio[ai])
            if n > 0:
                drop = rng.choice(pv_idx, n, replace=False)
                bids[drop, ai] = 0


def _get_winner(slot_pit):
    slot_pit = slot_pit.T
    num_pv, _ = slot_pit.shape
    winner = np.full((num_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        idx = np.argwhere(slot_pit == pos)
        if idx.size > 0:
            pv_i, a_i = idx.T
            winner[pv_i, pos - 1] = a_i
    return winner


def run_one_episode(controller: TrainingController, player_index: int,
                    episode_seed: int, buffer: RolloutBuffer,
                    cpa_penalty_coef: float = 10.0):
    controller.reset(episode=episode_seed)
    envs = controller.biddingEnv
    pvgen = controller.pvGenerator
    agents = controller.agents
    num_tick = controller.num_tick
    num_agent = len(agents)

    history_pv = []
    history_bids = []
    history_auc = []
    history_imp = []
    history_lwc = []

    ep_reward = 0.0
    ep_cost = 0.0
    ep_conversions = 0.0

    pre_buffer_len = len(buffer.obs)

    for t in range(num_tick):
        pv_values = pvgen.pv_values[t]
        pvalue_sigmas = pvgen.pValueSigmas[t]

        bids = [
            agents[i].bidding(
                t, pv_values[:, i], pvalue_sigmas[:, i],
                [x[i] for x in history_pv],
                [x[i] for x in history_bids],
                [x[i] for x in history_auc],
                [x[i] for x in history_imp],
                history_lwc,
            ) if agents[i].remaining_budget >= envs.min_remaining_budget
            else np.zeros(pv_values.shape[0])
            for i in range(num_agent)
        ]
        bids = np.array(bids).transpose()
        bids[bids < 0] = 0

        remaining = np.array([a.remaining_budget for a in agents])

        ratio_max = None
        cost = None
        winner_pit = None
        xi = slot = cost_pit = is_exposed = conv = lwc = None
        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                over = np.maximum((cost - remaining) / (cost + 1e-4), 0)
                _adjust_over_cost(bids, over, envs.slot_coefficients, winner_pit)
            xi, slot, cost_pit, is_exposed, conv, lwc, _ = envs.simulate_ad_bidding(
                pv_values, pvalue_sigmas, bids
            )
            real_cost = cost_pit * is_exposed
            cost = real_cost.sum(axis=1)
            winner_pit = _get_winner(slot)
            over = np.maximum((cost - remaining) / (cost + 1e-4), 0)
            ratio_max = over.max()

        for i, a in enumerate(agents):
            a.remaining_budget -= cost[i]

        # Player slot might already be wrapped in PlayerAgentWrapper; the
        # wrapped agent is what we logged into the buffer in this turn.
        # Player-only metrics:
        player_reward = float((pv_values[:, player_index] * is_exposed[player_index]).sum())
        player_cost = float(cost[player_index])
        player_conv = float(conv[player_index].sum())
        ep_reward += player_reward
        ep_cost += player_cost
        ep_conversions += player_conv

        done = (t == num_tick - 1) or (
            agents[player_index].remaining_budget < envs.min_remaining_budget
        )
        buffer.record_reward(player_reward, done)

        history_pv.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_bids.append(bids.transpose())
        history_auc.append(np.stack((xi, slot, cost_pit), axis=-1))
        history_imp.append(np.stack((is_exposed, conv), axis=-1))
        history_lwc.append(lwc)

        if done:
            break

    # Terminal CPA shaping: penalize the final logged tick if realized CPA
    # overruns the constraint.
    realized_cpa = (ep_cost / ep_conversions) if ep_conversions > 0 else float("inf")
    cpa_cap = float(agents[player_index].cpa)
    if realized_cpa > cpa_cap and len(buffer.rew) > 0:
        overrun = (realized_cpa - cpa_cap) / max(cpa_cap, 1e-6)
        buffer.rew[-1] -= cpa_penalty_coef * overrun

    buffer.finish_path(last_value=0.0)
    return ep_reward, ep_cost, ep_conversions, len(buffer.obs) - pre_buffer_len


def run_ppo():
    torch.manual_seed(1)
    np.random.seed(1)

    logger.info("Loading offline data for warm-start + normalize_dict ...")
    states, actions, normalize_dict = load_offline_for_warmstart()
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_normalize_dict(normalize_dict, SAVE_DIR)
    logger.info(f"Saved normalize_dict.pkl to {SAVE_DIR}")

    ppo = PPO(dim_obs=STATE_DIM)
    logger.info("Behavioral-cloning warm start ...")
    ppo.bc_pretrain(states, actions, epochs=5, batch_size=512)

    NUM_ITER = 200
    EPISODES_PER_ITER = 4
    ROTATE_PLAYER_IDX = list(range(7))

    buffer = RolloutBuffer(gamma=ppo.gamma, lam=ppo.lam)
    player_agent = PpoRolloutAgent(ppo, normalize_dict, buffer)
    controller = TrainingController(
        player_index=0,
        player_agent=player_agent,
        num_tick=NUM_TICK,
        num_agent_category=8,
        num_category=6,
        pv_num=500000,
        pv_generator_type="neuripsPvGen",
    )

    for it in range(NUM_ITER):
        ep_rewards = []
        for k in range(EPISODES_PER_ITER):
            pidx = ROTATE_PLAYER_IDX[(it * EPISODES_PER_ITER + k) % len(ROTATE_PLAYER_IDX)]
            controller.player_index = pidx
            # Rebuild the unwrapped opponent roster so load_agents() doesn't
            # leave stale PlayerAgentWrapper slots from a previous player_index.
            controller.agent_list = controller.initialize_agents()
            controller.agents = controller.load_agents()
            seed = it * 1000 + k
            r, c, conv, _ = run_one_episode(controller, pidx, seed, buffer)
            ep_rewards.append(r)

        batch = buffer.build_batch()
        pi, v, ent = ppo.update(batch, n_epochs=4, minibatch_size=256)
        logger.info(
            f"iter {it:03d} | avg_ep_reward={np.mean(ep_rewards):.2f} "
            f"pi_loss={pi:.4f} v_loss={v:.4f} ent={ent:.3f}"
        )

    ppo.save_jit(SAVE_DIR)
    logger.info(f"Saved PPO JIT model to {SAVE_DIR}/ppo_model.pth")


if __name__ == "__main__":
    run_ppo()
