"""Online evaluation of trained PPO in the full 48-agent auction simulator.

Runs the deterministic PPO policy across multiple player positions and
episodes, reporting per-player and aggregate metrics. Uses a different seed
range from training to test generalization.
"""

import os
import math
import logging
from dataclasses import dataclass

import numpy as np
import pickle
import torch

from bidding_train_env.baseline.ppo.state_builder import (
    build_state, apply_normalize, STATE_DIM, NUM_TICK,
)
from bidding_train_env.baseline.ppo.training_controller import TrainingController
from bidding_train_env.baseline.ppo.metrics import (
    MetricsTracker, EvalEpisodeMetrics, compute_score,
    plot_eval_summary,
)
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model_dir: str = "saved_model/PPOtest"
    num_episodes_per_player: int = 5
    player_indices: tuple = (0, 1, 2, 3, 4, 5, 6)
    seed: int = 10001


# ---------------------------------------------------------------------------
# Deterministic evaluation agent
# ---------------------------------------------------------------------------

class PpoEvalAgent(BaseBiddingStrategy):
    """Deterministic PPO inference for evaluation. No buffer, no sampling."""

    def __init__(self, model, normalize_dict: dict,
                 budget=100, name="Ppo-Eval", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        self.model = model
        self.normalize_dict = normalize_dict

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo,
                historyBid, historyAuctionResult, historyImpressionResult,
                historyLeastWinningCost):
        raw = build_state(
            timeStepIndex, pValues, historyPValueInfo, historyBid,
            historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
            self.budget, self.remaining_budget,
        )
        obs = apply_normalize(raw, self.normalize_dict)
        state_t = torch.tensor(obs, dtype=torch.float32)
        alpha = float(self.model(state_t).item())
        return alpha * np.asarray(pValues)


# ---------------------------------------------------------------------------
# Simulator helpers (shared with run_ppo.py)
# ---------------------------------------------------------------------------

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


def _adjust_over_cost(bids, over_cost_ratio, slot_coefficients, winner_pit):
    overcost = np.where(over_cost_ratio > 0)[0]
    for ai in overcost:
        for si, _ in enumerate(slot_coefficients):
            pv_idx = np.where(winner_pit[:, si] == ai)[0]
            rng = np.random.default_rng(seed=1)
            n = math.ceil(pv_idx.size * over_cost_ratio[ai])
            if n > 0:
                drop = rng.choice(pv_idx, n, replace=False)
                bids[drop, ai] = 0


# ---------------------------------------------------------------------------
# Single evaluation episode
# ---------------------------------------------------------------------------

def run_one_eval_episode(controller: TrainingController, player_index: int,
                         episode_seed: int) -> EvalEpisodeMetrics:
    controller.reset(episode=episode_seed)
    envs = controller.biddingEnv
    pvgen = controller.pvGenerator
    agents = controller.agents
    num_tick = controller.num_tick
    num_agent = len(agents)

    history_pv, history_bids = [], []
    history_auc, history_imp, history_lwc = [], [], []
    ep_reward, ep_cost, ep_conversions = 0.0, 0.0, 0.0
    ticks_active = 0

    for t in range(num_tick):
        pv_values = pvgen.pv_values[t]
        pvalue_sigmas = pvgen.pValueSigmas[t]

        bids = np.array([
            agents[i].bidding(
                t, pv_values[:, i], pvalue_sigmas[:, i],
                [x[i] for x in history_pv], [x[i] for x in history_bids],
                [x[i] for x in history_auc], [x[i] for x in history_imp],
                history_lwc,
            ) if agents[i].remaining_budget >= envs.min_remaining_budget
            else np.zeros(pv_values.shape[0])
            for i in range(num_agent)
        ]).transpose()
        bids[bids < 0] = 0

        remaining = np.array([a.remaining_budget for a in agents])

        ratio_max = None
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

        player_reward = float((pv_values[:, player_index] * is_exposed[player_index]).sum())
        player_cost = float(cost[player_index])
        player_conv = float(conv[player_index].sum())
        ep_reward += player_reward
        ep_cost += player_cost
        ep_conversions += player_conv
        ticks_active = t + 1

        history_pv.append(np.stack((pv_values.T, pvalue_sigmas.T), axis=-1))
        history_bids.append(bids.transpose())
        history_auc.append(np.stack((xi, slot, cost_pit), axis=-1))
        history_imp.append(np.stack((is_exposed, conv), axis=-1))
        history_lwc.append(lwc)

        done = (t == num_tick - 1) or (
            agents[player_index].remaining_budget < envs.min_remaining_budget
        )
        if done:
            break

    cpa = ep_cost / ep_conversions if ep_conversions > 0 else float("inf")
    cpa_constraint = float(agents[player_index].cpa)
    budget = float(agents[player_index].budget)
    score = compute_score(ep_conversions, cpa, cpa_constraint)

    return EvalEpisodeMetrics(
        player_index=player_index, episode_seed=episode_seed,
        reward=ep_reward, cost=ep_cost, conversions=ep_conversions,
        cpa=cpa, cpa_constraint=cpa_constraint, score=score,
        budget=budget,
        budget_utilization=ep_cost / budget if budget > 0 else 0.0,
        num_ticks_active=ticks_active,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval_ppo(config: EvalConfig | None = None,
                 tracker: MetricsTracker | None = None) -> MetricsTracker:
    if config is None:
        config = EvalConfig()
    if tracker is None:
        tracker = MetricsTracker()

    # Load trained model
    model_path = os.path.join(config.model_dir, "ppo_model.pth")
    dict_path = os.path.join(config.model_dir, "normalize_dict.pkl")
    model = torch.jit.load(model_path)
    with open(dict_path, "rb") as f:
        normalize_dict = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")

    eval_agent = PpoEvalAgent(model, normalize_dict)
    controller = TrainingController(
        player_index=0, player_agent=eval_agent,
        num_tick=NUM_TICK, num_agent_category=8, num_category=6,
        pv_num=500000, pv_generator_type="neuripsPvGen",
    )

    # Run evaluation episodes
    by_player: dict[int, list[EvalEpisodeMetrics]] = {}
    episode_counter = 0

    for pidx in config.player_indices:
        for k in range(config.num_episodes_per_player):
            controller.player_index = pidx
            controller.agent_list = controller.initialize_agents()
            controller.agents = controller.load_agents()

            seed = config.seed + episode_counter
            episode_counter += 1
            result = run_one_eval_episode(controller, pidx, seed)
            tracker.log_eval_episode(**result.__dict__)
            by_player.setdefault(pidx, []).append(result)

            logger.info(
                f"player={pidx} ep={k} seed={seed} | "
                f"score={result.score:.1f} conv={result.conversions:.0f} "
                f"cpa={result.cpa:.1f}/{result.cpa_constraint:.1f} "
                f"butil={result.budget_utilization:.2f} ticks={result.num_ticks_active}"
            )

    # Print summary table
    header = f"{'Player':>7} {'Eps':>4} {'Score':>8} {'Reward':>8} {'Conv':>6} {'CPA':>7} {'CPA_C':>7} {'Butil':>6}"
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    all_results = []
    for pidx in sorted(by_player.keys()):
        eps = by_player[pidx]
        all_results.extend(eps)
        logger.info(
            f"{pidx:>7d} {len(eps):>4d} "
            f"{np.mean([e.score for e in eps]):>8.1f} "
            f"{np.mean([e.reward for e in eps]):>8.1f} "
            f"{np.mean([e.conversions for e in eps]):>6.0f} "
            f"{np.mean([e.cpa for e in eps if np.isfinite(e.cpa)]):>7.1f} "
            f"{eps[0].cpa_constraint:>7.1f} "
            f"{np.mean([e.budget_utilization for e in eps]):>6.2f}"
        )
    logger.info("-" * len(header))
    logger.info(
        f"{'ALL':>7} {len(all_results):>4d} "
        f"{np.mean([e.score for e in all_results]):>8.1f} "
        f"{np.mean([e.reward for e in all_results]):>8.1f} "
        f"{np.mean([e.conversions for e in all_results]):>6.0f} "
        f"{np.mean([e.cpa for e in all_results if np.isfinite(e.cpa)]):>7.1f} "
        f"{'':>7} "
        f"{np.mean([e.budget_utilization for e in all_results]):>6.2f}"
    )
    logger.info("=" * len(header))

    # Save metrics and plots
    metrics_path = os.path.join(config.model_dir, "metrics.json")
    tracker.save(metrics_path)
    plot_eval_summary(tracker, config.model_dir)
    logger.info(f"Evaluation complete. Results saved to {config.model_dir}")
    return tracker


if __name__ == "__main__":
    run_eval_ppo()
