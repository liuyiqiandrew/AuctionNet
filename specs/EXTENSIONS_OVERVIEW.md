# RL for Large-Scale Auto-Bidding — Extensions Overview

COS 435 / ECE 433 (Spring 2026) final project. This document covers the six
extensions the team built on top of the AuctionNet PPO baseline, in the order
they're presented in `COS435_FinalProject_Presentation (2).pdf`.

Each section explains: (a) the motivation, (b) what concretely changes in the
pipeline, (c) the observed effect on the eval metric. Scores quoted are period-27
held-out sweep evaluations from the presentation's summary table.

---

## 0. Common baseline

All extensions sit on top of the same online PPO pipeline:

- **Environment**: `BiddingEnv` — a `gymnasium.Env` wrapping one advertiser's
  bidding problem over 48 timesteps for a fixed delivery period. Actions are a
  1-D log-bid coefficient on `[-10, 10]`; `bid = exp(action) × pvalue × target_cpa`.
- **Algorithm**: `stable_baselines3.PPO` with 20 `SubprocVecEnv` workers (one
  per training period 7..26).
- **Training**: 10M env steps, `batch_size=512`, `learning_rate=1e-4` (linear
  schedule), `clip_range=0.3`, `gae_lambda=0.9`, `ent_coef=3e-6`.
- **Evaluation**: period 27 held out; eval runs either `random` (100 sampled
  campaigns) or `sweep` (one deterministic rollout per advertiser).
- **Baseline PPO score**: **24.8** on the 16-feature observation.

Each extension below modifies exactly one layer of the pipeline (observation,
reward, sampling, policy network, or the optimiser loop).

---

## 1. Reward shaping (Ophelia)

**Layer modified**: reward function.

**Motivation**: the NeurIPS scoring function multiplies episode conversions by
`(target_cpa / actual_cpa)^2` whenever CPA is exceeded. So the *final* reward
punishes CPA overspend, but the *per-step* dense reward doesn't — there's no
in-episode gradient signal telling the policy to stop overspending *now*. PPO
ends up with healthy conversion volume but inflated CPA, losing score to the
penalty.

**What changes**: a Lagrangian direct-penalty term is added to the per-step
reward:

```
overspend_t = max(0, step_cost_t − target_cpa × step_conversions_t)
r_shaped_t  = r_t − λ × overspend_t
```

The penalty is zero when the running marginal CPA is at or below target, and
scales linearly with overspend otherwise. Multiple λ values swept — `1e-2`,
`1e-3`, `1e-4`. `λ=0` exactly recovers baseline PPO.

**Implementation**: `bidding_train_env/online/online_env.py` accepts a
`lambda_cpa` kwarg; applies the penalty in `_place_bids` and emits per-step
`overspend` in `info` for logging. `main_train_ppo.py` adds a `--lambda_cpa`
CLI flag.

**Result**: score **25.6** — a small but consistent improvement over the
16-feature baseline (24.8). CPA compliance trajectory tightens visibly without
sacrificing conversion count.

---

## 2. Replay reweighting (Claire)

**Layer modified**: training-period sampling.

**Motivation**: the default pipeline uses one `SubprocVecEnv` worker per
training period (7..26), so each rollout draws uniformly from all 20 periods.
Some periods have harder advertisers, tighter CPA constraints, or unusual
budget profiles. Uniform sampling wastes gradient budget on easy periods where
the policy already performs well.

**What changes**: the period-sampling distribution is tilted toward
historically-difficult periods. Each advertiser's past NeurIPS score becomes a
signal of difficulty — low score means the period is hard. Periods are sampled
with probabilities

```
w_j = α × softmax(−score_j / T) + (1 − α) × uniform
```

where `T` is a temperature (tried T=30) and `α ∈ [0, 1]` interpolates between
pure reweighted sampling and uniform. At `α=0.9, T=30`, the weighted
distribution concentrates on the lowest-scoring third of periods while still
sampling the others occasionally.

**Implementation**: on the `origin/weighted-sampling` branch. Replaces the
fixed period-per-worker assignment with a dynamic sampling layer that draws a
new `(advertiser, period)` per episode according to `w_j`.

**Result**: score **25.8** — the weighted-PPO training curve is consistently
above the uniform baseline across the full 10M steps (not just at convergence),
suggesting the gradient is being spent more efficiently, not just routing the
policy to a different local optimum.

---

## 3. Sequence encoder / GRU (Andrew)

**Layer modified**: policy network architecture.

**Motivation**: the standard observation vector includes hand-crafted
summaries of history (`last_three_X`, `historical_X` etc.). These are
fixed-window moving averages — a policy that needs to react to a trend over
the last 10 ticks, or detect a regime shift over the last 20, can't express
that from a 3-step mean alone. The fix is to let the policy *learn* its own
history summary.

**What changes**: instead of feeding the current flat observation to an MLP,
stack the last `K` observations into one vector, then pass through a residual
GRU feature extractor before the PPO actor/value heads:

```
[obs_{t-K+1}, ..., obs_t] → GRU(hidden=64) + residual(latest obs) → LayerNorm → PPO heads
```

The residual path (adding the most-recent observation back after the GRU)
ensures the policy never loses direct access to the current state. Separate
GRU extractors are used for actor and value networks.

**Implementation**: `TemporalGRUFeaturesExtractor` in
`bidding_train_env/online/temporal_policy.py`. Opt-in via
`--temporal_seq_len K` (defaults to 1, which is plain PPO). The env emits a
flattened `(K × obs_dim,)` vector when `K > 1`; the GRU unflattens it before
processing. Action space, reward, and checkpoint format all unchanged.

Best configuration: 16-feature state, hidden dim 64, sequence length 32.

**Result**: score **25.5** — a modest improvement over the 16-feature
baseline. The gap to the 60-feature baseline (27.0) suggests that hand-crafted
features still carry more signal at this scale than the GRU learns from raw
history; scaling hidden dim or training longer may close the gap.

---

## 4. Market-regime conditioning / state augmentation (Bonnie)

**Layer modified**: observation vector (Tier 1: pure JSON; Tier 2: small
additions in `_get_state_dict`).

**Motivation**: even at 60 features, the observation vector is a *snapshot* of
point statistics (per-tick means, last-three views, historical means). It
doesn't express three classes of information a human bidder would use:

1. **Time derivatives** — is competition heating up or cooling down?
2. **Second moments / distribution shape** — is the market volatile? Are there
   bidding "whales" pulling the right tail?
3. **Opponent aggression proxies** — the simulator doesn't expose opponent
   bids directly, but ratios like `floor-cost / available-pvalue` and
   `our-bid / floor-cost` infer opponent stance from auction outcomes.

**What changes**: a new 98-key observation config `obs_market_regime_v1.json`
adds 38 features on top of obs_60 in two tiers:

- **Tier 1 (20 features, no code change)** — features `BiddingEnv` already
  computes but `obs_60` doesn't expose: `cpa_exceedence_rate` × 3 views,
  `bid_success_count_slot_{1,2,3}` × 3 views (per-slot win counts — obs_60 has
  per-slot *costs* but not per-slot *wins*), `exposure_mean_slot_{1,2,3}` × 3
  views, raw `conversion_count`/`cost_sum` per tick, and episode cumulatives.
- **Tier 2 (18 features, added to `_get_state_dict`)** — derived from existing
  `history_info` data, no change to auction dynamics:
  - **Competitor distribution shape**: `lwc_trend_slope_last_five`,
    `lwc_std_last_five`, `lwc_cv_last_five` (coefficient of variation),
    `lwc_z_last` (z-score of current vs episode history),
    `lwc_pct_gap_last` (01-pct minus 10-pct → right-tail width).
  - **Opponent proxies**: `opponent_aggression_last` (lwc mean / pvalues
    mean), `bid_competitiveness_last` (our bid mean / lwc mean).
  - **Slot dynamics**: `cost_slot_1_over_slot_3_last`,
    `exposure_ratio_top_to_floor_last`,
    `slot_concentration_trend_last_five`.
  - **Regime z-scores**: `pv_intensity_z_last`, `win_rate_z_last`.
  - **Momentum**: `win_rate_trend_slope_last_five`,
    `pv_over_lwc_trend_slope_last_five`,
    `conversion_rate_trend_last_five`.
  - **Pacing**: `spend_velocity_residual` =
    `(total_cost / budget) − (1 − time_left)`.

**Implementation**: `bidding_train_env/online/configs/obs_market_regime_v1.json`
(new), plus ~60 lines in `BiddingEnv._get_state_dict` with two local helpers
(`_slope`, `_std`). Simulator dynamics (`step`, `reset`, `_place_bids`,
`_update_history`) and the action space are unchanged.

**Result**: score **29.7** — the largest single-extension gain in the team's
experiments, +4.9 over the 16-feature baseline and +2.7 over obs_60. The
extension is most effective because it directly addresses obs_60's blind
spots (temporal dynamics and opponent inference) rather than competing with
features obs_60 already captures.

Full writeup: `strategy_train_env/bidding_train_env/online/MARKET_REGIME_EXTENSION.md`.

---

## 5. State feature augmentation (16 → 60 features, team-wide)

**Layer modified**: observation vector (pure config change).

**Motivation**: the AuctionNet competition winner (OIL) used a 60-feature
observation. The default AuctionNet benchmark exposes only 16 — essentially
time, budget, and a handful of bid/cost/win-rate aggregates. Three things
obs_16 can't express that obs_60 can:

1. **CPA constraint exposure**: obs_16 has no `cpa` value. The policy literally
   doesn't know what its CPA target is, only that it's being penalised when it
   exceeds it at the end. obs_60 adds the raw `cpa` feature.
2. **Historical bidding/pacing context**: obs_60 adds historical averages of
   per-slot costs, bid-over-floor ratios, pvalue-over-floor ratios, and
   per-tick `last_X` snapshots (not just `last_three_X`).
3. **Competitor-bid distribution shape**: obs_16 shows only the *mean* winning
   cost; obs_60 adds the 10th and 1st percentile of competitor bids.

**What changes**: a new JSON observation config `obs_60_keys.json`. The
`BiddingEnv` already computed all 60 values — the change is purely a selection
list. Zero code change to the simulator or algorithm.

**Implementation**: `bidding_train_env/online/configs/obs_60_keys.json`; pass
`--obs_type obs_60_keys` to `main_train_ppo.py` and `main_eval_ppo.py`.

**Result**: score **27.0** vs 24.8 baseline. Visibly higher score across the
full 10M-step training curve and tighter cost-over-budget variance. Both
obs_60 and obs_16 continue to be supported; `obs_60` is the new default for
subsequent extensions (market-regime, reward shaping sweep, etc.).

---

## 6. RL training on LLM (GRPO + LoRA on Qwen3-8B, Andrew)

**Layer modified**: policy representation — swaps the MLP policy for a
pretrained 8-billion-parameter language model.

**Motivation**: can a pretrained LLM, fine-tuned with RL, reach competitive
bidding performance faster than training a policy from scratch? The
hypothesis: language models have already learned general numerical
reasoning / comparative ranking patterns, so RL fine-tuning should only need
to specialise them to the bidding task, not teach them from zero.

**What changes**:

- **Model**: Qwen3-8B, with LoRA adapters applied on top of the frozen base.
  LoRA rank and the projection layers it attaches to are the main
  capacity/efficiency knob.
- **Algorithm**: GRPO (Group Relative Policy Optimisation) — a PPO variant
  that *removes the value network*. Instead, each rollout step samples a group
  of `n=4` trajectories from the same prompt; the group's mean and std serve
  as the baseline for advantage estimation:

  ```
  A_j = (score_j − μ_group) / σ_group
  ```

  This halves memory/compute vs PPO (no value head to train) but makes
  advantage estimation much noisier — especially when the policy concentrates
  and the group's advantages collapse toward zero.
- **Prompt format**: each tick, the LLM is prompted with the
  current bidding state (time remaining, budget remaining, CPA, recent
  market statistics) and asked to emit a numerical bid coefficient.
- **Training hierarchy**: one trajectory = one `(advertiser, period)` over 48
  ticks. One rollout row = 4 independent trajectories sampled from the same
  prompt. One training step = 4 prompts × 4 samples = 16 episodes.

**Results and failure modes**:

- **Early phase (steps 0–100)**: validation reward climbs from 17.1
  (base-model baseline) to ~21, beating from-scratch PPO at matched step
  count. Score on train is noisy but trending upward.
- **Mid phase (steps ~100–150)**: train score continues climbing up to ~30
  in some batches, but validation plateaus around 21. Policy entropy drops
  sharply.
- **Late phase (steps ~150+)**: policy collapses into near-deterministic
  outputs. GRPO's group-advantage signal vanishes (when all 4 samples from a
  prompt produce essentially the same bid, their relative advantages are all
  zero). Gradients shrink to near zero and training effectively halts.

**Final score**: **21.2** — below from-scratch PPO ceiling (29.7 with
market-regime obs), but notable that an 8B LLM with ~250 gradient steps
reaches comparable performance to a from-scratch PPO policy at early-training
step counts. The sample efficiency win is real; the peak-performance loss is
the cost.

**Potential mitigations** (not implemented): KL penalty against the base
model to prevent mode collapse, larger group size (n=8 or 16) for more stable
advantage estimation, adaptive entropy bonus scheduling.

---

## Summary table (period-27 sweep eval, from presentation slide 7)

| Extension | Score | Layer modified | Effort |
|---|---:|---|---|
| PPO baseline (obs_16) | 24.8 | — | reference |
| + 60-feature state | 27.0 | observation (config) | tiny |
| + Reward shaping | 25.6 | reward fn | small |
| + Replay reweight | 25.8 | period sampling | small |
| + GRU encoder | 25.5 | policy net arch | medium |
| + Market-aware (obs + derived features) | **29.7** | observation (config + derivations) | medium |
| GRPO + LLM (Qwen3-8B) | 21.2 | policy model | large |

The extensions are mostly *independent* — each touches a different layer — so
in principle they compose. Future work: stack the top three (market-regime
obs + GRU encoder + reward shaping) in a single run to see whether the
gains stack additively or conflict.
