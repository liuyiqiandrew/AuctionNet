# PPO Lagrangian CPA-aware reward shaping

Per-step penalty on CPA overspend added to PPO's dense reward, mirroring the IQL
`iql_reward_shaping` branch:

```
overspend_t  = max(0, step_cost_t − target_cpa · step_conv_t)
r_shaped_t   = dense_reward_t − λ · overspend_t
```

λ=0.0 reproduces the baseline PPO behavior exactly (new code path is guarded by
`if self.lambda_cpa > 0.0`).

## What changed

- `bidding_train_env/online/online_env.py` — `BiddingEnv` accepts a `lambda_cpa`
  kwarg; applies the penalty in `_place_bids`; emits per-step `overspend` in
  `info` for logging.
- `bidding_train_env/online/main_train_ppo.py` — new `--lambda_cpa` CLI flag
  plumbed into `build_env_configs`.
- `bidding_train_env/online/definitions.py` — `overspend` added to
  `INFO_KEYWORDS` so the callback averages it per rollout.

## Baseline

Ran locally on CPU (MacBook Air, 10 cores, 85 min) with `--bc_range dense`
(budget 400–12000, target_cpa 6–12). The intended bc_range going forward is
`default` (each advertiser's raw `(budget, CPAConstraint)` from the parquet) —
so the shaping sweep below uses `--bc_range default` and a matching baseline
should be re-run under the same setting before comparing shaped runs against it.

Dense-range baseline eval (`main_eval_ppo.py --eval_mode both --n_eval_episodes 100 --bc_range dense`):

| mode   | score_mean |
|--------|-----------:|
| random |     0.1750 |
| sweep  |     4.8284 |

Sweep `target_cpa_over_cpa ≈ 0.05` end-of-training — i.e., actual CPA ~20× target.
This is the exact failure mode shaping is meant to address.

## λ sweep (needs GPU to finish in reasonable time)

Includes λ=0.0 as a matched baseline under `--bc_range default`:

```bash
for L in 0.0 0.5 1.0 2.0; do
  python bidding_train_env/online/main_train_ppo.py \
      --num_envs 20 --num_steps 10000000 --batch_size 512 \
      --seed 0 --bc_range default \
      --obs_type obs_16_keys --act_type act_1_key \
      --learning_rate 2e-5 --save_every 100000 \
      --lambda_cpa "$L" \
      --out_prefix "002_lambda${L}_" --out_suffix "_ppo_default_obs16"
done

for L in 0.0 0.5 1.0 2.0; do
  python bidding_train_env/online/main_eval_ppo.py \
      --load_path "../output/online/training/ongoing/002_lambda${L}_ppo_seed_0_ppo_default_obs16" \
      --eval_mode both --n_eval_episodes 100 \
      --obs_type obs_16_keys --bc_range default
done
```

On CPU each run is ~85 min. On a mid-range GPU expect ~10–20 min each → the
full sweep + eval should land in ~1 hour.

## Interpreting results

Check for each λ: eval `score_mean` (sweep mode is the comparable one),
`cost_over_budget`, `target_cpa_over_cpa`. If shaping works we expect
`target_cpa_over_cpa` to rise toward 1.0 (or overshoot to >1.0 meaning the
policy is *under*-spending, like the IQL shaped runs did on validation). Score
should stay comparable or improve vs baseline 4.83.

If scores collapse across all λ > 0: the per-step penalty is too harsh — try
clipping `overspend` at its 99th percentile inside `_place_bids`, or start the
sweep at smaller values like {0.1, 0.25, 0.5}.
