# Market-regime / opponent-aware observation extension

Extension of the online PPO pipeline that enriches the policy's observation
vector with features capturing **market regime** (how competition is evolving),
**opponent behavior** (inferred from auction outcomes), and **temporal
dynamics** (trends, second moments, z-scores) that the stock `obs_60_keys`
vector does not surface.

Owner: Bonnie (Person 5 in `specs/Team_Division_for_PPO.pdf`). Branch:
`bl/market-regime`.

## 1. Purpose

The AuctionNet simulator runs 48 advertisers competing in a repeated
second-price auction. The stock PPO policy observes a 60-dim snapshot
(`obs_60_keys.json`) of point statistics: per-tick means, last-three-tick
views, and episode-historical aggregates of bid / win / cost signals, plus
per-slot cost means.

Point statistics miss three classes of information that a human bidder would
clearly use:

1. **Time derivatives** (is competition heating up?): slope of competitor
   bids / win rate / ROI over recent ticks.
2. **Second moments / distribution shape** (is the market volatile? are
   there "whale" bidders?): std of competitor bids, top-percentile gap,
   skew of the bid distribution.
3. **Regime z-scores and opponent proxies** (am I in an unusual regime right
   now? how aggressive are opponents?): current-tick z-score vs episode mean;
   floor-cost per available pvalue; bid-over-floor ratio.

This extension adds those features to the observation vector and re-trains
PPO with the richer state, without touching the auction dynamics (simulator
step/reset/`_place_bids`) or the raw per-tick history cache.

## 2. Conceptual design

Two layers of addition, combined into a single new observation config:

### Tier 1 — free additions (no code changes)

The `BiddingEnv._update_history` method already populates ~125 named
per-tick statistics, of which `obs_60_keys.json` uses only 60. The
remaining ~80 are computed every step but never shown to the policy. Tier 1
is just a JSON selection step: pick the most policy-relevant ones and add
them to the new obs config. No Python code changes.

**Chosen Tier 1 additions (20 keys)**:

| Family | Keys | Why |
|---|---|---|
| CPA compliance trend | `last_/last_three_/historical_cpa_exceedence_rate` | Directly measures how far current CPA is from target. Upstream training loss already uses this but the policy never sees it. |
| Per-slot win counts | `last_three_/historical_bid_success_count_slot_{1,2,3}` | `obs_60` has per-slot *costs* but no per-slot *win counts*. Tells the policy which slots it's actually winning. |
| Per-slot exposure rates | `last_three_/historical_exposure_mean_slot_{1,2,3}` | Whether winning impressions actually get shown to users, by slot. |
| Raw conversion volume | `last_three_/historical_conversion_count` | Per-tick conversion counts (vs the `mean`-only views in `obs_60`). |
| Episode cumulatives | `total_conversions`, `total_cost`, `total_cpa` | Where we stand on the three scoring quantities across the full episode. |

### Tier 2 — new derived features (adds ~80 lines to `_get_state_dict`)

Computed from existing `history_info` data — no new simulator plumbing, no
change to auction dynamics.

**Chosen Tier 2 additions (18 keys)**, grouped by signal:

| Group | Keys | Signal |
|---|---|---|
| **Competitor distribution shape** | `lwc_trend_slope_last_five`, `lwc_std_last_five`, `lwc_cv_last_five`, `lwc_z_last`, `lwc_pct_gap_last`, `lwc_top_to_mean_last`, `lwc_pct_gap_trend_last_five` | How competitor bids are moving (trend), how volatile they are (std/cv), how unusual now is vs history (z), and how much the right tail (`01_pct`) pulls away from the typical bid (`10_pct`) |
| **Opponent aggression proxies** | `opponent_aggression_last` (=`lwc_mean`/`pvalues_mean`), `bid_competitiveness_last` (=our `bid_mean`/`lwc_mean`) | Since we can't observe opponent bids directly, infer aggression from auction outcomes. |
| **Slot dynamics** | `cost_slot_1_over_slot_3_last`, `exposure_ratio_top_to_floor_last`, `slot_concentration_trend_last_five` | Ratio of top vs floor slot cost/exposure — shifts signal concentration of competition. |
| **Regime z-scores** | `pv_intensity_z_last`, `win_rate_z_last` | How far the current tick is from the episode mean in traffic volume and win rate. |
| **Momentum / acceleration** | `win_rate_trend_slope_last_five`, `pv_over_lwc_trend_slope_last_five`, `conversion_rate_trend_last_five` | Are our competitiveness, ROI, and conversion productivity climbing or falling? |
| **Pacing** | `spend_velocity_residual` = `(total_cost/budget) − (1−time_left)` | Positive ⇒ overpacing budget, negative ⇒ under-pacing. |

All Tier 2 derivations use existing `self.history_info` entries plus two
local helpers defined inline: `_slope(xs)` (least-squares slope over a
window) and `_std(xs)` (safe std).

### Combined config

`configs/obs_market_regime_v1.json` = obs_60 (60) + Tier 1 (20) + Tier 2
(18) = **98 keys**.

`obs_16_keys.json` and `obs_60_keys.json` are untouched; existing runs that
reference them remain reproducible.

## 3. What changed (practical)

Three files on `bl/market-regime`:

### 3.1 `bidding_train_env/online/online_env.py`

One new block inside `BiddingEnv._get_state_dict`, added after the existing
deprecated-key alias block and before `return state_dict`. Two local helpers
(`_slope`, `_std`) and ~60 lines that populate Tier 2 keys from existing
`self.history_info` values and scalar episode counters.

**Not touched**: `step`, `reset`, `_place_bids`, `_update_history`,
`_load_pvalues_df`, `_load_bids_df`, or the action space / reward logic.

### 3.2 `bidding_train_env/online/configs/obs_market_regime_v1.json`

New file. Lists the 98 keys that become the policy's observation vector
when `--obs_type obs_market_regime_v1` is passed to `main_train_ppo.py`.

### 3.3 `slurm/obs_audit_5m.slurm` / `obs_audit_2m.slurm` / `obs_audit_5m_all.slurm`

Reusable parameterised scripts that take `OBS_TYPE=<config name>` and train
PPO for 5M / 2M / 5M (on the `all` partition) steps. Submit one per obs
config to compare.

## 4. How to run

All commands from the repo root. Training and eval use Adroit Slurm on the
CPU `class` partition (or `all` for heavier parallel jobs).

### Quick-signal run (2M steps, ~1.5 h on class, 10 envs)

```bash
sbatch --export=ALL,OBS_TYPE=obs_market_regime_v1 \
    strategy_train_env/slurm/obs_audit_2m.slurm
```

Writes to `output/online/training/ongoing/021_quick_obs_market_regime_v1_ppo_seed_0_2m/`.

### Full run (5M steps, ~2.5 h on `all` partition, 20 envs)

```bash
sbatch --export=ALL,OBS_TYPE=obs_market_regime_v1 \
    strategy_train_env/slurm/obs_audit_5m_all.slurm
```

Writes to `022_full_obs_market_regime_v1_ppo_seed_0_5m_all/`.

### Evaluation

```bash
RUN_NAME=022_full_obs_market_regime_v1_ppo_seed_0_5m_all \
OBS_TYPE=obs_market_regime_v1 \
    sbatch strategy_train_env/slurm/eval_ppo.slurm
```

Writes to `output/online/testing/022_full_obs_market_regime_v1_ppo_seed_0_5m_all/results_{random,sweep}_<ts>.json`.

Compare `score.mean` (sweep mode) against the `obs_60_keys` baseline run at
matched step count.

## 5. Workflow placement

This extension sits at the **observation layer** of the pipeline:

```
  BiddingEnv (simulator, untouched)
        │
        ▼
  _get_state_dict  ← + ~80 lines of Tier 2 derivations
        │
        ▼
  obs_market_regime_v1.json  ← new file, selects 98 named values
        │
        ▼
  PPO policy network  ← now receives 98-dim obs instead of 60-dim
```

Nothing between `_place_bids` and reward emission changes; the policy just
sees more of what the env already knows.

## 6. Relation to Phase 2.0 (obs_60 audit)

A separate set of ablation runs (`obs_16_keys`, `obs_60_drop_slot`,
`obs_60_drop_lwc_ratio`, `obs_60_drop_distribution`, each 5M steps)
measures which of `obs_60`'s 44-feature upgrade over `obs_16` carries the
training signal. Results from that audit inform **feature selection** in
the next iteration (`obs_market_regime_v2.json`): if a Tier 2 group
(e.g. distribution shape) aligns with the load-bearing `obs_60` group, we
retain and expand; if the ablation shows a group is redundant with what
`obs_60` already provides, we drop it to keep the obs vector compact.

## 7. Expected results / next steps

- **Minimum success**: `obs_market_regime_v1` sweep-mode `score.mean` is
  within 1 SEM of the obs_60 baseline (neutral — added features don't
  hurt) at matched step count.
- **Target**: statistically meaningful `score.mean` improvement on sweep
  eval vs obs_60 baseline, paired with lower `cost_over_budget` variance
  (cleaner pacing) or `target_cpa_over_cpa` closer to 1.0.
- **Next iteration**: use permutation importance on a trained v1 model to
  rank Tier 2 contributions, then produce `obs_market_regime_v2.json` with
  the 8–12 most useful additions (keeping total ≤ 75 keys). A leaner obs
  often trains faster and generalises better.

## 8. Files touched — one-line summary

| File | Purpose | Status |
|---|---|---|
| `bidding_train_env/online/online_env.py` | `+80 lines` in `_get_state_dict` for Tier 2 derivations | modified |
| `bidding_train_env/online/configs/obs_market_regime_v1.json` | 98-key obs selector | new |
| `slurm/obs_audit_2m.slurm`, `obs_audit_5m_all.slurm` | Parameterised training scripts (5M / 2M, class / all) | new |
| `obs_16_keys.json`, `obs_60_keys.json` | Legacy configs | **unchanged** |
| `_update_history`, `_place_bids`, `step`, `reset` | Simulator dynamics | **unchanged** |
