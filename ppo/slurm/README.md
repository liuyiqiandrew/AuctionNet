# Slurm Scripts

PPO training/eval scripts for the root `ppo/` package. All scripts assume:

- `module load anaconda3/2025.12 && conda activate auctionnet`
- repo root is the current checkout, or override with `AUCTIONNET_ROOT=...`
- run from repo root (the scripts `cd` themselves)

Use the appropriate cluster partition/account flags for your environment.

## Scripts

| Script | Purpose | Resources | Wall-clock |
|---|---|---|---|
| `prepare_data.slurm` | One-time: unzip raw parquets + run `prepare_data.py` | 2 cpus, 16G/cpu | ~30–90 min |
| `baseline_ppo_obs60_10m.slurm` | Full 10M PPO baseline (`obs_60_keys`, `lr=1e-4`, `bc_range default`) | 20 cpus, 4G/cpu | ~2–4 h |
| `eval_ppo.slurm` | Eval a trained run on period 27 (random + sweep modes). Templated via `RUN_NAME` env var | 2 cpus, 8G/cpu | ~20–40 min |

## Usage

From repo root:

```bash
# Phase 0: prep data once
sbatch ppo/slurm/prepare_data.slurm

# Phase 1: full baseline (after smoke clean)
sbatch ppo/slurm/baseline_ppo_obs60_10m.slurm

# Phase 1: eval the baseline
RUN_NAME=001_baseline_ppo_seed_0_ppo_default_obs60 \
    sbatch ppo/slurm/eval_ppo.slurm
```

Optional `eval_ppo.slurm` env vars: `OBS_TYPE` (default `obs_60_keys`),
`EVAL_MODE` (`random`/`sweep`/`both`, default `both`), `N_EVAL` (default 100).

## Outputs

- Slurm logs: `slurm_out/` (gitignored).
- Training artifacts: `output/ppo/training/ongoing/<RUN_NAME>/`
  (model zips, vecnormalize.pkl, args.json, env_config.json, rollout_log.jsonl).
- Eval artifacts: `output/ppo/testing/<RUN_NAME>/results_{random,sweep}_<ts>.json`.

## Resume

`main_train_ppo.py` auto-resumes from the latest checkpoint inside the run dir,
so on `TIMEOUT` just resubmit the same script.
