# Slurm scripts (Adroit, CPU `class` partition)

PPO training/eval scripts for the online pipeline at
`bidding_train_env/online/`. All scripts assume:

- `module load anaconda3/2025.12 && conda activate auctionnet`
- repo at `/scratch/network/$USER/AuctionNet` (override with `AUCTIONNET_ROOT=...`)
- run from repo root (the scripts `cd` themselves)

See `specs/Adroit_HPC_Guide.md` for general Adroit conventions (`sbatch`,
`squeue`, `sacct`, partitions, etc.).

## Scripts

| Script | Purpose | Resources | Wall-clock |
|---|---|---|---|
| `prepare_data.slurm` | One-time: unzip raw parquets + run `prepare_data.py` | 2 cpus, 16G/cpu | ~30–90 min |
| `smoke_ppo_obs60.slurm` | 100K-step PPO sanity (4 envs) — verifies env, data, checkpointing | 4 cpus, 4G/cpu | ~20–40 min |
| `baseline_ppo_obs60_10m.slurm` | Full 10M PPO baseline (`obs_60_keys`, `lr=1e-4`, `bc_range default`) | 20 cpus, 4G/cpu | ~2–4 h |
| `eval_ppo.slurm` | Eval a trained run on period 27 (random + sweep modes). Templated via `RUN_NAME` env var | 2 cpus, 8G/cpu | ~20–40 min |

## Usage

From repo root:

```bash
# Phase 0: prep data once
sbatch strategy_train_env/slurm/prepare_data.slurm

# Phase 1: smoke first
sbatch strategy_train_env/slurm/smoke_ppo_obs60.slurm

# Phase 1: full baseline (after smoke clean)
sbatch strategy_train_env/slurm/baseline_ppo_obs60_10m.slurm

# Phase 1: eval the baseline
RUN_NAME=001_baseline_ppo_seed_0_ppo_default_obs60 \
    sbatch strategy_train_env/slurm/eval_ppo.slurm
```

Optional `eval_ppo.slurm` env vars: `OBS_TYPE` (default `obs_60_keys`),
`EVAL_MODE` (`random`/`sweep`/`both`, default `both`), `N_EVAL` (default 100).

## Outputs

- Slurm logs: `strategy_train_env/slurm_out/` (gitignored).
- Training artifacts: `output/online/training/ongoing/<RUN_NAME>/`
  (model zips, vecnormalize.pkl, args.json, env_config.json, rollout_log.jsonl).
- Eval artifacts: `output/online/testing/<RUN_NAME>/results_{random,sweep}_<ts>.json`.

## Resume

`main_train_ppo.py` auto-resumes from the latest checkpoint inside the run dir,
so on `TIMEOUT` just resubmit the same script.
