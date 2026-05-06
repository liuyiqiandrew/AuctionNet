# AuctionNet: PPO and LLM Fine-Tuning for Auto-Bidding

This repository contains a PPO-based and an LLM-fine-tuning solution to the
[NeurIPS 2024 Auto-Bidding in Uncertain Environment Competition](https://neurips.cc/virtual/2024/competition/84793).
The work was carried out as a group project for **COS 435 — Introduction to
Reinforcement Learning** at Princeton University, Spring 2026.

The motivating goal is to develop a generic reinforcement-learning approach to
long-horizon, multi-agent decision making, using ad auctions as a concrete
testbed. Two directions were explored:

1. **PPO extensions.** A replay-driven Stable-Baselines3 PPO agent that selects
   the scalar bidding action, augmented with temporal-difference observation
   stacking and a residual-GRU policy backbone, score-weighted advertiser
   sampling, reward shaping, and richer market-regime features.
2. **LLM fine-tuning.** Prompted evaluation of open-weight LLMs as bidding
   agents, plus GRPO + LoRA fine-tuning via [VeRL](https://github.com/volcengine/verl)
   on the same action contract.

## Environment Setup

### 1. Data Preparation

Two options are supported. The first is recommended.

**Option A — Pre-converted parquet (recommended).** Download the pre-converted
period parquets from this [Google Drive folder](https://drive.google.com/file/d/1yywYPH_X1ZHkeiDcGyERqtAzpbMyBO9l/view?usp=sharing)
and place them in `data/traffic/` as `period-{N}.parquet`.

**Option B — Original competition CSVs.** Download the raw CSVs from the
[official competition release](https://github.com/alimama-tech/AuctionNet/blob/main/pre_generated_dataset/readme_dataset.md)
and convert each to parquet (e.g. `pd.read_csv(...).to_parquet(...)` with
`pyarrow` engine). The expected output is one `data/traffic/period-{N}.parquet`
per period.

Once raw parquets are in place, build the replay-ready (pvalues, bids,
constraints) tables consumed by both the PPO replay environment and the LLM
agent:

```bash
python ppo/prepare_data.py --first_period 7 --last_period 27
```

This writes `period-{N}_pvalues.parquet`, `period-{N}_bids.parquet`, and
`period-{N}_constraints.parquet` into `data/traffic/online_rl_data/`.

### 2. PPO Environment

The project pins Python 3.12 and uses `uv` for dependency management
(`pyproject.toml` + `uv.lock`). Either of the two flows below produces a
working environment.

**With `uv` (recommended):**

```bash
# Install uv if you do not have it: https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Resolve and install the locked dependency set.
uv sync

# Activate the resulting environment.
source .venv/bin/activate
```

**With conda:**

```bash
conda create -n auctionnet python=3.12 -y
conda activate auctionnet
pip install -e .
```

The dependency set covers PPO training and evaluation
(`stable-baselines3`, `gymnasium`, `torch`, `pandas`, `pyarrow`, `vllm`).

### 3. LLM and VeRL Environment

LLM evaluation and VeRL GRPO + LoRA fine-tuning have a tighter CUDA / vLLM /
FSDP stack than the PPO path and should live in a **separate** conda
environment. Follow the official VeRL installation guide:

> [VeRL — Installation](https://verl.readthedocs.io/en/latest/start/install.html)

After VeRL is installed in its own environment, install the lightweight
project-side dependencies (`pandas`, `pyarrow`, this repo as a package) into
the same environment so that `llm/` modules are importable.

## PPO Training and Evaluation

### Train

The default-baseline PPO run uses 20 parallel replay environments, one per
period, the 16-key observation, and the 1-key (scalar bidding factor) action:

```bash
python ppo/main_train_ppo.py \
    --num_envs 20 \
    --num_steps 10000000 \
    --batch_size 512 \
    --seed 0 \
    --bc_range default \
    --obs_type obs_16_keys \
    --act_type act_1_key \
    --out_prefix 001_ \
    --out_suffix _ppo_default_obs16
```

Checkpoints land in `output/ppo/training/ongoing/<run_name>/` along with
`args.json`, `env_config.json`, the launcher snapshot, and per-checkpoint
`rl_model_*_steps.zip` / `rl_model_vecnormalize_*_steps.pkl` files.

The same entry point also supports temporal-difference observation stacking
(`--temporal_seq_len K --temporal_hidden_dim H`) and score-weighted advertiser
sampling (`--weighted-sampling --temperature T --alpha A`); the two flags are
orthogonal and can be combined. See [`ppo/README.md`](ppo/README.md) for the
extended flag matrix, reward-shaping options, and market-regime observation
runs.

### Evaluate

```bash
python ppo/main_eval_ppo.py \
    --load_path output/ppo/training/ongoing/001_ppo_seed_0_ppo_default_obs16 \
    --eval_mode both \
    --n_eval_episodes 100 \
    --eval_period 27 \
    --obs_type obs_16_keys \
    --act_type act_1_key
```

`--eval_mode random` runs `--n_eval_episodes` random
(advertiser, budget, target_cpa) rollouts under `--bc_range`; `--eval_mode
sweep` runs one rollout per advertiser in the eval period using the raw
`(budget, CPAConstraint)` from the constraints parquet; `both` runs both.
For temporal-PPO checkpoints, pass the same `--temporal_seq_len K` used at
training time. Results are written to `output/ppo/testing/<run_name>/`.

## LLM Evaluation and VeRL Training

### Evaluate a Local or Served Model

```bash
python llm/main_eval_llm.py \
    --backend vllm_offline \
    --model Qwen/Qwen3.5-9B \
    --eval_period 27 \
    --out_prefix 001_qwen35_9b_
```

Other supported backends (vLLM HTTP server, OpenAI-compatible endpoints, etc.)
and the prompt and action contract are documented in
[`llm/README.md`](llm/README.md).

### Build VeRL Training Rows

Convert replay data into the prompt/response/reward rows VeRL consumes:

```bash
python -m llm.verl.build_rl_dataset
```

Output parquets are written to `data/llm/verl/`.

### Launch VeRL GRPO + LoRA Training

```bash
bash llm/main_verl_train.sh
```

The bash script wraps the VeRL launcher with the project's GRPO + LoRA
configuration. See [`llm/README.md`](llm/README.md) for model snapshot setup, vLLM
server usage, prompt/action contract details, and training-time tuning notes.
