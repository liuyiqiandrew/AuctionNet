# LLM bidding agent (online replay eval)

Minimum-viable LLM-driven auto-bidding baseline. The LLM is prompted each of
the 48 episode ticks with the current state of the bidding problem and emits a
single log-bid-multiplier `alpha`. Bids are then computed by the env as
`bid_i = exp(alpha) * pvalue_i * target_cpa` — the same action semantics as
the PPO policy with `act_1_key`, so this baseline is apples-to-apples
comparable to `main_eval_ppo.py`'s `--eval_mode sweep` output.

## Layout

```
bidding_train_env/llm/
├── prompt.py          # SYSTEM_PROMPT, build_user_message(state_dict), parse_alpha(text)
├── backends.py        # LLMBackend protocol + VLLMOfflineBackend + VLLMServerBackend
├── agent.py           # BiddingAgent — backend-agnostic multi-env chat loop
├── main_eval_llm.py   # CLI: period-27 sweep, tick-synchronous batched rollout
├── main_verl_train.sh # SLURM entry point for VeRL GRPO + LoRA training
└── verl/
    ├── bidding_agent_loop.py       # VeRL rollout adapter for 48-tick episodes
    ├── auction_reward_manager.py   # rollout metrics -> token-level reward tensor
    ├── bidding_reward.py           # terminal AuctionNet episode score reward
    ├── build_rl_dataset.py         # train.parquet + val.parquet builder
    ├── launch_train.py             # registers local VeRL extensions, then launches
    └── config/qwen3_8b_grpo_lora.yaml
```

The agent never imports `vllm` or `openai` — all vendor dependencies live inside the
concrete backends. Swap backends with a CLI flag; the agent is unchanged.

## Pre-downloading model weights (compute nodes have no internet)

```
# From AuctionNet/strategy_train_env/
mkdir -p bidding_train_env/llm/models

# Full model (~15 GB, bf16 safetensors, use Qwen3.5-9B as an example)
hf download Qwen/Qwen3.5-9B --local-dir bidding_train_env/llm/models/Qwen3.5-9B
```

On the compute node, export `HF_HUB_OFFLINE=1` to hard-disable any HTTP
fallback (so a typo in `--model` fails loudly instead of silently hanging):

```
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

If you run `vllm serve` to back the `vllm_server` backend, start it with the
**same local path** you'll pass as `--model`:

```
vllm serve bidding_train_env/llm/models/Qwen3.5-9B --port 8000
# then in the client:
python bidding_train_env/llm/main_eval_llm.py \
    --backend vllm_server --model Qwen/Qwen3.5-9B --base_url ... 
# resolver rewrites --model to the local path; vllm serve reports the same path as the model name, so they match.
```

## Running

From `AuctionNet/strategy_train_env/`:

**vLLM offline** (in-process engine, one batched `LLM.chat` per tick):
```
python bidding_train_env/llm/main_eval_llm.py \
    --backend vllm_offline --model Qwen/Qwen3.5-9B \
    --eval_period 27 --out_prefix 001_qwen35_9b_
```

**vLLM server** (OpenAI-compatible HTTP against a running `vllm serve`):
```
# In another shell:
vllm serve Qwen/Qwen3.5-9B --port 8000

# Then:
python bidding_train_env/llm/main_eval_llm.py \
    --backend vllm_server --model Qwen/Qwen3.5-9B \
    --base_url http://localhost:8000/v1 \
    --eval_period 27 --out_prefix 001_qwen35_9b_server_
```

### Smoke test (1 advertiser, CPU, tiny model)
```
python bidding_train_env/llm/main_eval_llm.py \
    --backend vllm_offline --model Qwen/Qwen3.5-9B \
    --eval_period 27 --max_advertisers 1 --out_prefix smoke_
```

## Outputs

Written under `AuctionNet/output/llm/testing/{run_name}/`:

- `args_{ts}.json` — full CLI args snapshot
- `results_sweep_{ts}.json` — schema matches `main_eval_ppo.py`'s sweep output
  (`agg.score`, `agg.cost_over_budget`, `agg.target_cpa_over_cpa`, `per_episode[]`)
  plus `agg.is_malformed_rate` (fraction of ticks where the response didn't match
  `<alpha>X</alpha>`).
- `trajectories_{ts}.jsonl` — one line per (env, tick) with the full chat
  history, raw response, parsed alpha, reward, done, info. This is the data
  format a future VeRL training loop can consume directly.

## Prompt / action contract

The system prompt tells the model:

- bids are `exp(alpha) * pvalue * target_cpa`, so `alpha = 0` is neutral
- `alpha ∈ [-10, 10]` (clipped in code)
- score is `min(1, (target_cpa/realized_cpa)^2) * conversions` at episode end
- output format is strictly one line `<alpha>X</alpha>`

`parse_alpha` uses a strict regex; anything else falls back to `alpha=0` and
marks the turn `is_malformed=True`. The aggregate `is_malformed_rate` surfaces
how often the model went off-format — if this is > 5%, tighten the prompt or
switch to a more instruction-tuned model.

## VeRL GRPO + LoRA Training

The VeRL path fine-tunes Qwen3-8B on AuctionNet periods 7..26 with period 27
as validation. Dataset rows are one `(period, advertiser)` seed each; the
custom `bidding_agent_loop.py` reconstructs the full 48-tick episode live and
reuses `prompt.SYSTEM_PROMPT`, `build_user_message`, and `parse_alpha`, so eval
and RL training share the same action contract.

Use a separate `verl` conda environment. VeRL's CUDA, vLLM, FSDP, and optional
Megatron stack should stay isolated from the rest of `strategy_train_env`.
`main_verl_train.sh` activates `conda activate verl` explicitly.

Prerequisites:

- local model snapshot at `bidding_train_env/llm/models/Qwen3-8B/`
- replay data under `data/traffic/online_rl_data/` for periods 7..27
- CUDA/vLLM/VeRL installed in the `verl` conda env

One-time VeRL environment sketch:

```bash
conda create -n verl python==3.12
conda activate verl

# From a verl source checkout. The current config uses FSDP, so Megatron is optional.
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

python -c "import verl, vllm, torch; print(verl.__version__, vllm.__version__, torch.__version__)"
```

Build train/val parquet rows manually, or let the SLURM script build them on
first run:

```bash
python -m bidding_train_env.llm.verl.build_rl_dataset
```

Launch training from `AuctionNet/strategy_train_env/`:

```bash
sbatch bidding_train_env/llm/main_verl_train.sh
```

The SLURM script sets offline Hugging Face flags, exports `PYTHONPATH`, builds
missing `verl/data/{train,val}.parquet`, then launches:

```bash
python -u -m bidding_train_env.llm.verl.launch_train \
  --config-path="$(realpath bidding_train_env/llm/verl/config)" \
  --config-name=qwen3_8b_grpo_lora
```

Config-only smoke check from a login node:

```bash
python -m bidding_train_env.llm.verl.launch_train \
  --config-path="$(realpath bidding_train_env/llm/verl/config)" \
  --config-name=qwen3_8b_grpo_lora \
  --cfg job
```

Training logs go to `slurm_out/llm_rl_p7_26-<jobid>.out`; model outputs go to
`output/llm/training/qwen3_8b_grpo_lora/`.

Qwen3 thinking mode is disabled inside the rollout path. After training starts,
verify that generated rollouts do not contain `<think>`:

```bash
grep -c '<think>' output/llm/training/*/rollout_*.jsonl
```

To evaluate a trained adapter, merge the LoRA adapter into a fresh copy of the
base model, then point `main_eval_llm.py --model` at the merged directory. The
result schema matches PPO sweep output, so existing comparison scripts can read
`results_sweep_*.json`.

Common failure modes:

- OOM under FSDP all-gather: lower rollout GPU memory utilization and max
  batched tokens in the YAML config.
- High malformed rate: add an explicit malformed-output penalty in
  `bidding_reward.py`.
- Reward stuck at zero: inspect rollout `reward_scores`; the reward manager
  expects terminal `episode_score`.
