# LLM Bidding Agent — Evaluation and VeRL Fine-Tuning

This module is the LLM half of the AuctionNet auto-bidding solution
(see the [root README](../README.md) for project context). It provides:

1. **Prompted evaluation** of open-weight LLMs as auto-bidding agents over the
   replay environment, via a swappable `LLMBackend` (vLLM offline engine or
   OpenAI-compatible HTTP server).
2. **GRPO + LoRA fine-tuning** of the same models with [VeRL](https://github.com/volcengine/verl),
   using a custom rollout adapter that reuses the evaluation prompt and parser
   so training and inference share one action contract.

At every one of the 48 ticks in an episode the LLM is shown the current state
of the bidding problem and emits a single log-bid-multiplier `alpha`. The
environment then computes per-impression bids as
`bid_i = exp(alpha) * pvalue_i * target_cpa`. This is the same action
semantics as PPO with `act_1_key`, so LLM eval results are directly comparable
to `ppo/main_eval_ppo.py --eval_mode sweep` output.

## Prompt and Action Contract

The system prompt defines:

- Bidding rule: `bid_i = exp(alpha) * pvalue_i * target_cpa`. `alpha = 0` is
  neutral (bids equal to `pvalue * target_cpa`).
- Action range: `alpha ∈ [-10, 10]` (clipped in code on parse).
- Episode score: `min(1, (target_cpa / realized_cpa)^2) * conversions`,
  evaluated at episode end.
- Output format: a single line `<alpha>X</alpha>`. Anything else falls back to
  `alpha = 0` and the turn is flagged `is_malformed=True`.

The parser (`prompt.parse_alpha`) is intentionally strict; the aggregate
`is_malformed_rate` surfaces how often the model goes off-format. If this
exceeds ~5% on a given model, the prompt should be tightened or a more
instruction-tuned variant chosen. The same prompt and parser are reused inside
the VeRL rollout, so eval and RL training stay in sync.

## Layout

```text
llm/
├── prompt.py             # SYSTEM_PROMPT, build_user_message(state_dict), parse_alpha(text)
├── backends.py           # LLMBackend protocol + VLLMOfflineBackend + VLLMServerBackend
├── agent.py              # BiddingAgent — backend-agnostic multi-env chat loop
├── main_eval_llm.py      # CLI: tick-synchronous batched rollout for one period
├── main_verl_train.sh    # Bash entry point for VeRL GRPO + LoRA training
└── verl/
    ├── bidding_agent_loop.py     # VeRL rollout adapter for 48-tick episodes
    ├── auction_reward_manager.py # rollout metrics → token-level reward tensor
    ├── bidding_reward.py         # terminal AuctionNet episode-score reward
    ├── build_rl_dataset.py       # train.parquet / val.parquet builder
    ├── launch_train.py           # registers local VeRL extensions, then launches
    └── config/qwen3_8b_grpo_lora.yaml
```

The `BiddingAgent` never imports `vllm` or `openai`; vendor dependencies live
inside the concrete backends and are selected by a CLI flag.

## Environment and Model Snapshots

LLM evaluation and VeRL fine-tuning share a single conda environment with a
tightly pinned CUDA / vLLM / FSDP stack. Follow the
[VeRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html)
to set this up; reuse the same environment for `main_eval_llm.py` so that
local imports of `llm/*` work without additional setup.

### Pre-Downloading Model Weights

Compute nodes typically have no outbound internet, so model weights must be
fetched once from a login node:

```bash
mkdir -p llm/models
hf download Qwen/Qwen3-8B --local-dir llm/models/Qwen3-8B
```

On the compute node, hard-disable Hugging Face's HTTP fallback so a typo in
`--model` fails loudly instead of hanging:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

When using the `vllm_server` backend, start `vllm serve` with the same local
path that the client passes as `--model`. The CLI resolver rewrites
`--model` to the local path before sending the request, and `vllm serve`
reports the same path as the model name, so the two ends agree:

```bash
vllm serve llm/models/Qwen3-8B --port 8000
```

## Evaluation

All commands are run from the repository root.

**vLLM offline backend** (in-process engine, one batched `LLM.chat` per tick):

```bash
python llm/main_eval_llm.py \
    --backend vllm_offline \
    --model Qwen/Qwen3-8B \
    --eval_period 27 \
    --out_prefix 001_qwen3_8b_
```

**vLLM server backend** (OpenAI-compatible HTTP against a running `vllm serve`):

```bash
# In one shell:
vllm serve Qwen/Qwen3-8B --port 8000

# In another shell:
python llm/main_eval_llm.py \
    --backend vllm_server \
    --model Qwen/Qwen3-8B \
    --base_url http://localhost:8000/v1 \
    --eval_period 27 \
    --out_prefix 001_qwen3_8b_server_
```

**Smoke test** (one advertiser, useful for verifying the wiring on a small
allocation):

```bash
python llm/main_eval_llm.py \
    --backend vllm_offline \
    --model Qwen/Qwen3-8B \
    --eval_period 27 \
    --max_advertisers 1 \
    --out_prefix smoke_
```

### Outputs

Results land in `output/llm/testing/<run_name>/`:

- `args_{ts}.json` — full CLI argument snapshot.
- `results_sweep_{ts}.json` — schema matches `ppo/main_eval_ppo.py` sweep
  output (`agg.score`, `agg.cost_over_budget`, `agg.target_cpa_over_cpa`,
  `per_episode[]`), plus `agg.is_malformed_rate` for the off-format response
  fraction.
- `trajectories_{ts}.jsonl` — one line per `(env, tick)` with the full chat
  history, raw response, parsed alpha, reward, done, and info. This is the
  format consumed directly by the VeRL training loop.

## VeRL GRPO + LoRA Training

The default VeRL configuration fine-tunes `Qwen3-8B` on AuctionNet periods
7..26, with period 27 held out for validation. Each dataset row is one
`(period, advertiser)` seed; `verl/bidding_agent_loop.py` reconstructs the
full 48-tick episode online and reuses `prompt.SYSTEM_PROMPT`,
`build_user_message`, and `parse_alpha`, so the training-time and
evaluation-time action contracts remain identical.

### Prerequisites

- Local model snapshot at `llm/models/Qwen3-8B/`.
- Replay parquet files under `data/traffic/online_rl_data/` for periods 7..27
  (produced by `python ppo/prepare_data.py`; see the root README).
- VeRL conda environment with CUDA, vLLM, and FSDP installed (see the VeRL
  install guide above).

### Dataset

Build the train/validation parquet rows. The bash launcher below also
materializes them on first run if missing:

```bash
python -m llm.verl.build_rl_dataset
```

Outputs are written to `data/llm/verl/{train,val}.parquet`.

### Launching Training

```bash
bash llm/main_verl_train.sh
```

The bash script activates the `verl` conda environment, sets the offline
Hugging Face flags, exports `PYTHONPATH`, builds any missing dataset
parquets, and then launches:

```bash
python -u -m llm.verl.launch_train \
    --config-path="$(realpath llm/verl/config)" \
    --config-name=qwen3_8b_grpo_lora
```

Training logs stream to stdout/stderr; redirect them if you want a log file.
Checkpoints and rollout traces go to `output/llm/training/qwen3_8b_grpo_lora/`.

For a config-only smoke check on a login node:

```bash
python -m llm.verl.launch_train \
    --config-path="$(realpath llm/verl/config)" \
    --config-name=qwen3_8b_grpo_lora \
    --cfg job
```

Qwen3 thinking mode is disabled inside the rollout path. After training
starts, confirm rollouts do not contain `<think>` tags:

```bash
grep -c '<think>' output/llm/training/*/rollout_*.jsonl
```

### Evaluating a Trained Adapter

Merge the LoRA adapter into a fresh copy of the base model and point
`main_eval_llm.py --model` at the merged directory. The result schema is
unchanged, so PPO-vs-LLM comparison scripts read the resulting
`results_sweep_*.json` without modification.

### Common Failure Modes

- **OOM during FSDP all-gather.** Lower `rollout.gpu_memory_utilization` and
  `actor.ppo_max_token_len_per_gpu` in the YAML config.
- **High malformed-output rate.** Add an explicit malformed-output penalty in
  `verl/bidding_reward.py` so the policy is shaped toward the strict
  `<alpha>X</alpha>` format.
- **Reward stuck at zero.** Inspect the per-rollout `reward_scores`; the
  reward manager expects the terminal `episode_score` field populated by
  `bidding_agent_loop.py`.
