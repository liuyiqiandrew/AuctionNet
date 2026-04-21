# VeRL GRPO + LoRA finetune for the LLM bidding agent

Finetunes the Qwen3.5-9B bidding agent via GRPO on AuctionNet periods 7..26.
Val = period 27. LoRA is **required** on a single H200.

This repo now targets the VeRL API actually installed in `.venv`
(`verl 0.4.1`). The training path uses:

- VeRL async VLLM rollout (`rollout.mode=async`)
- a custom `completion_callback` that runs the 48-tick AuctionNet env
- a custom reward manager that reads terminal episode metrics from the rollout

## Layout

```
bidding_train_env/llm/
├── verl/
│   ├── auction_completion_callback.py   # async VLLM chat loop -> AuctionNet env
│   ├── auction_reward_manager.py        # rollout metrics -> token-level reward tensor
│   ├── build_rl_dataset.py              # one-off: train.parquet + val.parquet
│   ├── config/
│   │   └── qwen35_9b_grpo_lora.yaml
│   ├── launch_train.py                 # registers local extensions then calls VeRL
│   └── data/                           # built by build_rl_dataset.py
├── main_verl_train.sh            # SLURM entry point
└── README_verl.md                # this file
```

`prompt.SYSTEM_PROMPT`, `build_user_message`, and `parse_alpha` are reused
inside the completion callback so training and `main_eval_llm.py` still share
the same action contract.

## Prerequisites

- `.venv` has `verl` installed alongside `vllm==0.19.1`, `torch==2.10.*`.
- Model snapshot at
  `bidding_train_env/llm/models/Qwen3.5-9B/` (already used by eval).
- Constraints parquets under `data/traffic/online_rl_data/` for periods 7..27.

## Running

From `AuctionNet/strategy_train_env/`:

```
sbatch bidding_train_env/llm/main_verl_train.sh
```

The SLURM script:
1. Activates `.venv`, sets offline HF flags, puts the repo on `PYTHONPATH`.
2. Builds `verl/data/{train,val}.parquet` on first run.
3. Launches `python -m bidding_train_env.llm.verl.launch_train ...`.

Lightweight config-only smoke check from the login node:

```
python -m bidding_train_env.llm.verl.launch_train \
  --config-path=$(realpath bidding_train_env/llm/verl/config) \
  --config-name=qwen35_9b_grpo_lora \
  --cfg job
```

That prints the merged Hydra job config and exits without starting a rollout.

Live monitoring during training:

```
tail -f slurm_out/llm_rl_p7_26-<jobid>.out
nvidia-smi                        # peak usage should hover around 90 GB
```

## Thinking is disabled

Qwen3.5 emits thinking content by default. For RL we disable it in two places:

1. The async callback sends OpenAI-compatible requests with
   `chat_template_kwargs={"enable_thinking": false}`.
2. The callback monkey-patches the tokenizer's `apply_chat_template(...)` so
   postprocessed training tokens also use `enable_thinking=False`.

Verify once training is up:

```
grep -c '<think>' output/llm/training/*/rollout_*.jsonl    # should stay 0
```

## Evaluating the trained adapter

LoRA output lands under `output/llm/training/qwen35_9b_grpo_lora/`. Merge the
adapter into a fresh copy of the base weights, then point
`bidding_train_env/llm/main_eval_llm.py --model` at the merged directory.

Output schema is identical to the PPO sweep (`results_sweep_*.json`) so the
existing plotting scripts can overlay the curves directly.

## Expected signal in the first few steps

- `episode_score` should move above the frozen-base LLM baseline.
- malformed rate should stay low because invalid outputs are parsed as
  `alpha=0`.
- GPU memory should remain comfortably below full-card usage because actor
  training is LoRA-only and rollout runs with conservative KV-cache headroom.

## When things go wrong

- **OOM under FSDP all-gather** — drop `rollout.gpu_memory_utilization` to
  0.25 and `max_num_batched_tokens` to 8192.
- **High malformed rate (>10% after 2 epochs)** — add a format penalty to
  `bidding_reward.py` (e.g. `-0.1 * malformed_count`).
- **reward stuck at 0.0** — inspect `non_tensor_batch["reward_scores"]` in the
  dumped rollout data. The custom reward manager expects an `episode_score`
  scalar there.
