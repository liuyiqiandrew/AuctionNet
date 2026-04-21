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
└── main_eval_llm.py   # CLI: period-27 sweep, tick-synchronous batched rollout
```

The agent never imports `vllm` or `openai` — all vendor dependencies live inside the
concrete backends. Swap backends with a CLI flag; the agent is unchanged.

## Pre-downloading model weights (compute nodes have no internet)

Run the download on a **login node** (with internet) once. `main_eval_llm.py`
auto-resolves `--model Qwen/Qwen2.5-7B-Instruct` to the local snapshot
`bidding_train_env/llm/models/Qwen2.5-7B-Instruct/` when it exists, so the CLI
invocation is identical on login vs compute nodes.

```
# From AuctionNet/strategy_train_env/
mkdir -p bidding_train_env/llm/models

# Full model (~15 GB, bf16 safetensors)
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

## VeRL compatibility (future)

The agent is intentionally structured so VeRL can drive it without changes:
the per-turn record is `{messages, response, alpha, reward, done, info}`,
which tokenizes cleanly into VeRL's `AgentLoopOutput`
(`response_ids` + `response_mask` per assistant span, `reward_score` summed
over ticks). A future `verl_rollout.py` would only need to implement the
tokenization shim + a VeRL-owned `LLMBackend` that logs token ids and
logprobs. Nothing in `agent.py` or `main_eval_llm.py` needs to change.
