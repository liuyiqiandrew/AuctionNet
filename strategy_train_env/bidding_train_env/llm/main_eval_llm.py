"""Online LLM auto-bidding eval on period 27 (or any period with constraints parquet).

Mirrors `bidding_train_env/online/main_eval_ppo.py`'s `sweep` mode, but the
"policy" is an LLM prompted each tick to emit a log-bid-multiplier `alpha`.
Runs one deterministic 48-tick episode per advertiser in the eval period, all
in parallel via tick-synchronous batched chat calls through `LLMBackend`.

Example (vLLM offline):
    python bidding_train_env/llm/main_eval_llm.py \\
        --backend vllm_offline --model Qwen/Qwen2.5-7B-Instruct \\
        --eval_period 27 --out_prefix 001_qwen25_7b_

Example (OpenAI-compatible HTTP to a running `vllm serve`):
    python bidding_train_env/llm/main_eval_llm.py \\
        --backend vllm_server --model Qwen/Qwen2.5-7B-Instruct \\
        --base_url http://localhost:8000/v1 \\
        --eval_period 27 --out_prefix 001_qwen25_7b_server_
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make `bidding_train_env` importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bidding_train_env.llm.agent import BiddingAgent
from bidding_train_env.llm.backends import SamplingSpec, build_backend
from bidding_train_env.online.definitions import (
    AUCTIONNET_ROOT,
    EPISODE_LENGTH,
    RL_DATA_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.online_env import EnvironmentFactory

LLM_OUTPUT_DIR = AUCTIONNET_ROOT / "output" / "llm"
# Pre-downloaded HF snapshots live here — compute nodes have no internet, so
# `resolve_model_path` rewrites `--model Qwen/Qwen2.5-7B-Instruct` to the
# absolute local path when `models/Qwen2.5-7B-Instruct/` exists.
LLM_MODELS_DIR = Path(__file__).resolve().parent / "models"


def resolve_model_path(model: str, models_dir: Path) -> str:
    """Return a local snapshot path if one exists, else the original `model`.

    - Absolute/relative filesystem path that exists: returned unchanged.
    - HF repo id like "Qwen/Qwen2.5-7B-Instruct": if `models_dir/Qwen2.5-7B-Instruct`
      is a directory, return its absolute path. Else return the repo id (lets
      vLLM fall back to the HF cache / HTTP when internet is available).
    """
    p = Path(model)
    if p.exists():
        return str(p.resolve())
    basename = model.rstrip("/").split("/")[-1]
    candidate = models_dir / basename
    if candidate.is_dir():
        return str(candidate.resolve())
    return model


def parse_args():
    p = argparse.ArgumentParser()

    # Backend
    p.add_argument("--backend", choices=["vllm_offline", "vllm_server"], default="vllm_offline")
    p.add_argument("--model", required=True,
                   help="HF repo id, served model name, or local path. "
                        "Repo ids are auto-resolved to `--models_dir/<basename>` if present.")
    p.add_argument("--models_dir", type=str, default=str(LLM_MODELS_DIR),
                   help="Directory of pre-downloaded HF snapshots (compute nodes have no internet)")
    p.add_argument("--no_resolve_model", action="store_true",
                   help="Pass --model through to the backend verbatim (skip local-snapshot resolution). "
                        "Use when --backend vllm_server and the server was launched with a specific path/name "
                        "that must be matched exactly.")
    p.add_argument("--base_url", default=None, help="Only for --backend vllm_server")
    p.add_argument("--api_key", default="EMPTY", help="Only for --backend vllm_server")
    p.add_argument("--max_workers", type=int, default=32,
                   help="Thread pool size for vllm_server batching")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Per-request timeout in seconds for the vllm_server backend. "
                        "Raise when --max_tokens is large (reasoning models can take minutes).")
    p.add_argument("--no_thinking", action="store_true",
                   help="Disable reasoning-mode tokens by passing "
                        "chat_template_kwargs={'enable_thinking': False}. "
                        "Required for Qwen3/3.5 so `max_tokens` isn't consumed by <think>...</think>.")

    # Sampling
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # vLLM engine knobs (only used for vllm_offline; ignored otherwise)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=1)

    # Env / eval
    p.add_argument("--eval_period", type=int, default=27)
    p.add_argument("--obs_type", default="obs_16_keys")
    p.add_argument("--act_type", default="act_1_key")
    p.add_argument("--rl_data_dir", type=str, default=str(RL_DATA_DIR))
    p.add_argument("--max_advertisers", type=int, default=None,
                   help="Cap the sweep to the first N advertisers (for smoke tests)")
    p.add_argument("--deterministic_conversion", action="store_true")

    # Output
    p.add_argument("--out_prefix", default="")
    p.add_argument("--out_suffix", default="")
    p.add_argument("--run_name", default=None,
                   help="Override the auto-generated run directory name")

    return p.parse_args()


def build_env(period, obs_keys, act_keys, rl_data_dir, seed, deterministic_conversion=False):
    """Single raw BiddingEnv (no VecNormalize wrapping — the LLM sees the state dict directly)."""
    rl_data_dir = Path(rl_data_dir)
    env = EnvironmentFactory.create(
        env_name="BiddingEnv",
        pvalues_df_path=str(rl_data_dir / f"period-{period}_pvalues.parquet"),
        bids_df_path=str(rl_data_dir / f"period-{period}_bids.parquet"),
        constraints_df_path=str(rl_data_dir / f"period-{period}_constraints.parquet"),
        obs_keys=obs_keys,
        act_keys=act_keys,
        # bc_range="default": env reads raw budget/CPAConstraint from the parquet at reset.
        budget_range=None,
        target_cpa_range=None,
        deterministic_conversion=deterministic_conversion,
        seed=seed,
    )
    return env


def _inner(env):
    """Strip the gym.wrappers.TimeLimit / OrderEnforcing layers gymnasium.make adds."""
    return env.unwrapped


def _state_dict(env):
    inner = _inner(env)
    pvalues, sigma = inner._get_pvalues_and_sigma()
    return inner._get_state_dict(pvalues, sigma)


def _json_encoder(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return "<ns>"


def aggregate(infos, keys=("score", "cost_over_budget", "target_cpa_over_cpa",
                            "conversions", "cpa", "total_reward", "is_malformed_rate")):
    out = {}
    for k in keys:
        xs = np.array([i.get(k, 0.0) for i in infos], dtype=np.float64)
        n = len(xs)
        out[k] = {
            "mean": float(xs.mean()) if n else 0.0,
            "std": float(xs.std(ddof=0)) if n else 0.0,
            "sem": float(xs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n": int(n),
        }
    return out


def main():
    args = parse_args()

    obs_keys = load_obs_keys(args.obs_type)
    act_keys = load_act_keys(args.act_type)

    # --- Load eval-period constraints to enumerate the advertisers to sweep. ---
    rl_data_dir = Path(args.rl_data_dir)
    cons_path = rl_data_dir / f"period-{args.eval_period}_constraints.parquet"
    cons = pd.read_parquet(cons_path)
    cons = cons[cons.deliveryPeriodIndex == args.eval_period].reset_index(drop=True)
    if args.max_advertisers is not None:
        cons = cons.head(int(args.max_advertisers)).reset_index(drop=True)
    n_envs = len(cons)
    if n_envs == 0:
        raise RuntimeError(f"No advertisers found for period={args.eval_period} in {cons_path}")
    print(f"[llm-eval] sweep size: {n_envs} advertisers on period {args.eval_period}")

    # --- Build envs and seat each one at its advertiser's raw (budget, CPA). ---
    envs = []
    for i, row in cons.iterrows():
        env = build_env(
            period=args.eval_period, obs_keys=obs_keys, act_keys=act_keys,
            rl_data_dir=args.rl_data_dir, seed=args.seed + int(i),
            deterministic_conversion=args.deterministic_conversion,
        )
        # gym.make wraps BiddingEnv in TimeLimit/OrderEnforcing, which track their
        # own "has been reset?" state. Call reset() first, then override the
        # campaign params on the unwrapped env (same pattern as main_eval_ppo.py).
        env.reset()
        _inner(env).set_campaign(
            advertiser=int(row.advertiserNumber),
            budget=float(row.budget),
            target_cpa=float(row.CPAConstraint),
            period=int(args.eval_period),
        )
        envs.append(env)

    # --- Build LLM backend + agent. ---
    backend_kwargs: dict = {}
    if args.backend == "vllm_offline":
        backend_kwargs.update(
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )
        if args.max_model_len is not None:
            backend_kwargs["max_model_len"] = int(args.max_model_len)
    elif args.backend == "vllm_server":
        if not args.base_url:
            raise ValueError("--base_url is required for --backend vllm_server")
        backend_kwargs.update(
            base_url=args.base_url, api_key=args.api_key,
            max_workers=args.max_workers, timeout=args.timeout,
        )
    # Prefer a local snapshot under --models_dir if present. For vllm_offline the
    # resolved path is loaded directly by vllm.LLM. For vllm_server the resolved
    # path must match what `vllm serve <path>` reports as the model name — if
    # you started the server with the HF repo id, pass --model <repo_id> too.
    if args.no_resolve_model:
        resolved_model = args.model
        print(f"[llm-eval] --no_resolve_model set; using --model verbatim: {resolved_model!r}",
              flush=True)
    else:
        resolved_model = resolve_model_path(args.model, Path(args.models_dir))
        if resolved_model != args.model:
            print(f"[llm-eval] resolved --model {args.model!r} -> local path {resolved_model}",
                  flush=True)
    print(f"[llm-eval] building backend={args.backend} model={resolved_model} ...",
          flush=True)
    t_backend = time.time()
    backend = build_backend(args.backend, model=resolved_model, **backend_kwargs)
    print(f"[llm-eval] backend ready in {time.time() - t_backend:.1f}s", flush=True)

    sampling = SamplingSpec(
        temperature=args.temperature, max_tokens=args.max_tokens,
        top_p=args.top_p, seed=args.seed,
        chat_template_kwargs=({"enable_thinking": False} if args.no_thinking else {}),
    )
    agent = BiddingAgent(backend=backend, sampling=sampling, episode_length=EPISODE_LENGTH)
    agent.reset(n_envs=n_envs)

    # --- Output dir ---
    run_name = args.run_name or (
        f"{args.out_prefix}llm_seed_{args.seed}{args.out_suffix}"
        or f"llm_seed_{args.seed}"
    )
    out_dir = LLM_OUTPUT_DIR / "testing" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    traj_path = out_dir / f"trajectories_{ts}.jsonl"
    # Clear the file if it exists so each run writes a fresh trajectory stream.
    traj_path.write_text("")

    # Human-readable live log: one line per tick + sampled LLM responses so
    # you can `tail -f` during a slurm run without parsing JSONL.
    live_log_path = out_dir / f"live_log_{ts}.txt"
    live_log_path.write_text("")
    live_log = open(live_log_path, "a", buffering=1)  # line-buffered
    live_log.write(f"[init] run={run_name} ts={ts} model={resolved_model}\n")

    # Snapshot run args for reproducibility; include the resolved local path.
    args_dump = vars(args) | {"resolved_model_path": resolved_model}
    with open(out_dir / f"args_{ts}.json", "w") as f:
        json.dump(args_dump, f, indent=2, default=_json_encoder)

    # --- Tick-synchronous rollout. ---
    done = [False] * n_envs
    final_infos: list[dict] = [None] * n_envs  # type: ignore
    total_rewards = [0.0] * n_envs
    malformed_counts = [0] * n_envs

    def _trunc(s: str, n: int = 80) -> str:
        s = " ".join((s or "").split())
        return s if len(s) <= n else s[: n - 1] + "…"

    print(f"[llm-eval] entering rollout: n_envs={n_envs} episode_length={EPISODE_LENGTH}",
          flush=True)
    t_start = time.time()
    for tick in range(EPISODE_LENGTH):
        alive = [i for i in range(n_envs) if not done[i]]
        if not alive:
            break

        tick_t0 = time.time()
        if tick == 0:
            print(f"[tick 00] collecting state dicts for {len(alive)} envs ...", flush=True)
        state_dicts = [_state_dict(envs[i]) for i in alive]
        if tick == 0:
            print(f"[tick 00] state dicts built in {time.time() - tick_t0:.2f}s; "
                  f"calling backend.chat on {len(alive)} convs (max_tokens={args.max_tokens}, "
                  f"max_model_len={args.max_model_len}) ...", flush=True)
        ticks = [tick] * len(alive)
        gens = agent.act(state_dicts, ticks, env_indices=alive)
        gen_s = time.time() - tick_t0
        if tick == 0:
            print(f"[tick 00] backend.chat returned in {gen_s:.2f}s", flush=True)

        # Step each alive env with its parsed alpha; log per-turn record.
        malformed_this_tick = 0
        with open(traj_path, "a") as fout:
            for i, g in zip(alive, gens):
                alpha = g["alpha"]
                action = np.array([alpha], dtype=np.float32)
                _, reward, terminated, truncated, info = envs[i].step(action)
                d = bool(terminated or truncated)

                total_rewards[i] += float(reward)
                if g["is_malformed"]:
                    malformed_counts[i] += 1
                    malformed_this_tick += 1
                if d:
                    done[i] = True
                    final_infos[i] = dict(info)  # snapshot terminal info

                rec = {
                    "env_index": int(i),
                    "advertiser": int(cons.advertiserNumber.iloc[i]),
                    "tick": int(tick),
                    "alpha": float(alpha),
                    "is_malformed": bool(g["is_malformed"]),
                    "reward": float(reward),
                    "done": d,
                    "response": g["response"],
                    "messages": g["messages"],
                    "info": info,
                }
                fout.write(json.dumps(rec, default=_json_encoder) + "\n")

        # Per-tick summary: alive count, timing, alpha range, malformed rate.
        alphas = [float(g["alpha"]) for g in gens]
        a_min, a_mean, a_max = min(alphas), sum(alphas) / len(alphas), max(alphas)
        summary = (
            f"[tick {tick:02d}/{EPISODE_LENGTH}] alive={len(alive)} "
            f"gen={gen_s:.2f}s total={time.time() - t_start:.1f}s "
            f"alpha(min/mean/max)={a_min:+.2f}/{a_mean:+.2f}/{a_max:+.2f} "
            f"malformed={malformed_this_tick}/{len(alive)}"
        )
        print(summary, flush=True)
        live_log.write(summary + "\n")

        # Sample up to 3 responses to show what the model is saying.
        for k, (i, g) in enumerate(zip(alive, gens)):
            if k >= 3:
                break
            adv = int(cons.advertiserNumber.iloc[i])
            live_log.write(
                f"    adv={adv:4d} alpha={g['alpha']:+.3f} "
                f"resp={_trunc(g['response'])!r}\n"
            )

    # --- Aggregate. ---
    per_ep = []
    for i in range(n_envs):
        info = final_infos[i] or {}
        info = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in info.items()}
        info["total_reward"] = total_rewards[i]
        info["is_malformed_rate"] = malformed_counts[i] / float(EPISODE_LENGTH)
        info["advertiser"] = int(cons.advertiserNumber.iloc[i])
        info["budget"] = float(cons.budget.iloc[i])
        info["target_cpa"] = float(cons.CPAConstraint.iloc[i])
        per_ep.append(info)

    result = {
        "mode": "sweep",
        "n": n_envs,
        "eval_period": args.eval_period,
        "model": args.model,
        "backend": args.backend,
        "sampling": vars(sampling),
        "agg": aggregate(per_ep),
        "per_episode": per_ep,
    }
    results_path = out_dir / f"results_sweep_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_encoder)

    agg = result["agg"]
    print(
        f"[sweep] score_mean={agg['score']['mean']:.4f} "
        f"cost_over_budget_mean={agg['cost_over_budget']['mean']:.4f} "
        f"target_cpa_over_cpa_mean={agg['target_cpa_over_cpa']['mean']:.4f} "
        f"malformed_rate_mean={agg['is_malformed_rate']['mean']:.4f} "
        f"n={n_envs}"
    )
    print(f"[saved] {results_path}")
    print(f"[saved] {traj_path}")
    print(f"[saved] {live_log_path}")
    live_log.write(
        f"[done] score_mean={result['agg']['score']['mean']:.4f} "
        f"malformed_rate_mean={result['agg']['is_malformed_rate']['mean']:.4f}\n"
    )
    live_log.close()


if __name__ == "__main__":
    main()
