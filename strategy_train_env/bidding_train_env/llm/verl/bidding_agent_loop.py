"""VeRL AgentLoop that runs our BiddingEnv for 48 ticks per rollout.

One rollout = one 48-tick episode for one (period, advertiser). Every tick:
    1. read env state via the existing `_get_state_dict` helper,
    2. render the user turn via `prompt.build_user_message`,
    3. extend `response_ids` with the user-turn tokens (mask = 0, not trained),
    4. call the async rollout server for an assistant turn,
    5. extend `response_ids` with the generated tokens (mask = 1, trained),
    6. parse α via `prompt.parse_alpha` and step the env,
    7. stop early if terminated/truncated.

At the end we return `AgentLoopOutput(prompt_ids, response_ids, response_mask,
num_turns, metrics={"episode_score": float})`. The reward function reads
`episode_score` back.

Thinking (`<think>...</think>`) is disabled at three layers (see plan §2a):
    - YAML `actor_rollout_ref.rollout.chat_template_kwargs.enable_thinking=false`
    - explicit `chat_template_kwargs={"enable_thinking": False}` on every
      `tokenizer.apply_chat_template` call here
    - YAML rollout `stop: ["<think>"]` as a belt-and-braces safeguard.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Let this module be importable from an arbitrary cwd: verl spawns rollout
# workers under its own process root, so we need the repo on sys.path.
_STRATEGY_TRAIN_ENV = Path(__file__).resolve().parents[3]
if str(_STRATEGY_TRAIN_ENV) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_TRAIN_ENV))

# verl API — imports are lazy-tolerant so the module can be imported for
# unit tests even without verl installed (the actual @register decorator
# only runs when verl is present, which is the only case anyone will run
# this code in).
try:
    from verl.experimental.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "verl is required to use BiddingAgentLoop. Install a verl version "
        "that provides `verl.experimental.agent_loop.{AgentLoopBase, "
        "AgentLoopOutput, register}`."
    ) from e

from bidding_train_env.llm.prompt import (
    SYSTEM_PROMPT,
    build_user_message,
    parse_alpha,
)
from bidding_train_env.online.definitions import (
    EPISODE_LENGTH,
    RL_DATA_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.online_env import EnvironmentFactory

_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


@register("bidding_agent")
class BiddingAgentLoop(AgentLoopBase):
    """One rollout = one 48-tick episode for one advertiser."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        info = kwargs["extra_info"]
        period = int(info["period"])
        advertiser = int(info["advertiser"])
        budget = float(info["budget"])
        target_cpa = float(info["target_cpa"])

        obs_keys = load_obs_keys("obs_16_keys")
        act_keys = load_act_keys("act_1_key")
        env = EnvironmentFactory.create(
            env_name="BiddingEnv",
            pvalues_df_path=str(RL_DATA_DIR / f"period-{period}_pvalues.parquet"),
            bids_df_path=str(RL_DATA_DIR / f"period-{period}_bids.parquet"),
            constraints_df_path=str(RL_DATA_DIR / f"period-{period}_constraints.parquet"),
            obs_keys=obs_keys,
            act_keys=act_keys,
            budget_range=None,
            target_cpa_range=None,
            seed=0,
        )
        env.reset()
        env.unwrapped.set_campaign(
            advertiser=advertiser, budget=budget, target_cpa=target_cpa, period=period
        )

        tokenizer = self.tokenizer
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        prompt_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            chat_template_kwargs=_CHAT_TEMPLATE_KWARGS,
        )

        response_ids: list[int] = []
        response_mask: list[int] = []
        episode_score = 0.0
        n_turns = 0
        malformed = 0

        for tick in range(EPISODE_LENGTH):
            inner = env.unwrapped
            pv, sig = inner._get_pvalues_and_sigma()
            state = inner._get_state_dict(pv, sig)
            messages.append(build_user_message(state, tick, EPISODE_LENGTH))

            # Re-render the full chat to get the incremental user-turn tokens
            # (plus the assistant-prefix trailer). Slice off everything we've
            # already emitted so we only append the delta to response_ids.
            full_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                chat_template_kwargs=_CHAT_TEMPLATE_KWARGS,
            )
            delta = full_ids[len(prompt_ids) + len(response_ids):]
            response_ids.extend(delta)
            response_mask.extend([0] * len(delta))  # env/user tokens — no gradient

            # Hand the full current context to the rollout server.
            gen = await self.server_manager.generate(
                request_id=self.request_id,
                prompt_ids=full_ids,
                sampling_params=sampling_params,
            )
            response_ids.extend(gen)
            response_mask.extend([1] * len(gen))  # policy-generated tokens — trained

            resp_text = tokenizer.decode(gen, skip_special_tokens=True)
            alpha, is_malformed = parse_alpha(resp_text)
            if is_malformed:
                malformed += 1
            messages.append({"role": "assistant", "content": resp_text})

            _, _, terminated, truncated, step_info = env.step(
                np.array([alpha], dtype=np.float32)
            )
            n_turns = tick + 1

            if terminated or truncated:
                # Terminal info carries `score = min(1, (target_cpa/cpa)^2) * conversions`.
                episode_score = float(step_info.get("score", 0.0))
                break

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=n_turns,
            metrics={
                "episode_score": float(episode_score),
                "malformed_count": int(malformed),
                "malformed_rate": float(malformed) / max(n_turns, 1),
            },
        )
