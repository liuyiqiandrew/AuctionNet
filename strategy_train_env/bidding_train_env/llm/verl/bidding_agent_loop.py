"""VeRL AgentLoop that runs our BiddingEnv for 48 ticks per rollout.

One rollout = one 48-tick episode for one (period, advertiser). Every tick:
    1. read env state via the existing `_get_state_dict` helper,
    2. render the user turn via `prompt.build_user_message`,
    3. extend `response_ids` with the user-turn tokens (mask = 0, not trained),
    4. call the async rollout server for an assistant turn,
    5. extend `response_ids` with the generated tokens (mask = 1, trained),
    6. parse alpha via `prompt.parse_alpha` and step the env,
    7. stop early if terminated/truncated.

Targets verl 0.8 `AgentLoopBase`. Returns `AgentLoopOutput(prompt_ids,
response_ids, response_mask, num_turns, reward_score=episode_score,
extra_fields={...})`. The trainer auto-fills `batch["rm_scores"]` from
`reward_score`, so no custom RewardManager is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

# verl spawns rollout workers from its own process root — make sure our repo
# is importable regardless of cwd.
_STRATEGY_TRAIN_ENV = Path(__file__).resolve().parents[3]
if str(_STRATEGY_TRAIN_ENV) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_TRAIN_ENV))

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

from bidding_train_env.llm.prompt import build_user_message, parse_alpha
from bidding_train_env.online.definitions import (
    EPISODE_LENGTH,
    RL_DATA_DIR,
    load_act_keys,
    load_obs_keys,
)
from bidding_train_env.online.online_env import EnvironmentFactory


@register("bidding_agent")
class BiddingAgentLoop(AgentLoopBase):
    """One rollout = one 48-tick AuctionNet episode for one advertiser."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self._obs_keys = load_obs_keys("obs_16_keys")
        self._act_keys = load_act_keys("act_1_key")

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        info = kwargs.get("extra_info") or {}
        period = int(info["period"])
        advertiser = int(info["advertiser"])
        budget = float(info["budget"])
        target_cpa = float(info["target_cpa"])
        seed = int(info.get("seed", 0))

        env = EnvironmentFactory.create(
            env_name="BiddingEnv",
            pvalues_df_path=str(RL_DATA_DIR / f"period-{period}_pvalues.parquet"),
            bids_df_path=str(RL_DATA_DIR / f"period-{period}_bids.parquet"),
            constraints_df_path=str(RL_DATA_DIR / f"period-{period}_constraints.parquet"),
            obs_keys=self._obs_keys,
            act_keys=self._act_keys,
            budget_range=None,
            target_cpa_range=None,
            seed=seed,
        )
        env.reset()
        env.unwrapped.set_campaign(
            advertiser=advertiser, budget=budget, target_cpa=target_cpa, period=period
        )

        request_id = uuid4().hex
        prompt_ids: list[int] = await self.apply_chat_template(messages)

        response_ids: list[int] = []
        response_mask: list[int] = []
        episode_score = 0.0
        malformed = 0
        n_turns = 0
        sampling_params = dict(sampling_params) if sampling_params else {}

        for tick in range(EPISODE_LENGTH):
            # Ask the rollout server for an assistant turn conditioned on the
            # current full chat.
            full_ids = await self.apply_chat_template(messages)
            gen = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=full_ids,
                sampling_params=sampling_params,
            )
            gen_ids = list(gen.token_ids)
            response_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))

            resp_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            alpha, is_malformed = parse_alpha(resp_text)
            if is_malformed:
                malformed += 1
            messages.append({"role": "assistant", "content": resp_text})
            n_turns = tick + 1

            _, _, terminated, truncated, step_info = env.step(
                np.array([alpha], dtype=np.float32)
            )
            if terminated or truncated:
                episode_score = float(step_info.get("score", 0.0))
                break

            # Render the next user turn (observation of new env state) and
            # append its tokens to response_ids with mask=0 so the policy sees
            # them at the next step but doesn't train on them.
            inner = env.unwrapped
            pv, sig = inner._get_pvalues_and_sigma()
            state = inner._get_state_dict(pv, sig)
            messages.append(build_user_message(state, inner.time_step, EPISODE_LENGTH))

            next_full_ids = await self.apply_chat_template(messages)
            delta = next_full_ids[len(prompt_ids) + len(response_ids):]
            response_ids.extend(delta)
            response_mask.extend([0] * len(delta))

            if len(response_ids) >= self.response_length:
                break

        env.close()

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=n_turns,
            reward_score=float(episode_score),
            metrics=AgentLoopMetrics(),
            extra_fields={
                "episode_score": float(episode_score),
                "malformed_count": int(malformed),
                "malformed_rate": float(malformed) / max(n_turns, 1),
            },
        )
