"""AuctionNet multi-turn callback for VeRL's async VLLM chat scheduler.

This callback turns VeRL's OpenAI-style chat rollout into a 48-tick AuctionNet
episode:

1. The initial prompt already contains the system message and tick-0 user state.
2. Each assistant turn is parsed into `alpha`.
3. The AuctionNet replay env steps with that alpha.
4. If the episode continues, the next user state is appended and generation is
   resubmitted.
5. Post-processing reconstructs token ids plus a loss mask that trains only on
   assistant tokens, while storing terminal episode metrics for the reward
   manager.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import CompletionCallback
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    TokenizationSanityCheckModeEnum,
)

from bidding_train_env.llm.prompt import build_user_message, parse_alpha
from bidding_train_env.online.definitions import EPISODE_LENGTH, RL_DATA_DIR, load_act_keys, load_obs_keys
from bidding_train_env.online.online_env import EnvironmentFactory

_META_RE = re.compile(r"<auction_meta>(.*?)</auction_meta>", re.DOTALL)
_META_TAG = "auction_meta"
_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


@dataclass
class _EpisodeState:
    env: Any
    meta: dict[str, Any]
    malformed_count: int = 0
    total_reward: float = 0.0
    final_info: dict[str, Any] | None = None


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _left_pad_1d(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, -t.size(0) :] = t
    return out


def _right_pad_1d(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensors)
    out = torch.full((len(tensors), max_len), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


class AuctionBiddingCompletionCallback(CompletionCallback):
    """Drive the AuctionNet replay env via async VLLM chat completions."""

    def __init__(self, config, scheduler):
        super().__init__(config, scheduler)
        self.prompt_length = int(config.data.max_prompt_length)
        self.response_length = int(config.data.max_response_length)
        self.max_model_len = int(config.actor_rollout_ref.rollout.max_model_len)
        self.obs_keys = load_obs_keys("obs_16_keys")
        self.act_keys = load_act_keys("act_1_key")
        orig_apply_chat_template = self.tokenizer.apply_chat_template

        def _apply_chat_template_no_think(*args, **kwargs):
            try:
                return orig_apply_chat_template(*args, enable_thinking=False, **kwargs)
            except TypeError:
                return orig_apply_chat_template(*args, **kwargs)

        # Keep the actual tokenizer instance type so VeRL's pydantic models accept it.
        self.tokenizer.apply_chat_template = _apply_chat_template_no_think
        self._episodes: dict[int, _EpisodeState] = {}

    @property
    def extra_body(self) -> dict[str, Any]:
        # Per-turn generations should stay tiny and must keep Qwen3.5 thinking off.
        return {
            "chat_template_kwargs": dict(_CHAT_TEMPLATE_KWARGS),
            "max_tokens": 32,
        }

    async def __call__(self, messages: list[dict[str, Any]], completions: ChatCompletion, info: dict[str, Any]):
        conv_id = id(messages)
        episode = self._episodes.get(conv_id)
        if episode is None:
            episode = self._initialize_episode(messages)
            self._episodes[conv_id] = episode

        assistant_msg = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        content = assistant_msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        messages.append({"role": "assistant", "content": content})

        alpha, is_malformed = parse_alpha(content)
        if is_malformed:
            episode.malformed_count += 1

        _, reward, terminated, truncated, step_info = episode.env.step(np.array([alpha], dtype=np.float32))
        episode.total_reward += float(reward)

        done = bool(terminated or truncated)
        if done:
            episode.final_info = {k: _jsonable_scalar(v) for k, v in dict(step_info).items()}
            return

        user_turn = build_user_message(self._state_dict(episode.env), tick=episode.env.unwrapped.time_step, episode_length=EPISODE_LENGTH)
        messages.append(user_turn)
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    def postprocess(self, batch: DataProto, batch_conversations: list[list[dict[str, Any]]], n: int) -> DataProto:
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        reward_scores, num_turns, messages_meta = [], [], []

        raw_prompts = batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)
        for raw_prompt, conversation in zip(raw_prompts, batch_conversations):
            raw_prompt_list = self._normalize_messages(raw_prompt)
            conv_id = id(conversation)
            episode = self._episodes.pop(conv_id, None)
            if episode is None:
                raise RuntimeError("missing AuctionNet episode state during postprocess")

            req = AsyncRolloutRequest(
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=raw_prompt_list,
                tool_schemas=[],
                tools_kwargs={},
                interaction_kwargs={},
                input_ids=[],
                prompt_ids=[],
                response_ids=[],
                attention_mask=[],
                prompt_attention_mask=[],
                response_attention_mask=[],
                position_ids=[],
                prompt_position_ids=[],
                response_position_ids=[],
                loss_mask=[],
                prompt_loss_mask=[],
                response_loss_mask=[],
                reward_scores={},
                max_prompt_len=self.prompt_length,
                max_response_len=self.response_length,
                max_model_len=self.max_model_len,
                use_inference_chat_template=False,
                tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.OFF,
                processing_class=self.tokenizer,
            )

            for message in conversation[len(raw_prompt_list) :]:
                role = message.get("role")
                content = message.get("content", "")
                if role == "assistant":
                    req.add_assistant_message(self.tokenizer, content)
                elif role == "user":
                    req.add_user_message(self.tokenizer, content)
                else:
                    raise ValueError(f"unsupported conversation role: {role!r}")

            episode_scores = self._episode_scores(episode)
            req.finalize(
                self.tokenizer,
                reward_scores=episode_scores,
                finish_reason_type=FinishReasonTypeEnum.STOP,
            )
            episode.env.close()

            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int))
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int))
            reward_scores.append(episode_scores)
            num_turns.append(sum(1 for m in conversation[len(raw_prompt_list) :] if m.get("role") == "assistant"))
            messages_meta.append({"messages": conversation})

        prompt_ids = _left_pad_1d(prompt_ids, self.tokenizer.pad_token_id)
        response_ids = _right_pad_1d(response_ids, self.tokenizer.pad_token_id)
        prompt_attention_mask = _left_pad_1d(prompt_attention_mask, 0)
        response_attention_mask = _right_pad_1d(response_attention_mask, 0)
        prompt_position_ids = _left_pad_1d(prompt_position_ids, 0)
        response_position_ids = _right_pad_1d(response_position_ids, 0)
        prompt_loss_mask = _left_pad_1d(prompt_loss_mask, 0)
        response_loss_mask = _right_pad_1d(response_loss_mask, 0)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=1)
        loss_mask = torch.cat([prompt_loss_mask, response_loss_mask], dim=1)

        batch_td = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(batch_conversations),
        )
        return DataProto(
            batch=batch_td,
            non_tensor_batch={
                "reward_scores": np.array(reward_scores, dtype=object),
                "messages": np.array(messages_meta, dtype=object),
                "__num_turns__": np.array(num_turns, dtype=np.int32),
            },
        )

    def _initialize_episode(self, messages: list[dict[str, Any]]) -> _EpisodeState:
        meta = self._parse_meta(messages)
        env = EnvironmentFactory.create(
            env_name="BiddingEnv",
            pvalues_df_path=str(RL_DATA_DIR / f"period-{meta['period']}_pvalues.parquet"),
            bids_df_path=str(RL_DATA_DIR / f"period-{meta['period']}_bids.parquet"),
            constraints_df_path=str(RL_DATA_DIR / f"period-{meta['period']}_constraints.parquet"),
            obs_keys=self.obs_keys,
            act_keys=self.act_keys,
            budget_range=None,
            target_cpa_range=None,
            seed=int(meta.get("seed", 0)),
        )
        env.reset()
        env.unwrapped.set_campaign(
            advertiser=int(meta["advertiser"]),
            budget=float(meta["budget"]),
            target_cpa=float(meta["target_cpa"]),
            period=int(meta["period"]),
        )
        return _EpisodeState(env=env, meta=meta)

    def _parse_meta(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            m = _META_RE.search(content)
            if m is not None:
                return json.loads(m.group(1))
        raise ValueError(f"missing <{_META_TAG}>...</{_META_TAG}> block in initial prompt")

    def _state_dict(self, env) -> dict[str, Any]:
        inner = env.unwrapped
        pvalues, sigma = inner._get_pvalues_and_sigma()
        return inner._get_state_dict(pvalues, sigma)

    def _normalize_messages(self, raw_prompt: Any) -> list[dict[str, Any]]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        return [dict(msg) for msg in raw_prompt]

    def _episode_scores(self, episode: _EpisodeState) -> dict[str, float]:
        info = episode.final_info or {}
        episode_score = _to_float(info.get("score", 0.0))
        out = {
            "episode_score": episode_score,
            "total_reward": float(episode.total_reward),
            "malformed_count": float(episode.malformed_count),
            "malformed_rate": float(episode.malformed_count) / float(EPISODE_LENGTH),
        }
        for key in (
            "conversions",
            "cost",
            "cpa",
            "target_cpa",
            "budget",
            "cost_over_budget",
            "target_cpa_over_cpa",
            "score",
        ):
            if key in info:
                out[key] = _to_float(info[key])
        return out
