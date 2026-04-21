"""Backend-agnostic LLM bidding agent.

`BiddingAgent` holds a reference to an `LLMBackend` (not a specific vendor
class). It maintains per-env chat histories across a 48-tick episode, batches
each tick's user turns into a single `backend.chat(...)` call, parses α from
each response, and returns the per-turn records the eval loop feeds into
`env.step()`.
"""

from __future__ import annotations

from typing import Any

from bidding_train_env.llm.backends import LLMBackend, SamplingSpec
from bidding_train_env.llm.prompt import SYSTEM_PROMPT, build_user_message, parse_alpha


class BiddingAgent:
    def __init__(
        self,
        backend: LLMBackend,
        sampling: SamplingSpec,
        system_prompt: str = SYSTEM_PROMPT,
        episode_length: int = 48,
    ):
        self.backend = backend
        self.sampling = sampling
        self.system_prompt = system_prompt
        self.episode_length = int(episode_length)
        self._histories: list[list[dict]] = []

    def reset(self, n_envs: int) -> None:
        """Start fresh chat histories for `n_envs` parallel episodes."""
        self._histories = [
            [{"role": "system", "content": self.system_prompt}] for _ in range(int(n_envs))
        ]

    @property
    def n_envs(self) -> int:
        return len(self._histories)

    def act(
        self,
        state_dicts: list[dict],
        ticks: list[int],
        env_indices: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """One multi-env tick: build user turns, batch-generate responses, parse α.

        Args:
            state_dicts: one obs dict per still-alive env (order matches env_indices).
            ticks: current tick index per env (same order).
            env_indices: indices into `self._histories` for the alive envs. If None,
                defaults to `list(range(len(state_dicts)))` and assumes all envs are alive.

        Returns one record per input in the same order with keys:
            messages:      full conversation (incl. this tick's user turn, excl. assistant)
            response:      raw assistant text
            alpha:         parsed float in [-10, 10] (fallback 0.0 on parse failure)
            is_malformed:  True if the response did not match `<alpha>X</alpha>`
        """
        if env_indices is None:
            env_indices = list(range(len(state_dicts)))
        if not (len(state_dicts) == len(ticks) == len(env_indices)):
            raise ValueError(
                f"mismatched lengths: state_dicts={len(state_dicts)}, "
                f"ticks={len(ticks)}, env_indices={len(env_indices)}"
            )

        # Append the user turn to each selected history.
        user_turns = [
            build_user_message(s, int(t), self.episode_length)
            for s, t in zip(state_dicts, ticks)
        ]
        for idx, u in zip(env_indices, user_turns):
            self._histories[idx].append(u)

        batch = [self._histories[idx] for idx in env_indices]
        responses = self.backend.chat(batch, self.sampling)
        if len(responses) != len(env_indices):
            raise RuntimeError(
                f"backend returned {len(responses)} responses for {len(env_indices)} conversations"
            )

        out = []
        for idx, resp in zip(env_indices, responses):
            alpha, malformed = parse_alpha(resp)
            record = {
                # Snapshot the history up to & including this user turn (assistant turn
                # appended below). Shallow copy of the list; dict items are reused.
                "messages": list(self._histories[idx]),
                "response": resp,
                "alpha": float(alpha),
                "is_malformed": bool(malformed),
            }
            out.append(record)
            self._histories[idx].append({"role": "assistant", "content": resp})
        return out
