"""LLM backend abstraction.

`LLMBackend.chat(conversations, sampling) -> list[str]` is the one interface
the agent depends on. Vendor imports (`vllm`, `openai`) happen lazily inside
the concrete backend `__init__` / methods so the agent module can be imported
without every LLM dependency installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SamplingSpec:
    temperature: float = 0.0
    max_tokens: int = 64
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] | None = None
    # Chat-template kwargs forwarded to the tokenizer's chat template. Used to
    # disable reasoning on Qwen3/3.5 and similar ({"enable_thinking": False}).
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


Conversation = list[dict]


@runtime_checkable
class LLMBackend(Protocol):
    def chat(self, conversations: list[Conversation], sampling: SamplingSpec) -> list[str]:
        ...


class VLLMOfflineBackend:
    """In-process vLLM engine. One `engine.chat` call fans out over all conversations."""

    def __init__(self, model: str, **engine_kwargs):
        from vllm import LLM  # lazy

        self.model = model
        self.engine = LLM(model=model, **engine_kwargs)

    def chat(self, conversations: list[Conversation], sampling: SamplingSpec) -> list[str]:
        from vllm import SamplingParams  # lazy

        sp_kwargs = dict(
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            top_p=sampling.top_p,
        )
        if sampling.seed is not None:
            sp_kwargs["seed"] = int(sampling.seed)
        if sampling.stop:
            sp_kwargs["stop"] = list(sampling.stop)
        sp_kwargs.update(sampling.extra)
        sp = SamplingParams(**sp_kwargs)

        chat_kwargs: dict[str, Any] = {"sampling_params": sp, "use_tqdm": False}
        if sampling.chat_template_kwargs:
            chat_kwargs["chat_template_kwargs"] = dict(sampling.chat_template_kwargs)
        outputs = self.engine.chat(conversations, **chat_kwargs)
        return [o.outputs[0].text for o in outputs]


class VLLMServerBackend:
    """OpenAI-compatible /v1/chat/completions client (primary target: `vllm serve`).

    Also works against any other OpenAI-compatible endpoint (sglang, hosted
    OpenAI, etc.). Fans out over the batch with a thread pool since the sync
    `openai` client blocks per request.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_workers: int = 32,
        timeout: float = 600.0,
    ):
        from openai import OpenAI  # lazy

        self.model = model
        self.base_url = base_url
        self.max_workers = int(max_workers)
        self.timeout = float(timeout)
        # Also disable openai's builtin retries — on a timeout we'd rather surface
        # the error immediately than silently wait another `timeout` seconds.
        self.client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0,
        )

    def _one(self, conv: Conversation, sampling: SamplingSpec) -> str:
        kwargs = dict(
            model=self.model,
            messages=conv,
            temperature=sampling.temperature,
            max_tokens=sampling.max_tokens,
            top_p=sampling.top_p,
        )
        if sampling.seed is not None:
            kwargs["seed"] = int(sampling.seed)
        if sampling.stop:
            kwargs["stop"] = list(sampling.stop)
        # vLLM's OpenAI server accepts non-standard fields via `extra_body`.
        # `chat_template_kwargs` is how you disable Qwen3/3.5 thinking mode.
        extra_body: dict[str, Any] = {}
        if sampling.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(sampling.chat_template_kwargs)
        if extra_body:
            kwargs["extra_body"] = extra_body
        kwargs.update(sampling.extra)
        r = self.client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def chat(self, conversations: list[Conversation], sampling: SamplingSpec) -> list[str]:
        if not conversations:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda c: self._one(c, sampling), conversations))


def build_backend(name: str, model: str, **kwargs) -> LLMBackend:
    if name == "vllm_offline":
        return VLLMOfflineBackend(model=model, **kwargs)
    if name == "vllm_server":
        return VLLMServerBackend(model=model, **kwargs)
    raise ValueError(f"unknown LLM backend: {name!r}. options: vllm_offline, vllm_server")
