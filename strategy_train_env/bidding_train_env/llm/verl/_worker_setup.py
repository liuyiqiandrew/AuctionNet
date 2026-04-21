"""Ray worker setup hook.

Every Ray worker process (including the TaskRunner actor that calls
`load_reward_manager`) executes `setup()` before any user task runs. We use
it to patch transformers + transformers↔verl API drift and to trigger the
`@register("auction_env")` decorator inside our reward manager module.
"""
from __future__ import annotations


def _alias_vision2seq() -> None:
    # transformers>=5 dropped AutoModelForVision2Seq (replaced by
    # AutoModelForImageTextToText). verl 0.4.1 still imports Vision2Seq in
    # fsdp_workers.py / model.py / base_model_merger.py / fsdp_checkpoint_manager.py.
    # Our text-only Qwen3.5-9B never instantiates it — verl only checks
    # `_model_mapping.keys()` — so aliasing to ImageTextToText is safe.
    import transformers

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        return
    transformers.AutoModelForVision2Seq = AutoModelForImageTextToText


def setup() -> None:
    _alias_vision2seq()
    import bidding_train_env.llm.verl.auction_reward_manager  # noqa: F401
