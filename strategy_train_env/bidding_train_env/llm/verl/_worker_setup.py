"""Ray worker setup hook.

Every Ray worker process runs `setup()` before user tasks. We use it to
trigger the `@register("bidding_agent")` decorator inside our AgentLoop
module so it lands in the worker's process-local agent-loop registry.
"""
from __future__ import annotations


def setup() -> None:
    import bidding_train_env.llm.verl.bidding_agent_loop  # noqa: F401
