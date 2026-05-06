"""LLM-driven auto-bidding agent for the AuctionNet online replay env.

- `prompt`: system prompt + user-turn rendering + response parser.
- `backends`: pluggable LLM backends (vLLM offline, OpenAI-compatible HTTP).
- `agent`: `BiddingAgent` — backend-agnostic multi-env chat loop emitting α per tick.
- `main_eval_llm`: CLI entry for period-27 sweep evaluation.
"""
