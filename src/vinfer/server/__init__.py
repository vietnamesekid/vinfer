"""Thin OpenAI-compatible HTTP API with streaming.

Generated text becomes ``content`` / ``reasoning_content`` / ``tool_calls`` through the output
parsers (``parsers/``), chosen per model family by name; the server code itself never branches
on the model type, and the scheduler and runtime never see the parsers.

Must not: contain batching logic (that belongs to the scheduler).
"""
