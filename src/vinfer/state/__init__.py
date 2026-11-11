"""State manager: allocates, pages, snapshots and frees per-request state.

Handles both KV cache (per token; attention and MLA latent layouts) and fixed-size
recurrent state (Gated DeltaNet). Recurrent state can only be reused at positions
where a snapshot exists (docs/architecture.md); when to take snapshots is an open
design question.

Must not: know about requests or SLOs.
"""
