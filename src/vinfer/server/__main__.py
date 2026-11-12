"""Run the server: ``python -m vinfer.server --model <checkpoint dir> [--port 8000]``.

Batch size 1, one request at a time (``SerialScheduler``). The flags build one ``EngineConfig``.
"""

from __future__ import annotations

import argparse

import torch

from vinfer.config import EngineConfig, KernelConfig, ServingConfig, StateConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local checkpoint directory")
    p.add_argument("--served-model-name", help="name in the API (default: --model)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seq-len", type=int, default=StateConfig.max_seq_len)
    p.add_argument("--kernels", help="e.g. attention_backend=reference,gdn_backend=reference")
    p.add_argument("--seed", type=int, default=0, help="seeds requests that do not give one")
    p.add_argument(
        "--reasoning-parser",
        default=ServingConfig.reasoning_parser,
        help='splits reasoning_content from content: "auto" (from the model type), "none", '
        "or a name (vinfer.server.parsers.REASONING_PARSERS)",
    )
    p.add_argument(
        "--tool-call-parser",
        default=ServingConfig.tool_call_parser,
        help='extracts tool_calls: "auto" (from the model type), "none", or a name '
        "(vinfer.server.parsers.TOOL_CALL_PARSERS)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import uvicorn

    from vinfer.engine import build_engine
    from vinfer.models import read_model_type
    from vinfer.scheduler.serial import SerialScheduler
    from vinfer.server.app import create_app
    from vinfer.server.parsers import OutputParsers, resolve_parser_names

    cfg = EngineConfig(
        model=a.model,
        dtype=a.dtype,
        device=a.device,
        seed=a.seed,
        kernels=KernelConfig.from_string(a.kernels),
        state=StateConfig(num_slots=1, max_seq_len=a.max_seq_len),
        serving=ServingConfig(a.reasoning_parser, a.tool_call_parser),
    )
    # Checked before the weights are loaded: a typo fails fast.
    parsers = OutputParsers(
        *resolve_parser_names(
            cfg.serving.reasoning_parser, cfg.serving.tool_call_parser, read_model_type(cfg.model)
        )
    )
    engine = build_engine(cfg)
    scheduler = SerialScheduler(
        engine.model,
        engine.pool,
        engine.tokenizer.eos_token_ids,
        engine.tokenizer.decode,
        seed=cfg.seed,
    )
    app = create_app(scheduler, engine.tokenizer, a.served_model_name or a.model, parsers)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
