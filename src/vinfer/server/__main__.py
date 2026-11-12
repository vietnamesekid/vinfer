"""Run the server: ``python -m vinfer.server --model <checkpoint dir> [--port 8000]``.

Batch size 1, one request at a time (``SerialScheduler``). The flags build one ``EngineConfig``.
"""

from __future__ import annotations

import argparse

import torch

from vinfer.config import EngineConfig, StateConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local checkpoint directory")
    p.add_argument("--served-model-name", help="name in the API (default: --model)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seq-len", type=int, default=StateConfig.max_seq_len)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import uvicorn

    from vinfer.engine import build_engine
    from vinfer.scheduler.serial import SerialScheduler
    from vinfer.server.app import create_app

    cfg = EngineConfig(
        model=a.model,
        dtype=a.dtype,
        device=a.device,
        state=StateConfig(num_slots=1, max_seq_len=a.max_seq_len),
    )
    engine = build_engine(cfg)
    scheduler = SerialScheduler(
        engine.model, engine.pool, engine.tokenizer.eos_token_ids, engine.tokenizer.decode
    )
    app = create_app(scheduler, engine.tokenizer, a.served_model_name or a.model)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
