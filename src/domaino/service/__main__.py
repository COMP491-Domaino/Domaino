"""Run the inference service.

    python -m domaino.service                        # mocked predictions
    python -m domaino.service --checkpoint path.pt   # a trained T3.2 model

With no checkpoint the service serves clearly labelled simulated results,
which is what makes the T4.1 interface demonstrable before WP3 finishes.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence

from .app import Service, serve
from .jobs import JobStore
from .predict import build_predictor


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--checkpoint", default=None,
        help="a T3.2 checkpoint; omit to serve mocked predictions",
    )
    parser.add_argument(
        "--encoder-config", default=None,
        help="configs/representation_v1.json, for the encoder's cache path",
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--static", default=None,
        help="a directory to serve the interface from, alongside the API",
    )
    parser.add_argument(
        "--job-retention-minutes", type=float, default=60.0,
        help="how long a finished result is kept before it is forgotten",
    )
    args = parser.parse_args(argv)

    encoder_config = None
    if args.encoder_config:
        from ..model.embed import load_config
        from ..model.representation import EncoderConfig

        block = dict(load_config(args.encoder_config)["encoder"])
        encoder_config = EncoderConfig(
            **{
                key: value
                for key, value in block.items()
                if key in EncoderConfig.__dataclass_fields__
            }
        )

    predictor = build_predictor(
        args.checkpoint, encoder_config, args.device, log=log
    )
    service = Service(
        predictor=predictor,
        store=JobStore(retention_seconds=args.job_retention_minutes * 60.0),
        log=log,
    )
    serve(args.host, args.port, service, static_root=args.static)
    return 0


if __name__ == "__main__":
    sys.exit(main())
