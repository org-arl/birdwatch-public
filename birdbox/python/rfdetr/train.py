#!/usr/bin/env python3
"""Fine-tune RF-DETR Nano or Large, from Roboflow pretrain or a .pth checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rfdetr import RFDETRLarge, RFDETRNano, from_checkpoint

from pad import TARGET

ARCH = {"nano": RFDETRNano, "large": RFDETRLarge}


def build_model(model: str, device: str | None):
    kwargs = dict(resolution=TARGET)
    if device:
        kwargs["device"] = device
    if model in ARCH:
        return ARCH[model](**kwargs)
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    kwargs["trust_checkpoint"] = True
    return from_checkpoint(str(path), **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="nano, large, or a path to a .pth checkpoint")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--skip-best-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not (args.dataset_dir / "data.yaml").is_file():
        print(f"[ERR] missing {args.dataset_dir / 'data.yaml'}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.model, args.device)
    kwargs = dict(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        early_stopping=True,
        early_stopping_patience=args.patience,
        skip_best_epochs=args.skip_best_epochs,
        batch_size=args.batch,
        grad_accum_steps=1,
        resolution=TARGET,
        checkpoint_interval=max(args.epochs + 1, 1000),
    )
    if args.device:
        kwargs["device"] = args.device
    print(f"[TRAIN] model={args.model}  epochs={args.epochs}  batch={args.batch}")
    print(f"[DATA]  {args.dataset_dir}")
    print(f"[OUT]   {args.output_dir}")
    model.train(**kwargs)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
