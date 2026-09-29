#!/usr/bin/env python3
"""Run RF-DETR on spectrograms and write YOLO-format labels in original image coords."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from rfdetr import from_checkpoint

from pad import TARGET, clip_yolo, pad, padded_to_yolo, xyxy_to_xywh


def load_model(checkpoint: Path, device: str | None):
    kwargs = dict(trust_checkpoint=True, resolution=TARGET)
    if device:
        kwargs["device"] = device
    return from_checkpoint(str(checkpoint), **kwargs)


def predict_image(model, path: Path, conf: float) -> str:
    img = Image.open(path)
    padded, orig_w, orig_h = pad(img)
    detections = model.predict(padded, threshold=conf)
    xyxy = detections.xyxy
    scores = detections.confidence
    if xyxy is None or len(xyxy) == 0:
        return ""
    lines = []
    for box, score in zip(xyxy, scores):
        xc_p, yc_p, w_p, h_p = xyxy_to_xywh(
            float(box[0]), float(box[1]), float(box[2]), float(box[3]),
            TARGET, TARGET,
        )
        xc, yc, w, h = padded_to_yolo(xc_p, yc_p, w_p, h_p, orig_w, orig_h)
        mapped = clip_yolo(xc, yc, w, h)
        if mapped is None:
            continue
        xc, yc, w, h = mapped
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {float(score):.6f}")
    return "\n".join(lines) + "\n" if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    model = load_model(args.checkpoint, args.device)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(args.images.glob("*.png")):
        (out_dir / f"{path.stem}.txt").write_text(predict_image(model, path, args.confidence))
        n += 1
    print(f"wrote predictions for {n} images -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
