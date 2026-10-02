"""Pad spectrograms to a square canvas without stretching time vs frequency."""

from PIL import Image

TARGET = 1024


def pad(img: Image.Image, target: int = TARGET) -> tuple[Image.Image, int, int]:
    img = img.convert("RGB")
    w, h = img.size
    if w > target or h > target:
        raise ValueError(f"image {w}x{h} exceeds target {target}")
    if w == target and h == target:
        return img, w, h
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas, w, h


def padded_to_yolo(xc_p, yc_p, w_p, h_p, orig_w, orig_h, target: int = TARGET):
    return (xc_p * target / orig_w, yc_p * target / orig_h,
            w_p * target / orig_w, h_p * target / orig_h)


def xyxy_to_xywh(x1, y1, x2, y2, img_w, img_h):
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h,
            (x2 - x1) / img_w, (y2 - y1) / img_h)


def clip_yolo(xc, yc, w, h):
    x0 = max(0.0, xc - w / 2)
    x1 = min(1.0, xc + w / 2)
    y0 = max(0.0, yc - h / 2)
    y1 = min(1.0, yc + h / 2)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
