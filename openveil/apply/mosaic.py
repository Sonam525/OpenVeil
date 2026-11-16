# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Mosaic (pixelation) over the union of boxes, on one fixed grid per image.

The grid is anchored at the image origin with square cells of ``block`` pixels. Every cell that
touches a box is filled with the mean of its own pixels (edge cells that are cut by the image
border use only the pixels inside the image). Because the grid does not depend on the boxes,
overlapping boxes give one consistent mosaic and the result does not depend on the order of the
boxes; a cell is never partly mosaicked.

Videos are mosaicked on their decoded I420 (yuv420p) planes: the luma plane with cells of
``block`` px, the two half-size chroma planes with cells of ``block // 2`` px and the boxes
halved, so luma and chroma cells cover the same image area.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..geometry import scale_boxes

__all__ = ["block_size", "cells", "mosaic_array", "mosaic_yuv420"]

Boxes = Sequence[Sequence[float]]


def block_size(height: int, divisor: float = 36.0, minimum: int = 16) -> int:
    """Mosaic cell size in pixels for an image ``height`` px high: ``max(minimum,
    round(height / divisor))``, i.e. 20 px at 720p, 30 px at 1080p, 36 px at 1296p."""
    return max(int(minimum), int(round(height / float(divisor))))


def cells(boxes: Boxes, width: int, height: int, block: int) -> np.ndarray:
    """Boolean grid ``(rows, cols)`` of the ``block``-px cells that touch any box (boxes are
    clipped to the image; a box of zero area inside the image touches nothing)."""
    rows, cols = (height + block - 1) // block, (width + block - 1) // block
    g = np.zeros((rows, cols), bool)
    for b in boxes:
        x1, y1 = max(0.0, b[0]), max(0.0, b[1])
        x2, y2 = min(float(width), b[2]), min(float(height), b[3])
        if x2 <= x1 or y2 <= y1:
            continue
        g[int(y1 // block):int(np.ceil(y2 / block)), int(x1 // block):int(np.ceil(x2 / block))] = True
    return g


def mosaic_array(img: np.ndarray, boxes: Boxes, block: Optional[int] = None) -> np.ndarray:
    """Mosaic ``img`` (``H x W`` or ``H x W x C``, any integer or float dtype) in place over the
    union of ``boxes`` (pixels of this image) and return it.

    ``block`` defaults to :func:`block_size` of the image height. Cell means are rounded to the
    nearest value for integer images. Vectorised: the means come from a block reshape of the
    bounding region of the touched cells and are written back only where a cell touches a box.
    """
    H, W = img.shape[:2]
    B = int(block or block_size(H))
    if B < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    if not boxes:
        return img
    g = cells(boxes, W, H, B)
    if not g.any():
        return img
    r0, r1 = np.nonzero(g.any(1))[0][[0, -1]]
    c0, c1 = np.nonzero(g.any(0))[0][[0, -1]]
    y0, y1, x0, x1 = r0 * B, min(H, (r1 + 1) * B), c0 * B, min(W, (c1 + 1) * B)
    sub = img[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    ph, pw = (-h) % B, (-w) % B
    tail = img.shape[2:]
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2)
    f = np.pad(sub.astype(np.float32), pad, mode="edge")
    nr, nc = f.shape[0] // B, f.shape[1] // B
    blk = f.reshape(nr, B, nc, B, *tail)
    if ph or pw:                                   # mean over the real pixels only (edge cells)
        wts = np.pad(np.ones((h, w), np.float32), ((0, ph), (0, pw))).reshape(nr, B, nc, B)
        wts_ = wts.reshape(nr, B, nc, B, *([1] * (img.ndim - 2)))
        mean = (blk * wts_).sum(axis=(1, 3)) / wts.sum(axis=(1, 3)).reshape(nr, nc, *([1] * (img.ndim - 2)))
    else:
        mean = blk.mean(axis=(1, 3))
    m = g[r0:r1 + 1, c0:c1 + 1]
    full = np.repeat(np.repeat(mean, B, axis=0), B, axis=1)[:h, :w]
    mm = np.repeat(np.repeat(m, B, axis=0), B, axis=1)[:h, :w]
    if img.ndim == 3:
        mm = mm[..., None]
    vals = np.round(full).astype(img.dtype) if np.issubdtype(img.dtype, np.integer) else full.astype(img.dtype)
    sub[...] = np.where(mm, vals, sub)
    return img


Planes = Union[np.ndarray, Sequence[np.ndarray]]


def _split_i420(a: np.ndarray, width: Optional[int], height: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Views of the Y, U and V planes of a packed I420 buffer of shape ``(H * 3 // 2, W)`` (the
    layout of ``av.VideoFrame.to_ndarray(format="yuv420p")``)."""
    if a.ndim != 2:
        raise ValueError("a packed I420 buffer must be 2-D (H * 3 / 2 rows, W columns)")
    if not a.flags.c_contiguous:
        raise ValueError("the packed I420 buffer must be C-contiguous (it is mosaicked in place)")
    W = width or a.shape[1]
    H = height or (a.shape[0] * 2) // 3
    cw, ch = (W + 1) // 2, (H + 1) // 2
    flat = a.reshape(-1)
    ny, nc = H * W, cw * ch
    if flat.size < ny + 2 * nc or a.shape[1] != W:
        raise ValueError(f"buffer of shape {a.shape} is not a {W} x {H} I420 frame")
    y = flat[:ny].reshape(H, W)
    u = flat[ny:ny + nc].reshape(ch, cw)
    v = flat[ny + nc:ny + 2 * nc].reshape(ch, cw)
    return y, u, v


def mosaic_yuv420(planes: Planes, boxes: Boxes, block: Optional[int] = None,
                  width: Optional[int] = None, height: Optional[int] = None) -> Planes:
    """Mosaic an I420 frame in place over the union of ``boxes`` (luma pixel coordinates).

    ``planes`` is either ``(y, u, v)`` (three 2-D arrays, chroma at half size) or one packed
    C-contiguous buffer of shape ``(H * 3 // 2, W)`` as returned by PyAV for ``yuv420p``
    (``width``/``height`` are needed only when they cannot be inferred from that shape).
    Luma cells are ``block`` px (default :func:`block_size` of the luma height); chroma cells are
    ``max(1, block // 2)`` px with the boxes halved. Returns ``planes``.
    """
    if isinstance(planes, np.ndarray):
        y, u, v = _split_i420(planes, width, height)
    else:
        y, u, v = planes
    B = int(block or block_size(y.shape[0]))
    mosaic_array(y, boxes, B)
    half = scale_boxes(boxes, 0.5, 0.5)
    mosaic_array(u, half, max(1, B // 2))
    mosaic_array(v, half, max(1, B // 2))
    return planes
