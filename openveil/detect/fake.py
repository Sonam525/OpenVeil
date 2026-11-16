# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""A model-free stand-in for the detector, to try the whole pipeline without any download.

``openveil detect VIDEO --out tables/ --factory openveil.detect.fake:factory`` runs detection
with :class:`BrightBlobProposer` (it boxes the largest region that is much brighter than the
rest of the frame) and the pass-through verifier. It finds no people: it only exercises detect,
review and apply on synthetic or test footage, for example a clip with a bright rectangle
moving over a dark background.

A factory is any module-level function ``factory(cfg) -> (proposer, verifier)``:

* ``proposer(frames_bgr) -> [[[x1, y1, x2, y2, score], ...] per frame]`` (frames are H x W x 3
  uint8 BGR arrays, boxes in pixels of the frame);
* ``verifier`` has ``threshold`` (float), ``mode`` (str) and ``score_frames(frames_bgr,
  boxes_per_frame) -> [array of scores per frame]``; a box is kept when its score is >=
  ``threshold``.

The factory runs inside each worker process, so it must be importable by name.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

__all__ = ["BrightBlobProposer", "factory"]


class BrightBlobProposer:
    """Boxes the pixels that are at least ``margin`` grey levels brighter than the frame median.

    One box per frame (the bounding box of those pixels) when at least ``min_pixels`` of them
    exist, with score 0.9. Only numpy is used.
    """

    def __init__(self, margin: int = 80, min_pixels: int = 50, score: float = 0.9):
        self.margin, self.min_pixels, self.score = int(margin), int(min_pixels), float(score)

    def __call__(self, frames_bgr: Sequence[np.ndarray]) -> List[List[List[float]]]:
        out = []
        for f in frames_bgr:
            g = np.asarray(f, dtype=np.uint8).mean(axis=2) if np.ndim(f) == 3 else np.asarray(f, dtype=np.float32)
            mask = g >= float(np.median(g)) + self.margin
            ys, xs = np.nonzero(mask)
            if len(xs) < self.min_pixels:
                out.append([])
                continue
            out.append([[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), self.score]])
        return out


def factory(cfg: Any):
    """``(BrightBlobProposer(), PassThroughVerifier(threshold 0))``: no model, no download."""
    from .verifier import PassThroughVerifier  # noqa: PLC0415

    return BrightBlobProposer(), PassThroughVerifier(0.0)
