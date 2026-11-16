# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Fake proposer / verifier and synthetic videos for the detection tests (no models, no people).

The synthetic scene is a dark background with two flat rectangles:

* a bright "walker" rectangle (value 235) that is visible only in a time window and moves to
  the right: the fake proposer boxes it and the fake verifier accepts it;
* a mid-grey "decoy" rectangle (value 150) that is always visible: the fake proposer boxes it
  too, but the fake verifier rejects it (like a calf that the proposer mistakes for a person).

A frame whose top-left pixel is very dark (background 10 instead of 60) makes the fake proposer
hang, to test the stall watchdog. The factory functions are module-level so that worker
processes started with ``spawn`` can unpickle them. This module holds no tests itself.
"""
from __future__ import annotations

import os
import time
from fractions import Fraction

import numpy as np

try:
    import av  # noqa: F401
    HAVE_AV = True
except ImportError:  # pragma: no cover
    HAVE_AV = False

W, H, FPS = 160, 120, 10
WALK_T0, WALK_T1 = 3.0, 7.0                 # seconds in which the walker is visible
WALKER = 235
DECOY = 150
BACKGROUND = 60
HANG_BACKGROUND = 10


def walker_box(t: float) -> list[float]:
    """True walker box at time t (pixels, x2/y2 exclusive)."""
    x = 10 + int(round(8 * (t - WALK_T0)))
    return [float(x), 30.0, float(x + 20), 90.0]


DECOY_BOX = [110.0, 70.0, 150.0, 110.0]


def synth_frame(i: int, fps: int = FPS, background: int = BACKGROUND, walker: bool = True) -> np.ndarray:
    """Grey frame ``i`` (H x W x 3 uint8) with the decoy and, inside its window, the walker."""
    img = np.full((H, W, 3), background, np.uint8)
    x1, y1, x2, y2 = (int(v) for v in DECOY_BOX)
    img[y1:y2, x1:x2] = DECOY
    t = i / fps
    if walker and WALK_T0 <= t < WALK_T1:
        x1, y1, x2, y2 = (int(v) for v in walker_box(t))
        img[y1:y2, x1:x2] = WALKER
    return img


def make_video(path: str, seconds: float = 12.0, fps: int = FPS, background: int = BACKGROUND, walker: bool = True) -> str:
    """Write a near-lossless H.264 (or MPEG-4) MP4 of the synthetic scene."""
    import av

    n = int(round(seconds * fps))
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(path, "w", format="mp4") as oc:
        st = oc.add_stream(codec, rate=fps)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.codec_context.gop_size = fps
        st.codec_context.time_base = Fraction(1, fps)
        if codec == "libx264":
            st.options = {"crf": "4", "bf": "0", "preset": "ultrafast"}
        else:
            st.options = {"qscale": "1"}
        for i in range(n):
            fr = av.VideoFrame.from_ndarray(synth_frame(i, fps, background, walker)[:, :, ::-1].copy(), format="bgr24")
            fr.pts = i
            fr.time_base = Fraction(1, fps)
            for pkt in st.encode(fr):
                oc.mux(pkt)
        for pkt in st.encode():
            oc.mux(pkt)
    return path


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class FakeProposer:
    """Boxes bright (walker, score 0.9) and mid-grey (decoy, score 0.3) regions; records calls."""

    def __init__(self, hang_s: float = 3600.0):
        self.calls: list[int] = []
        self.hang_s = hang_s

    def __call__(self, frames):
        self.calls.append(len(frames))
        out = []
        for f in frames:
            g = f[:, :, 1].astype(np.int16)
            if int(g[0, 0]) < 30:
                time.sleep(self.hang_s)          # a hanging model, for the watchdog test
            boxes = []
            b = _bbox(g > 200)
            if b:
                boxes.append(b + [0.9])
                boxes.append([b[0] + 1, b[1], b[2] + 1, b[3], 0.5])   # a near duplicate: merged by the candidate rule
            d = _bbox((g > 120) & (g < 180))
            if d:
                boxes.append(d + [0.3])
            boxes.append([0.0, 0.0, 5.0, 5.0, 0.01])                  # below cand_thr: never a candidate
            out.append(boxes)
        return out


class FakeVerifier:
    """Accepts boxes whose centre pixel is bright (the walker), rejects the rest."""

    mode = "fake"

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        self.n_boxes = 0

    def score_frames(self, frames, boxes_per_frame):
        out = []
        for f, boxes in zip(frames, boxes_per_frame):
            sc = []
            for b in boxes:
                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                cy, cx = min(max(cy, 0), f.shape[0] - 1), min(max(cx, 0), f.shape[1] - 1)
                sc.append(0.95 if f[cy, cx, 1] > 200 else 0.02)
            self.n_boxes += len(boxes)
            out.append(np.array(sc, np.float32))
        return out


def fake_factory(cfg):
    """Picklable factory for :func:`openveil.detect.run.run_many`."""
    return FakeProposer(), FakeVerifier(cfg.verifier.threshold)


def counting_factory(cfg):
    """Like :func:`fake_factory`, but appends one line per model build to ``$OV_TEST_COUNT_FILE``."""
    p = os.environ.get("OV_TEST_COUNT_FILE")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write("build\n")
    return fake_factory(cfg)


def broken_factory(cfg):
    """A factory whose models cannot be built (as with a GPU out of memory): no video is to blame."""
    raise RuntimeError("the model cannot be loaded")
