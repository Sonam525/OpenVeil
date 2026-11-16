# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Quality of a masked video against its source (numbers only).

For every frame the luma planes of source and output are compared:

* frames that are bit-identical (copied GOPs) are counted;
* on a sample of the frames that differ (every ``every``-th frame), the PSNR of the luma
  *outside* the mosaic cells measures the loss caused by re-encoding, and the share of
  mosaic-cell pixels that changed shows that the mosaic is really there.

Needs PyAV (``av``) and numpy.
"""
from __future__ import annotations

import itertools
import math
import os
from typing import Any, Dict, Iterator, Tuple

__all__ = ["quality_report", "cell_mask"]


def cell_mask(boxes, width: int, height: int, block: int):
    """Boolean ``height x width`` mask of the pixels in mosaic cells touched by ``boxes``
    (the same grid as :func:`openveil.apply.mosaic.mosaic_array`)."""
    import numpy as np
    from .mosaic import cells

    g = cells(boxes, width, height, block)
    return np.kron(g, np.ones((block, block), bool))[:height, :width]


def _gray_frames(path: str) -> Iterator[Tuple[int, Any]]:
    import av

    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for f in c.decode(video=0):
            yield f.pts, f.to_ndarray(format="gray")


def quality_report(src: str, dst: str, table: Any, cfg: Any = None, every: int = 50,
                   plan: Any = None) -> Dict[str, Any]:
    """Compare ``dst`` (a masked copy of ``src`` made with ``table``) with ``src``.

    Returns ``frames`` (decoded from both, compared pairwise), ``frames_src`` /
    ``frames_dst``, ``frames_identical``, ``sampled_reencoded`` and, over the sampled frames,
    ``psnr_outside_mosaic_median`` / ``_p5`` (dB; 99 when the luma outside the mosaic is
    unchanged) and ``mosaic_pixels_changed_median`` (share of mosaic-cell pixels whose luma
    changed). Raises ``ValueError`` when a frame pair has different pts.
    """
    import numpy as np
    from ..config import load_config
    from .mosaic import block_size
    from .plan import MaskPlan
    from .video import _load_table, _seconds_fn

    cfg = load_config(cfg)
    table = _load_table(table)
    plan = plan if plan is not None else MaskPlan(table, cfg)
    import av
    with av.open(str(src)) as c:
        tb = c.streams.video[0].time_base
    sec = _seconds_fn(table, tb)
    div = getattr(plan, "block_divisor", None) or cfg.apply.block_divisor
    mn = getattr(plan, "block_min", None) or cfg.apply.block_min
    psnr_re, cells_changed = [], []
    n, n_same, n_src, n_dst = 0, 0, 0, 0
    for a_, b_ in itertools.zip_longest(_gray_frames(src), _gray_frames(dst)):
        n_src += a_ is not None
        n_dst += b_ is not None
        if a_ is None or b_ is None:
            continue
        (ps, a), (pd_, b) = a_, b_
        n += 1
        if ps != pd_:
            raise ValueError(f"pts differ at frame {n}: {ps} vs {pd_}")
        if np.array_equal(a, b):
            n_same += 1
            continue
        if n % every:
            continue
        H, W = a.shape
        B = block_size(H, div, mn)
        m = cell_mask(plan.boxes_at(sec(ps)), W, H, B)
        out = ~m
        d = a.astype(np.float64) - b.astype(np.float64)
        mse = float((d[out] ** 2).mean()) if out.any() else 0.0
        psnr_re.append(99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
        if m.any():
            cells_changed.append(float((np.abs(d[m]) > 0).mean()))
    return dict(video=os.path.basename(str(src)), frames=n, frames_src=n_src, frames_dst=n_dst,
                frames_identical=n_same, sampled_reencoded=len(psnr_re),
                psnr_outside_mosaic_median=round(float(np.median(psnr_re)), 2) if psnr_re else None,
                psnr_outside_mosaic_p5=round(float(np.percentile(psnr_re, 5)), 2) if psnr_re else None,
                mosaic_pixels_changed_median=round(float(np.median(cells_changed)), 3) if cells_changed else None)
