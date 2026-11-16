# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""What to mosaic, frame by frame, from a box table.

    plan = MaskPlan(table)          # BoxTable, its dict, or a path; production settings
    plan.boxes_at(t)                # mosaic boxes for the frame at t seconds (table clock)
    plan.intervals()                # [(t0, t1), ...] seconds in which some mosaic is drawn
    block_size(height)              # mosaic cell size

Rules, in this order (values of the 'production' preset):

1. the giant rule (:func:`openveil.rules.giant_rule`) on the raw boxes of every table frame:
   a box covering >= 55% of the image that holds >= 2 smaller boxes is dropped, the smaller
   boxes stay;
2. every remaining box is enlarged by 20% of its width / height on each side, clipped to the
   image;
3. temporal hold: the frame at time t gets every enlarged box of every table frame within
   t - 2 s .. t + 2 s (by the table's pts, so variable frame rates are handled).

The mosaic itself (:mod:`openveil.apply.mosaic`) is drawn on one grid anchored at the image
origin over the union of the boxes.
"""
from __future__ import annotations

import bisect
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from ..geometry import enlarge
from ..rules import apply_rules_to_table
from ..table import BoxTable, load_table, seconds
from .mosaic import block_size as _block_size

__all__ = ["MaskPlan", "block_size"]


def block_size(height: int, divisor: float = 36.0, minimum: int = 16) -> int:
    """Mosaic cell size in pixels: ``max(minimum, round(height / divisor))`` (20 px at 720p,
    30 px at 1080p, 36 px at 1296p)."""
    return _block_size(height, divisor, minimum)


def _apply_cfg(cfg: Any) -> Any:
    """The ApplyConfig inside ``cfg`` (an OpenVeilConfig, an ApplyConfig or None)."""
    if cfg is None:
        from ..config import ApplyConfig
        return ApplyConfig()
    return getattr(cfg, "apply", cfg)


class MaskPlan:
    """The mosaic boxes of every frame of one video (or image sequence).

    ``table``: a :class:`~openveil.table.BoxTable`, its dict, or a path to a table file.
    ``cfg``: an :class:`~openveil.config.OpenVeilConfig` or :class:`~openveil.config.ApplyConfig`
    (default: the production preset). The keyword arguments override single values of ``cfg``.

    Attributes: ``width``, ``height``, ``hold`` (seconds), ``rule_stats`` (giant-rule statistics
    of :func:`~openveil.rules.apply_rules_to_table`), ``times`` (sorted seconds of the table rows
    that keep a box) and ``boxes`` (their enlarged boxes).
    """

    def __init__(self, table: Union[BoxTable, Dict[str, Any], str, os.PathLike], cfg: Any = None, *,
                 min_share: Optional[float] = None, min_inner: Optional[int] = None,
                 inner_frac: Optional[float] = None, pad: Optional[float] = None, hold_s: Optional[float] = None,
                 keep_kinds: Optional[List[str]] = None, block_divisor: Optional[float] = None,
                 block_min: Optional[int] = None):
        a = _apply_cfg(cfg)
        if isinstance(table, (str, os.PathLike)):
            table = load_table(table)
        pick = lambda v, name: getattr(a, name) if v is None else v          # noqa: E731
        self.min_share = pick(min_share, "min_share")
        self.min_inner = pick(min_inner, "min_inner")
        self.inner_frac = pick(inner_frac, "inner_frac")
        self.pad = pick(pad, "pad")
        self.keep_kinds = list(pick(keep_kinds, "keep_kinds") or [])
        self.block_divisor = pick(block_divisor, "block_divisor")
        self.block_min = pick(block_min, "block_min")
        self.hold = float(pick(hold_s, "hold_s"))
        d = table.to_dict(include_source=True) if isinstance(table, BoxTable) else table
        self.video = d.get("video")
        self.width, self.height = int(d["width"]), int(d["height"])
        t2, self.rule_stats = apply_rules_to_table(d, self.min_share, self.min_inner, self.inner_frac, self.keep_kinds)
        rows = []
        for fr in t2["frames"]:
            if fr[1]:
                rows.append((seconds(d, fr[0]), [enlarge(b[:4], self.pad, self.width, self.height) for b in fr[1]]))
        rows.sort(key=lambda r: r[0])
        self.times: List[float] = [r[0] for r in rows]
        self.boxes: List[List[List[float]]] = [r[1] for r in rows]

    # kept for readers of the production code
    @property
    def W(self) -> int:
        return self.width

    @property
    def H(self) -> int:
        return self.height

    def __bool__(self) -> bool:
        """False when no frame of the video gets a mosaic."""
        return bool(self.times)

    def block(self) -> int:
        """Mosaic cell size for this video's height."""
        return block_size(self.height, self.block_divisor, self.block_min)

    def boxes_at(self, t: float) -> List[List[float]]:
        """Enlarged boxes ``[[x1, y1, x2, y2], ...]`` to mosaic in the frame at ``t`` seconds: the
        boxes of every table row within ``t +- hold`` (duplicates by coordinates rounded to 0.1 px
        removed, first occurrence kept)."""
        i = bisect.bisect_left(self.times, t - self.hold - 1e-9)
        j = bisect.bisect_right(self.times, t + self.hold + 1e-9)
        out, seen = [], set()
        for k in range(i, j):
            for b in self.boxes[k]:
                key = tuple(round(x, 1) for x in b)
                if key not in seen:
                    seen.add(key)
                    out.append(b)
        return out

    def intervals(self, duration: Optional[float] = None) -> List[Tuple[float, float]]:
        """Sorted, merged ``[(t0, t1), ...]`` seconds in which some mosaic is drawn (every table
        row's time +- hold, starting at 0 and, with ``duration``, ending at most at ``duration``)."""
        iv: List[List[float]] = []
        for t in self.times:
            a, b = max(0.0, t - self.hold), t + self.hold
            if duration:
                b = min(b, duration)
            if iv and a <= iv[-1][1]:
                iv[-1][1] = max(iv[-1][1], b)
            else:
                iv.append([a, b])
        return [(x[0], x[1]) for x in iv]

    def masked_seconds(self, duration: Optional[float] = None) -> float:
        """Total length of :meth:`intervals` in seconds."""
        return sum(b - a for a, b in self.intervals(duration))
