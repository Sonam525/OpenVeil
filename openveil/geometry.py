# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Box geometry shared by every stage of OpenVeil.

A box is a sequence ``[x1, y1, x2, y2, ...]`` in pixels: (x1, y1) is the top-left corner,
(x2, y2) the bottom-right corner. Extra trailing fields (a kind such as ``"det"``, a score)
are ignored by every function here. Functions never modify their inputs.

These helpers replace the copies that used to live in several modules, so that every stage
measures overlap and area in exactly the same way (the formulas, including the treatment of
degenerate boxes, are kept bit-for-bit identical to the production code).
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

Box = Sequence[float]

__all__ = [
    "area", "inter", "iou", "clip_box", "share", "contains_frac", "union_box",
    "enlarge", "scale_boxes",
]


def area(b: Box) -> float:
    """Area of box ``b``; 0 for a degenerate box (zero or negative width or height)."""
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def inter(a: Box, b: Box) -> float:
    """Area of the intersection of boxes ``a`` and ``b`` (0 when they do not overlap)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0 when the union is empty.

    The union uses the signed areas ``(x2 - x1) * (y2 - y1)`` of the two boxes, as the
    detector post-processing always did; for valid boxes this is the usual IoU.
    """
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0.0


def clip_box(b: Box, width: float, height: float) -> List[float]:
    """The part of box ``b`` inside the image ``[0, width] x [0, height]`` as ``[x1, y1, x2, y2]``.

    Extra fields are dropped. A box completely outside the image becomes a degenerate box
    of zero area.
    """
    return [min(max(b[0], 0.0), width), min(max(b[1], 0.0), height),
            min(max(b[2], 0.0), width), min(max(b[3], 0.0), height)]


def share(b: Box, width: float, height: float) -> float:
    """Share of the image covered by the part of box ``b`` that lies inside it (0 to 1)."""
    return area(clip_box(b, width, height)) / (width * height)


def contains_frac(inner_box: Box, outer_box: Box) -> float:
    """Share of the area of ``inner_box`` that lies inside ``outer_box``.

    Returns 0.0 for a degenerate ``inner_box`` (zero area), so such a box never counts as
    lying inside another one.
    """
    a = area(inner_box)
    return inter(outer_box, inner_box) / a if a > 0 else 0.0


def union_box(boxes: Iterable[Box]) -> List[float]:
    """The smallest box ``[x1, y1, x2, y2]`` that encloses all ``boxes`` (at least one box)."""
    boxes = list(boxes)
    if not boxes:
        raise ValueError("union_box needs at least one box")
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def enlarge(b: Box, pad: float, width: float, height: float) -> List[float]:
    """Box ``b`` grown by ``pad`` times its width (left and right) and its height (top and
    bottom), clipped to the image ``[0, width] x [0, height]``.

    ``pad=0.20`` is the mask enlargement of the production pipeline: a 100 x 200 px box grows
    by 20 px on the left and right and by 40 px at the top and bottom.
    """
    w, h = b[2] - b[0], b[3] - b[1]
    return [max(b[0] - pad * w, 0.0), max(b[1] - pad * h, 0.0),
            min(b[2] + pad * w, width), min(b[3] + pad * h, height)]


def scale_boxes(boxes: Iterable[Box], sx: float, sy: float) -> List[List[float]]:
    """Boxes scaled by ``sx`` horizontally and ``sy`` vertically (for example 0.5, 0.5 to map
    luma coordinates onto the half-size chroma planes of an I420 frame)."""
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
