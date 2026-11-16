# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Box rules: which boxes are kept, per detector frame and per table frame.

Detection (per detector frame, in this order)
    :func:`candidates`     proposer boxes -> candidates (score >= thr, best first, a box that
                           overlaps a better one at IoU >= merge_iou is merged into it)
    :func:`postprocess`    boxes the verifier kept -> near duplicates removed (:func:`dedupe`),
                           a loose box around two or more smaller ones replaced by their union
                           (:func:`container_rule`)

Apply (per table frame, before the enlargement and the temporal hold)
    :func:`giant_rule`, :func:`apply_rules_to_table`, :func:`firing_episodes`

THE GIANT RULE
    When two people stand far apart, a detector sometimes also returns one big box around both;
    after the 20% enlargement the mosaic then covers most of the image. The giant rule drops that
    big box and keeps the boxes of the individual people. For every box B of a frame::

        B is a giant    if area(B) >= min_share * W * H     (raw box clipped to the image)
        inner(B)        = the other boxes b with 0 < area(b) < area(B) and
                          inter(B, b) / area(b) >= inner_frac             (b lies >= 80% inside B)
        B is dropped    if B is a giant and len(inner(B)) >= min_inner
                        and the kind of B is not in keep_kinds

    Every box is judged against the ORIGINAL list of its frame, so the result does not depend on
    the order of the boxes; boxes keep their order and their extra fields; the input is never
    modified. Boxes are clipped to the image before their area is measured; returned boxes keep
    their original coordinates. A box of zero area is never a giant and never an inner box.

    Safety argument, and its limit: only a box with at least ``min_inner`` (>= 1) other boxes
    inside it is ever dropped, and those boxes stay, so the people they surround remain masked by
    their own boxes. A lone big box (for example a person close to the camera) is never dropped.
    The rule therefore never empties a frame (the smallest box has nothing inside it) and is
    idempotent. It does NOT prove that nobody is left uncovered: a big box around ONE person who
    was also boxed in parts (head, torso) is dropped and only the parts stay. That has to be
    checked by a human on the worst frame of every firing episode before release;
    :func:`firing_episodes` lists them. In the case study a domain reviewer checked such frames:
    0.50 left one unsafe frame of 91, 0.40 three of 139, 0.55 none of 45, so 0.55 is the default.

    Boxes of a kind in ``keep_kinds`` (by default ``manual``, boxes a reviewer added) are never
    dropped, but they do count as inner boxes of other boxes.

Boxes are ``[x1, y1, x2, y2, ...]`` in pixels of the frame they were found in.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .geometry import area, clip_box, enlarge, inter, iou, share
from .table import BoxTable, seconds

__all__ = [
    "DEFAULT_MIN_SHARE", "DEFAULT_MIN_INNER", "DEFAULT_INNER_FRAC", "DEFAULT_KEEP_KINDS",
    "DEFAULT_PAD", "EPISODE_GAP_S", "candidates", "dedupe", "container_rule", "postprocess",
    "giant_rule", "apply_rules_to_table", "covered_share", "firing_episodes", "format_stats",
]

DEFAULT_MIN_SHARE = 0.55      # a giant covers at least this share of the image (raw box)
DEFAULT_MIN_INNER = 2         # ... and is dropped when at least this many smaller boxes lie inside it
DEFAULT_INNER_FRAC = 0.8      # a box lies inside another when at least this share of its area does
DEFAULT_KEEP_KINDS = ("manual",)   # boxes of these kinds are never dropped by the giant rule
DEFAULT_PAD = 0.20            # mask enlargement per side, used by covered_share
EPISODE_GAP_S = 1.0           # firings at most this many seconds apart form one episode


# ---------------------------------------------------------------------------------------------
# detection: candidates and post-processing
# ---------------------------------------------------------------------------------------------
def candidates(boxes: Iterable[Sequence[float]], thr: float = 0.05, merge_iou: float = 0.6) -> List[Sequence[float]]:
    """Proposer boxes ``[[x1, y1, x2, y2, score, ...], ...]`` -> candidates.

    Keeps the boxes with ``score >= thr``, best first (equal scores keep their input order); a box
    overlapping an already kept one at IoU >= ``merge_iou`` is dropped. This is the rule the crop
    verifier was trained on.
    """
    rows = sorted((b for b in boxes if b[4] >= thr), key=lambda b: -b[4])
    keep: List[Sequence[float]] = []
    for b in rows:
        if all(iou(b, k) < merge_iou for k in keep):
            keep.append(b)
    return keep


def dedupe(boxes: Iterable[Sequence[float]], nms: float = 0.85) -> List[Sequence[float]]:
    """Drop a box that is (nearly) the same as a better one (IoU >= ``nms``).

    Looser overlaps stay: a loose box with a slightly higher score must not hide the tight one.
    Boxes are ``[x1, y1, x2, y2, score, ...]``; the result is sorted best first.
    """
    keep: List[Sequence[float]] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(b, k) < nms for k in keep):
            keep.append(b)
    return keep


def container_rule(boxes: Sequence[Sequence[float]], min_inner: int = 2, ratio: float = 3.0,
                   inner_frac: float = 0.8) -> List[Sequence[float]]:
    """Replace a loose box around several people by the union of their boxes.

    A box that contains at least ``min_inner`` other boxes (each at least ``inner_frac`` inside
    it, smaller than it and not already replaced) and whose area is at least ``ratio`` times the
    area of their union is replaced by that union (which keeps the score of the replaced box).
    Boxes are processed in input order; replaced boxes are appended at the end.
    """
    drop, add = set(), []
    for i, B in enumerate(boxes):
        inner = [b for j, b in enumerate(boxes)
                 if j != i and j not in drop and area(b) < area(B) and inter(B, b) / (area(b) + 1e-9) >= inner_frac]
        if len(inner) >= min_inner:
            u = [min(b[0] for b in inner), min(b[1] for b in inner), max(b[2] for b in inner), max(b[3] for b in inner)]
            if area(B) >= ratio * area(u):
                drop.add(i)
                add.append(u + [B[4]])
    return [b for i, b in enumerate(boxes) if i not in drop] + add


def postprocess(boxes: Iterable[Sequence[float]], nms: float = 0.85, container: bool = True,
                min_inner: int = 2, ratio: float = 3.0, inner_frac: float = 0.8) -> List[Sequence[float]]:
    """:func:`dedupe` at IoU ``nms``, then (if ``container``) :func:`container_rule`."""
    out = dedupe(boxes, nms)
    return container_rule(out, min_inner, ratio, inner_frac) if container else out


# ---------------------------------------------------------------------------------------------
# apply: the giant rule
# ---------------------------------------------------------------------------------------------
def _check(width: float, height: float, min_share: float, min_inner: int, inner_frac: float) -> None:
    if not (width > 0 and height > 0):
        raise ValueError(f"image size must be positive, got {width} x {height}")
    if not min_share > 0:
        raise ValueError(f"min_share must be > 0, got {min_share}")
    if not (isinstance(min_inner, int) and not isinstance(min_inner, bool) and min_inner >= 1):
        raise ValueError(f"min_inner must be an integer >= 1 (with 0 a lone big box, e.g. a person close to the camera, "
                         f"would be dropped), got {min_inner!r}")
    if not 0 < inner_frac <= 1:
        raise ValueError(f"inner_frac must be in (0, 1], got {inner_frac}")


def _kind(b: Sequence[Any]) -> Any:
    return b[4] if len(b) > 4 else None


def giant_rule(boxes: Sequence[Sequence[Any]], width: float, height: float, min_share: float = DEFAULT_MIN_SHARE,
               min_inner: int = DEFAULT_MIN_INNER, inner_frac: float = DEFAULT_INNER_FRAC,
               keep_kinds: Iterable[str] = DEFAULT_KEEP_KINDS) -> Tuple[List[list], List[list]]:
    """Drop every box that covers >= ``min_share`` of the image and has >= ``min_inner`` smaller
    boxes (each >= ``inner_frac`` of its area inside) inside it. See the module docstring.

    ``boxes``: ``[[x1, y1, x2, y2, extra...], ...]`` in pixels of a ``width`` x ``height`` image;
    extra fields are kept. Returns ``(kept, dropped)``: new lists of new boxes (lists), both in
    input order; the input is not modified.
    """
    _check(width, height, min_share, min_inner, inner_frac)
    keep_kinds = tuple(keep_kinds or ())
    clipped = [clip_box(b, width, height) for b in boxes]
    areas = [area(c) for c in clipped]
    limit = min_share * width * height
    kept: List[list] = []
    dropped: List[list] = []
    for i, b in enumerate(boxes):
        if areas[i] >= limit and not (keep_kinds and _kind(b) in keep_kinds):   # limit > 0: a zero-area box is never a giant
            n_inner = 0
            for j, c in enumerate(clipped):
                if j != i and 0 < areas[j] < areas[i] and inter(clipped[i], c) / areas[j] >= inner_frac:
                    n_inner += 1
                    if n_inner >= min_inner:
                        break
            if n_inner >= min_inner:
                dropped.append(list(b))
                continue
        kept.append(list(b))
    return kept, dropped


def _rule_record(min_share: float, min_inner: int, inner_frac: float, keep_kinds: Iterable[str]) -> Dict[str, Any]:
    return dict(rule="giant", min_share=min_share, min_inner=min_inner, inner_frac=inner_frac,
                keep_kinds=list(keep_kinds or ()))


def _same_rule(old: Any, record: Dict[str, Any], frames: Iterable[Any]) -> bool:
    """True when an existing ``box_rules`` record gives the same result as ``record`` on ``frames``.

    A record without ``keep_kinds`` (written by tools that predate it) counts as
    ``keep_kinds=[]``; different keep_kinds are equivalent when no box of the table has a kind
    in which they differ.
    """
    if not isinstance(old, Mapping):
        return False
    for k in ("rule", "min_share", "min_inner", "inner_frac"):
        if old.get(k) != record.get(k):
            return False
    diff = set(old.get("keep_kinds") or ()) ^ set(record.get("keep_kinds") or ())
    if not diff:
        return True
    return not any(_kind(b) in diff for fr in frames for b in (fr[1] or ()))


def apply_rules_to_table(table: Any, min_share: float = DEFAULT_MIN_SHARE, min_inner: int = DEFAULT_MIN_INNER,
                         inner_frac: float = DEFAULT_INNER_FRAC,
                         keep_kinds: Iterable[str] = DEFAULT_KEEP_KINDS) -> Tuple[Any, Dict[str, Any]]:
    """Apply the giant rule to every frame of a box table (:class:`BoxTable` or its dict).

    Returns ``(new_table, stats)``; ``new_table`` has the type of ``table`` and is a copy (the
    input is not modified) in which every frame's boxes are filtered and frames left without boxes
    are removed (also frames that had none); ``params['box_rules']`` records the parameters.
    ``stats``: frames_total_with_boxes, frames_changed (frames that lost a box), boxes_total,
    boxes_dropped, frames_emptied (always 0, see the safety argument), dropped_share_max (the
    largest dropped box as a share of the image, 0.0 if nothing was dropped).

    A table that already carries a different ``box_rules`` record is refused: apply the rule to
    the original table, not on top of other parameters. A record that differs only by a missing
    or different ``keep_kinds`` is accepted when no box of the table has one of the kinds in
    question (the result is then the same).
    """
    if isinstance(table, BoxTable):
        new, stats = apply_rules_to_table(table.to_dict(include_source=True), min_share, min_inner, inner_frac, keep_kinds)
        return BoxTable.from_dict(new), stats
    width, height = table["width"], table["height"]
    _check(width, height, min_share, min_inner, inner_frac)
    record = _rule_record(min_share, min_inner, inner_frac, keep_kinds)
    old = (table.get("params") or {}).get("box_rules")
    if old is not None and old != record and not _same_rule(old, record, table["frames"]):
        raise ValueError(f"the table was already filtered with {old}; apply the rule to the original table, "
                         f"not on top of other parameters ({record})")
    stats: Dict[str, Any] = dict(frames_total_with_boxes=0, frames_changed=0, boxes_total=0, boxes_dropped=0,
                                 frames_emptied=0, dropped_share_max=0.0)
    frames = []
    for fr in table["frames"]:
        boxes = fr[1]
        if not boxes:
            continue                                   # a frame without a box carries no information
        kept, dropped = giant_rule(boxes, width, height, min_share, min_inner, inner_frac, keep_kinds)
        stats["frames_total_with_boxes"] += 1
        stats["boxes_total"] += len(boxes)
        if dropped:
            stats["frames_changed"] += 1
            stats["boxes_dropped"] += len(dropped)
            stats["dropped_share_max"] = max(stats["dropped_share_max"], max(share(d, width, height) for d in dropped))
        if kept:
            frames.append([fr[0], kept] + list(fr[2:]))
        else:
            stats["frames_emptied"] += 1
    new: Dict[str, Any] = {}
    for key, value in table.items():
        if key == "frames":
            new[key] = frames
        elif key == "params":
            p = copy.deepcopy(value) if value else {}
            p["box_rules"] = record
            new[key] = p
        else:
            new[key] = copy.deepcopy(value)
    if "params" not in new:
        new["params"] = {"box_rules": record}
    return new, stats


# ---------------------------------------------------------------------------------------------
# review support
# ---------------------------------------------------------------------------------------------
def covered_share(box: Sequence[float], others: Iterable[Sequence[float]], width: float, height: float,
                  grid: int = 24, pad: float = DEFAULT_PAD) -> float:
    """Share of the area of ``box`` inside the union of the ``pad``-enlarged ``others`` (what the
    mask still covers of it), sampled on a ``grid`` x ``grid`` raster. 1.0 for an empty box."""
    en = [enlarge(o, pad, width, height) for o in others]
    x1, y1, x2, y2 = box[:4]
    if x2 <= x1 or y2 <= y1:
        return 1.0
    hit = 0
    for i in range(grid):
        y = y1 + (i + 0.5) * (y2 - y1) / grid
        for j in range(grid):
            x = x1 + (j + 0.5) * (x2 - x1) / grid
            if any(e[0] <= x <= e[2] and e[1] <= y <= e[3] for e in en):
                hit += 1
    return hit / (grid * grid)


def firing_episodes(table: Any, min_share: float = DEFAULT_MIN_SHARE, min_inner: int = DEFAULT_MIN_INNER,
                    inner_frac: float = DEFAULT_INNER_FRAC, gap_s: float = EPISODE_GAP_S,
                    keep_kinds: Iterable[str] = DEFAULT_KEEP_KINDS, pad: float = DEFAULT_PAD) -> List[Dict[str, Any]]:
    """The episodes in which the giant rule fires, for the human check that nobody is left uncovered.

    An episode is a run of frames (at most ``gap_s`` apart) in which a box was dropped. One dict
    per episode: ``start_s``, ``end_s`` (seconds from the first frame of the table), ``frames``,
    ``boxes``, ``max_share`` (largest dropped box, share of the image), ``min_covered_after``
    (the lowest share of a dropped box that the enlarged kept boxes of its frame still cover; a low
    value means the people inside it may be only partly covered by their own boxes),
    ``pts_at_min`` / ``t_at_min_s`` (the frame to look at), ``n_inner_at_min``, ``kind_at_min``.
    """
    width, height = _field(table, "width"), _field(table, "height")
    _check(width, height, min_share, min_inner, inner_frac)
    episodes: List[Dict[str, Any]] = []
    for fr in _field(table, "frames"):
        pts, boxes = fr[0], fr[1]
        kept, dropped = giant_rule(boxes, width, height, min_share, min_inner, inner_frac, keep_kinds)
        if not dropped:
            continue
        t = seconds(table, pts)
        for g in dropped:
            gc = clip_box(g, width, height)
            cov = covered_share(gc, kept, width, height, pad=pad)
            n_in = 0
            for q in boxes:
                qc = clip_box(q, width, height)
                if 0 < area(qc) < area(gc) and inter(gc, qc) / area(qc) >= inner_frac:
                    n_in += 1
            item = dict(t=t, pts=pts, share=share(g, width, height), cov=cov, n_inner=n_in,
                        kind=g[4] if len(g) > 4 else "")
            if episodes and t - episodes[-1]["items"][-1]["t"] <= gap_s:
                episodes[-1]["items"].append(item)
            else:
                episodes.append(dict(items=[item]))
    out = []
    for e in episodes:
        it = e["items"]
        w = min(it, key=lambda x: x["cov"])
        out.append(dict(start_s=round(it[0]["t"], 3), end_s=round(it[-1]["t"], 3), frames=len({x["pts"] for x in it}),
                        boxes=len(it), max_share=round(max(x["share"] for x in it), 4),
                        min_covered_after=round(w["cov"], 3), pts_at_min=w["pts"], t_at_min_s=round(w["t"], 3),
                        n_inner_at_min=w["n_inner"], kind_at_min=w["kind"]))
    return out


def _field(table: Any, key: str) -> Any:
    return getattr(table, key) if isinstance(table, BoxTable) else table[key]


def format_stats(stats: Mapping[str, Any]) -> str:
    """Readable multi-line summary of the stats of :func:`apply_rules_to_table`."""
    n, nb = stats["frames_total_with_boxes"], stats["boxes_total"]
    lines = [f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}" for k, v in stats.items()]
    lines.append(f"  {'share of frames changed':24s} {stats['frames_changed'] / n:.4%}" if n else f"  {'share of frames changed':24s} n/a")
    lines.append(f"  {'share of boxes dropped':24s} {stats['boxes_dropped'] / nb:.4%}" if nb else f"  {'share of boxes dropped':24s} n/a")
    return "\n".join(lines)
