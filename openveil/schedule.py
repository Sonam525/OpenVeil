# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Which frames the detector runs on, and how the frames between detections get boxes.

Everything here is free of video and GPU code, so it can be tested and tuned on saved detections.

:func:`plan_chunks`       cut a video into chunks of ``chunk_s`` seconds
:func:`chunk_mode`        the scan mode of a chunk: ``full``, ``sparse`` or ``skip``
                          (:class:`~openveil.config.DayNightPolicy`; default: always ``full``)
:func:`sparse_scan`       run the detector on every ``stride``-th frame (a checkpoint); when a
                          checkpoint has a hit, go back to the previous clear checkpoint and run
                          every ``dense_stride``-th frame from there ("dense mode") until
                          ``clear_gaps`` strides pass without a hit
:func:`interpolate_gaps`  when the same person is boxed before and after a gap, place boxes on
                          the straight line between the two (linear in pts time; boxes matched by
                          centre distance relative to box size, Hungarian assignment)
:func:`scan_chunk`        both, for one chunk, with the production schedule

Frame times are source pts in seconds, so variable frame rates are handled.
Boxes are ``[x1, y1, x2, y2, ...]`` in pixels.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DayNightPolicy, ScheduleConfig
from .geometry import iou

__all__ = [
    "plan_chunks", "clock_hour", "mode_at", "chunk_mode", "start_local_from_name", "sparse_scan",
    "interpolate_gaps", "scan_chunk", "default_is_hit",
]

Detect = Callable[[List[int]], Dict[int, List[Sequence[float]]]]


# ---------------------------------------------------------------------------------------------
# chunks and day / night
# ---------------------------------------------------------------------------------------------
def plan_chunks(duration_s: float, chunk_s: float = 60.0) -> List[Tuple[float, float]]:
    """``[(start_s, dur_s), ...]`` covering ``[0, duration_s)`` in windows of ``chunk_s``.

    At least one chunk; the last one may reach past the end (decoding stops at the end anyway).
    """
    n = max(1, int(math.ceil(duration_s / chunk_s)))
    return [(k * chunk_s, chunk_s) for k in range(n)]


def clock_hour(start_local: str, offset_s: float) -> float:
    """Hour of the day (0-24, fractional) ``offset_s`` seconds into a video that started at
    ``start_local`` (ISO time, for example ``'2024-03-05T12:34:56'``)."""
    t = dt.datetime.fromisoformat(start_local) + dt.timedelta(seconds=float(offset_s))
    return t.hour + t.minute / 60.0 + t.second / 3600.0


def mode_at(start_local: Optional[str], offset_s: float, policy: Optional[DayNightPolicy] = None) -> str:
    """Scan mode at one point of a video.

    ``always_full`` policy (the default), or no ``start_local``: ``"full"``. ``clock`` policy:
    ``"full"`` between ``day_start`` and ``day_end`` of the camera clock, else ``night_mode``.
    When ``day_start`` > ``day_end`` the day crosses midnight (for example 11 to 4 for a camera
    clock in UTC whose local day is 06:00-23:00 at UTC-5).
    """
    p = policy or DayNightPolicy()
    if p.policy == "always_full" or not start_local:
        return "full"
    if p.policy != "clock":
        raise ValueError(f"unknown day/night policy {p.policy!r}")
    h = clock_hour(start_local, offset_s)
    if p.day_start <= p.day_end:
        day = p.day_start <= h < p.day_end
    else:                                   # the day crosses midnight
        day = h >= p.day_start or h < p.day_end
    return "full" if day else p.night_mode


def chunk_mode(start_local: Optional[str], start_s: float, dur_s: float, policy: Optional[DayNightPolicy] = None) -> str:
    """Scan mode of the chunk ``[start_s, start_s + dur_s)``.

    With the ``clock`` policy a chunk is ``full`` if its start, or its end plus ``margin_s``, is
    day (so a chunk shortly before the day starts is already scanned as day: camera clocks can
    lag), else ``night_mode``. With ``always_full``, or without a start time, always ``full``.
    """
    p = policy or DayNightPolicy()
    if p.policy == "always_full" or not start_local:
        return "full"
    mode = p.night_mode
    for o in (start_s, start_s + dur_s + p.margin_s):
        if mode_at(start_local, o, p) == "full":
            mode = "full"
            break
    return mode


def start_local_from_name(name: str, pattern: str = r"(\d{8})[-_](\d{6})") -> Optional[str]:
    """ISO camera start time from a file name such as ``cam1_20240305-123456.mp4``.

    ``pattern`` must capture the date (YYYYMMDD) and the time (HHMMSS). Returns ``None`` when the
    name does not match or the date is invalid.
    """
    m = re.search(pattern, name)
    if not m:
        return None
    try:
        t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return t.isoformat()


# ---------------------------------------------------------------------------------------------
# sparse scan
# ---------------------------------------------------------------------------------------------
def sparse_scan(n_frames: int, detect: Detect, stride: int = 5, clear_gaps: int = 2, lookahead: int = 4,
                is_hit: Callable[[List[Sequence[float]]], bool] = bool, dense_stride: int = 1,
                last_checkpoint: bool = False) -> Tuple[Dict[int, list], List[int]]:
    """Run the checkpoint / dense schedule over ``n_frames`` frames.

    Returns ``(results, ran)``: ``results[f]`` is the box list of every frame the detector ran on
    and ``ran`` the sorted list of those frames.

    ``detect(frame_indices) -> {frame_index: [[x1, y1, x2, y2, score, ...], ...]}`` is a batch
    call (called many times). ``is_hit(boxes)`` decides whether a frame counts as "a person found"
    (it starts or prolongs the dense mode); the default is "any box". ``dense_stride``: in the
    dense mode the detector runs on every ``dense_stride``-th frame only (1 = every frame); the
    frames in between are filled by :func:`interpolate_gaps`. ``last_checkpoint``: the last frame
    is a checkpoint too (without it a person entering in the last ``stride - 1`` frames of a chunk
    is first seen in the next chunk, where the back-fill cannot reach).
    """
    results: Dict[int, list] = {}
    mode = "sparse"
    last_hit = -10 ** 9           # last frame with a hit
    prev_cp = -1                  # last checkpoint that was run in sparse mode
    f = 0                         # next frame not yet decided
    while f < n_frames:
        if mode == "sparse":
            cps = [c for c in range(f, min(n_frames, f + lookahead * stride))
                   if c % stride == 0 or (last_checkpoint and c == n_frames - 1)]
            if not cps:           # f is not at a checkpoint (only right after the dense mode): move to the next one
                f = (f // stride + 1) * stride
                continue
            res = detect(cps)
            hit = None
            for c in cps:
                results[c] = res[c]
                if is_hit(res[c]):
                    hit = c
                    break
            if hit is None:
                prev_cp = cps[-1]
                f = prev_cp + 1
                continue
            # a hit at checkpoint `hit`: run the frames from the previous clear checkpoint on
            start = (cps[cps.index(hit) - 1] if cps.index(hit) > 0 else prev_cp) + 1
            back = [j for j in range(max(start, 0), hit) if j not in results and (j % dense_stride == 0 or j == hit - 1)]
            if back:
                results.update(detect(back))
            last_hit = hit
            mode = "dense"
            f = hit + 1
        else:                     # dense: one stride of frames at a time, then test for clear_gaps clear strides
            end = min(n_frames, f + stride)
            grp = [j for j in range(f, end) if j not in results and (j % dense_stride == 0 or j == end - 1)]
            if grp:
                res = detect(grp)
                results.update(res)
                for j in grp:
                    if is_hit(res[j]):
                        last_hit = max(last_hit, j)
            f = end
            if f - 1 - last_hit >= clear_gaps * stride:
                mode = "sparse"
                prev_cp = ((f - 1) // stride) * stride
                if prev_cp >= f:
                    prev_cp -= stride
    return results, sorted(results)


# ---------------------------------------------------------------------------------------------
# interpolation between two detections of the same person
# ---------------------------------------------------------------------------------------------
def _centre(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _size(b: Sequence[float]) -> float:
    return max(1.0, float(np.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1])))))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    """Centre distance in units of the mean box size."""
    return np.linalg.norm(_centre(a) - _centre(b)) / (0.5 * (_size(a) + _size(b)))


_LSA: List[Any] = []


def _linear_sum_assignment() -> Optional[Callable]:
    """scipy's Hungarian solver, or None when scipy is not installed (imported once, lazily)."""
    if not _LSA:
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            linear_sum_assignment = None
        _LSA.append(linear_sum_assignment)
    return _LSA[0]


def _match(last_boxes: List[Sequence[float]], new_boxes: List[Sequence[float]], gate: float,
           use_scipy: bool = True) -> List[Tuple[int, int]]:
    """Assignment on centre distance / box size minus 2 * IoU; ``[(i_last, j_new)]`` for the pairs
    inside the gate (distance <= gate or IoU > 0.05). Hungarian with scipy, greedy otherwise."""
    if not last_boxes or not new_boxes:
        return []
    cost = np.zeros((len(last_boxes), len(new_boxes)))
    for i, a in enumerate(last_boxes):
        for j, b in enumerate(new_boxes):
            cost[i, j] = _dist(a, b) - 2.0 * iou(a, b)            # prefers overlapping boxes, then nearby ones
    lsa = _linear_sum_assignment() if use_scipy else None
    if lsa is not None:
        ri, ci = lsa(cost)
        pairs = list(zip(ri.tolist(), ci.tolist()))
    else:                                                          # greedy fallback
        pairs, used_i, used_j = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = divmod(int(flat), cost.shape[1])
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)
    out = []
    for i, j in pairs:
        a, b = last_boxes[i], new_boxes[j]
        if _dist(a, b) <= gate or iou(a, b) > 0.05:
            out.append((i, j))
    return out


def _score_of(b: Sequence[Any]) -> Optional[float]:
    if len(b) > 4 and isinstance(b[4], (int, float)) and not isinstance(b[4], bool):
        return float(b[4])
    return None


def interpolate_gaps(results: Mapping[int, List[Sequence[float]]], times: Sequence[float], max_gap_s: float = 2.0,
                     gate_base: float = 1.0, gate_per_s: float = 1.5, keep_score: bool = False,
                     use_scipy: bool = True) -> Dict[int, List[list]]:
    """Fill the frames between two detections of the same person.

    ``results``: ``{frame: [[x1, y1, x2, y2, (score, ...)], ...]}`` for the frames the detector
    ran on. ``times``: pts in seconds of every frame (``len = n_frames``).

    Returns ``{frame: [[x1, y1, x2, y2, kind], ...]}`` with kind ``"det"`` or ``"interp"``; frames
    without a box are absent. With ``keep_score`` a detection keeps its score as a sixth field
    (``[x1, y1, x2, y2, "det", score]``) when it has one.

    A gap of g seconds is bridged only if g <= ``max_gap_s`` and the two boxes lie within
    ``gate_base + gate_per_s * g`` box sizes of each other, or overlap at IoU > 0.05 (a person
    cannot teleport). ``use_scipy=False`` forces the greedy assignment.
    """
    frames = sorted(f for f, b in results.items() if b)
    out: Dict[int, List[list]] = {}
    for f in frames:
        rows = []
        for b in results[f]:
            s = _score_of(b) if keep_score else None
            rows.append([*b[:4], "det"] if s is None else [*b[:4], "det", s])
        out[f] = rows
    tracks: List[Dict[str, Any]] = []                              # each: dict(box, f)
    for f in frames:
        new = [list(b[:4]) for b in results[f]]
        live = [t for t in tracks if times[f] - times[t["f"]] <= max_gap_s]
        pairs: List[Tuple[int, int]] = []
        if live:
            # the gate depends on each track's gap: assign with the loosest gate, check each pair below
            pairs = _match([t["box"] for t in live], new, gate_base + gate_per_s * max_gap_s, use_scipy)
        matched_new = set()
        for i, j in pairs:
            t, b = live[i], new[j]
            gap = times[f] - times[t["f"]]
            if f - t["f"] > 1 and _dist(t["box"], b) > gate_base + gate_per_s * gap and iou(t["box"], b) <= 0.05:
                continue                                           # too far for this gap
            if f - t["f"] > 1:
                t0, t1 = times[t["f"]], times[f]
                for g in range(t["f"] + 1, f):
                    w = (times[g] - t0) / (t1 - t0) if t1 > t0 else 0.5
                    ib = [(1 - w) * t["box"][k] + w * b[k] for k in range(4)]
                    out.setdefault(g, []).append([*ib, "interp"])
            t["box"], t["f"] = b, f
            matched_new.add(j)
        for j, b in enumerate(new):
            if j not in matched_new:
                tracks.append(dict(box=b, f=f))
        tracks = [t for t in tracks if times[f] - times[t["f"]] <= max_gap_s]
    return out


# ---------------------------------------------------------------------------------------------
# one chunk
# ---------------------------------------------------------------------------------------------
def default_is_hit(boxes: List[Sequence[Any]]) -> bool:
    """Day-mode hit test: any box whose sixth field (a hit flag) is > 0; a box without that field
    counts as a hit."""
    return any((b[5] > 0) if len(b) > 5 else True for b in boxes)


def _param(P: Any, name: str, default: Any = None) -> Any:
    if isinstance(P, Mapping):
        return P.get(name, default)
    return getattr(P, name, default)


def scan_chunk(n_frames: int, times: Sequence[float], detect: Detect, mode: str, fps: float, P: Any = None,
               is_hit: Optional[Callable[[List[Sequence[float]]], bool]] = None,
               keep_score: bool = False) -> Tuple[Dict[int, List[list]], List[int]]:
    """Run the schedule on one chunk of ``n_frames`` frames.

    ``times``: seconds of every frame; ``detect(list of frame idx) -> {idx: [[x1, y1, x2, y2,
    score, (hit)], ...]}``; ``mode``: ``"full"``, ``"sparse"`` or ``"skip"``; ``P``: a
    :class:`~openveil.config.ScheduleConfig` or a dict with base_s, dense_s, night_base_s,
    clear_gaps, lookahead, max_gap_s (optional: last_checkpoint, night_gap_factor, gate_base,
    gate_per_s). Returns ``(frame_boxes, ran)``: ``frame_boxes`` maps every frame that carries a
    box to ``[[x1, y1, x2, y2, kind(, score)], ...]``; ``ran`` lists the frames the detector ran on.

    Full mode: checkpoints every ``base_s``, dense every ``dense_s`` around a hit (``is_hit``,
    default :func:`default_is_hit`). Sparse mode: checkpoints every ``night_base_s`` only, a hit
    never starts the dense mode, and gaps up to ``max(max_gap_s, night_gap_factor * night_base_s)``
    are bridged (one missed checkpoint included). In both modes the last frame of the chunk is a
    checkpoint too (``last_checkpoint``, default true).
    """
    P = ScheduleConfig() if P is None else P
    if mode == "skip" or n_frames == 0:
        return {}, []
    gate_base = _param(P, "gate_base", 1.0)
    gate_per_s = _param(P, "gate_per_s", 1.5)
    if mode == "sparse":
        stride = max(1, int(round(_param(P, "night_base_s") * fps)))
        # the last frame is a checkpoint here too: otherwise a person who stays in view loses the
        # mask for up to night_base_s - hold at every chunk boundary and at the end of the video
        results, ran = sparse_scan(n_frames, detect, stride=stride, clear_gaps=_param(P, "clear_gaps"),
                                   lookahead=_param(P, "lookahead"), is_hit=lambda bx: False,
                                   last_checkpoint=_param(P, "last_checkpoint", True))
        max_gap = max(_param(P, "max_gap_s"), _param(P, "night_gap_factor", 2.2) * _param(P, "night_base_s"))
        return interpolate_gaps(results, times, max_gap_s=max_gap, gate_base=gate_base, gate_per_s=gate_per_s,
                                keep_score=keep_score), ran
    if mode != "full":
        raise ValueError(f"unknown scan mode {mode!r} (expected 'full', 'sparse' or 'skip')")
    stride = max(1, int(round(_param(P, "base_s") * fps)))
    dense_s = _param(P, "dense_s")
    dense_stride = max(1, int(round(dense_s * fps))) if dense_s else 1
    results, ran = sparse_scan(n_frames, detect, stride=stride, clear_gaps=_param(P, "clear_gaps"),
                               lookahead=_param(P, "lookahead"), is_hit=is_hit or default_is_hit,
                               dense_stride=dense_stride, last_checkpoint=_param(P, "last_checkpoint", True))
    return interpolate_gaps(results, times, max_gap_s=_param(P, "max_gap_s"), gate_base=gate_base,
                            gate_per_s=gate_per_s, keep_score=keep_score), ran
