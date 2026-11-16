# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Evaluate the detector cascade from out-of-fold verifier scores.

The unit of evaluation is the cascade "proposer candidates -> verifier keeps a box if its score
>= t", measured on the labelled frames of a crop table
(:func:`openveil.train.labels.build_crop_table`):

* **people recall**: share of the labelled people (also those no candidate covers) that have at
  least one kept candidate covering at least half of them;
* **false-alarm frames**: share of the person-free labelled frames (also those without any
  candidate) with at least one kept candidate.

:func:`operating_points` finds, for each target recall, the largest threshold that still
reaches it and the false-alarm rates there, overall and per camera / day-night / colour-grey.
Label frames are usually sampled by bucket, which over-represents frames with a strong
proposer hit; :func:`frame_weights` re-weights them to the bucket shares of everyday footage.
:func:`ordinary_burden` measures, on unlabelled everyday frames, how many frames get a mosaic
and how much of the image it covers, with 95 % intervals from resampling whole videos.

numpy only (no scikit-learn needed).
"""
from __future__ import annotations

import collections
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["TARGETS", "load_oof_scores", "auc", "average_precision", "metrics", "frame_weights", "operating_points",
           "area_share", "ordinary_burden"]

TARGETS = (0.90, 0.95, 0.97)


def load_oof_scores(path: str, n: int) -> tuple[np.ndarray, dict]:
    """Out-of-fold scores written by :func:`openveil.train.classifier.cross_validate`.

    Returns ``(scores, state)``; ``scores[i]`` is the score of row ``i`` (NaN when no fold scored it).
    """
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    s = np.full(n, np.nan)
    for k in st["folds"].values():
        for i, v in k["scores"].items():
            s[int(i)] = v
    return s, st


def auc(y: Sequence[int], p: Sequence[float]) -> float:
    """Area under the ROC curve (ties count one half); NaN with a single class."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, float)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y: Sequence[int], p: Sequence[float]) -> float:
    """Average precision (step-wise area under the precision-recall curve, as scikit-learn)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, float)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]       # last index of each distinct threshold
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / npos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def metrics(rows: Sequence[Mapping[str, Any]], p: Sequence[float]) -> dict:
    """``{auc, ap}`` of scores ``p`` against ``row["label"]``."""
    y = [int(r["label"]) for r in rows]
    return dict(auc=auc(y, p), ap=average_precision(y, p))


def frame_weights(frames: Sequence[Mapping[str, Any]], share: Mapping[str, float] | None) -> dict[str, float]:
    """Weight per frame key so that the buckets get the shares ``share`` (``None``: weight 1)."""
    if share is None:
        return {f["key"]: 1.0 for f in frames}
    n = collections.Counter(f["bucket"] for f in frames)
    return {f["key"]: share.get(f["bucket"], 0.0) / n[f["bucket"]] for f in frames}


def operating_points(rows: Sequence[Mapping[str, Any]], people: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]],
                     score: Mapping[int, float] | Sequence[float], weights: Mapping[str, float],
                     targets: Sequence[float] = TARGETS) -> dict:
    """Threshold and false-alarm rates for each target people recall.

    ``score[row id]`` is the verifier score of a candidate row. For each target the largest
    threshold that reaches it is chosen (``reachable=False`` with the ``ceiling`` recall when no
    threshold does, because some people have no covering candidate). False-alarm rates are
    given overall (``fp_all``), by clock day / night, by colour / grey frames and by camera.
    """
    w_p = np.array([weights[p["key"]] for p in people], float)
    best = np.array([max([score[i] for i in p["cand"]], default=-1.0) for p in people], float)
    by_frame: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        by_frame[r["key"]].append(score[r["id"]])
    free = [f for f in frames if f["n_people"] == 0]
    fmax = {f["key"]: max(by_frame[f["key"]], default=-1.0) for f in free}
    wf = {f["key"]: weights[f["key"]] for f in free}

    def rate(sel: Callable[[Mapping[str, Any]], bool], t: float) -> float:
        s = [f for f in free if sel(f)]
        tot = sum(wf[f["key"]] for f in s)
        return sum(wf[f["key"]] * (fmax[f["key"]] >= t) for f in s) / tot if tot else float("nan")

    def recall(sel: Callable[[Mapping[str, Any]], bool], t: float) -> float:
        m = np.array([sel(p) for p in people], bool)
        return float((w_p[m] * (best[m] >= t)).sum() / w_p[m].sum()) if m.any() and w_p[m].sum() else float("nan")

    out: dict = {}
    if not len(people) or not w_p.sum():
        return {tgt: dict(reachable=False, ceiling=float("nan")) for tgt in targets}
    cand_t = np.unique(np.concatenate([best[best >= 0], [0.0]]))[::-1]
    for tgt in targets:
        t = next((float(c) for c in cand_t if (w_p * (best >= c)).sum() / w_p.sum() >= tgt), None)
        if t is None:
            out[tgt] = dict(reachable=False, ceiling=float((w_p * (best >= 0)).sum() / w_p.sum()))
            continue
        row = dict(reachable=True, t=t, recall=float((w_p * (best >= t)).sum() / w_p.sum()))
        row["fp_all"] = rate(lambda f: True, t)
        for name, sel in (("clock_day", lambda f: not f.get("night")), ("clock_night", lambda f: bool(f.get("night"))),
                          ("colour", lambda f: not f.get("ir")), ("grey", lambda f: bool(f.get("ir")))):
            row[f"fp_{name}"] = rate(sel, t)
        for cam in sorted({str(f.get("camera", "")) for f in frames}):
            row[f"fp_colour_{cam}"] = rate(lambda f, cam=cam: str(f.get("camera", "")) == cam and not f.get("ir"), t)
            row[f"recall_{cam}"] = recall(lambda p, cam=cam: str(p.get("camera", "")) == cam, t)
        row["recall_colour"] = recall(lambda p: not p.get("ir"), t)
        row["recall_grey"] = recall(lambda p: bool(p.get("ir")), t)
        row["recall_clock_night"] = recall(lambda p: bool(p.get("night")), t)
        row["n_people_grey"] = int(sum(bool(p.get("ir")) for p in people))
        row["n_people_clock_night"] = int(sum(bool(p.get("night")) for p in people))
        out[tgt] = row
    return out


def area_share(boxes: Sequence[Sequence[float]], width: int = 1280, height: int = 720, grid: int = 4) -> float:
    """Share of the image covered by the union of ``boxes``, measured on a ``grid``-pixel raster."""
    if not boxes:
        return 0.0
    m = np.zeros((height // grid, width // grid), bool)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v / grid)) for v in b[:4]]
        m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
    return float(m.mean())


def ordinary_burden(ordinary: Mapping[str, Any], scores: Mapping[str, float], t: float, frame_ir: Mapping[str, bool] | None = None,
                    boot: int = 1000, seed: int = 0) -> dict:
    """Mosaic burden on unlabelled everyday frames at threshold ``t``.

    ``ordinary = {frames: [{fid, video, camera, night, width, height}], candidates: [{id, fid, box}]}``,
    ``scores[str(candidate id)]``. Returns per cut (``all``, ``clock_day``, ``clock_night``,
    optional ``colour`` / ``grey``, each camera): ``n`` frames, ``with_mosaic`` and ``area`` as
    ``(mean, low, high)`` with 95 % intervals from ``boot`` resamples of whole videos.
    """
    by: dict[Any, list] = collections.defaultdict(list)
    for r in ordinary["candidates"]:
        by[r["fid"]].append(r)
    fr = list(ordinary["frames"])
    hit = np.zeros(len(fr))
    ar = np.zeros(len(fr))
    for i, f in enumerate(fr):
        kept = [r["box"] for r in by[f["fid"]] if scores[str(r["id"])] >= t]
        hit[i] = bool(kept)
        ar[i] = area_share(kept, int(f.get("width", 1280)), int(f.get("height", 720)))
    vids = sorted({f["video"] for f in fr})
    vi = {v: [i for i, f in enumerate(fr) if f["video"] == v] for v in vids}
    rng = np.random.default_rng(seed)

    def ci(x: np.ndarray, sel: np.ndarray) -> tuple[float, float, float]:
        if not sel.any():
            return (float("nan"),) * 3
        vals = []
        for _ in range(boot):
            ids = np.concatenate([vi[v] for v in rng.choice(vids, len(vids))]).astype(int)
            ids = ids[sel[ids]]
            vals.append(x[ids].mean() if len(ids) else np.nan)
        return float(x[sel].mean()), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

    night = np.array([bool(f.get("night")) for f in fr], bool)
    cuts = {"all": np.ones(len(fr), bool), "clock_day": ~night, "clock_night": night}
    if frame_ir:
        grey = np.array([bool(frame_ir[str(f["fid"])]) for f in fr], bool)
        cuts.update({"colour": ~grey, "grey": grey})
    for c in sorted({str(f.get("camera", "")) for f in fr}):
        cuts[c] = np.array([str(f.get("camera", "")) == c for f in fr], bool)
    return {k: dict(n=int(v.sum()), with_mosaic=ci(hit, v), area=ci(ar, v)) for k, v in cuts.items()}
