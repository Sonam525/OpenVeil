# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Training data for the crop verifier: candidate boxes, labels and cross-validation folds.

The verifier learns to answer one question about a proposer box: does it protect a person?
Training rows are built from frames in which a human drew a box around every person:

1. the proposer's boxes on the frame become **candidates** with :func:`phrase_candidates`
   (positive-phrase score >= ``thr``, best first, a box overlapping a better one at IoU >= 0.6
   dropped), the same rule the detector applies at run time;
2. each candidate is **labelled** with :func:`label_candidate`: positive when it covers at
   least half of a labelled person (optionally only when that person also fills at least
   ``tight`` of the box, so that a loose box around a whole pen is not taught as correct);
3. rows are split into **folds** whose groups are whole videos (or recording days, or
   cameras), so that a test fold never shares a video with its training data
   (:func:`assign_folds`, :func:`add_folds`).

:func:`build_crop_table` does all three and returns the table used by
:mod:`openveil.train.classifier` and :mod:`openveil.train.evaluate`.

Input records (plain dicts, so any labelling tool can produce them)::

    frame = {"key": "cam1_0800.mp4:12.00",      # unique frame id
             "video": "cam1_0800.mp4", "camera": "cam1", "day": "2024-03-05",
             "image": "frames/cam1_0800_000012.jpg", "width": 1280, "height": 720,
             "night": False,
             "boxes": [[x1, y1, x2, y2, s_phrase1, s_phrase2, ...], ...]}   # proposer, detailed
    label = {"key": "cam1_0800.mp4:12.00", "persons": [[x1, y1, x2, y2], ...]}

Box coordinates are pixels of the stored frame image. Standard library and numpy only.
"""
from __future__ import annotations

import collections
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..geometry import area, inter, iou

__all__ = [
    "phrase_candidates",
    "bucket",
    "label_candidate",
    "assign_folds",
    "add_folds",
    "holdout_videos",
    "colourfulness",
    "is_grey",
    "build_crop_table",
    "GREY_COLOURFULNESS",
    "MERGE_IOU",
]

MERGE_IOU = 0.6
GREY_COLOURFULNESS = 0.05      # mean(|R-G| + |G-B|) at 320 x 180 below this = a grey (infrared) frame


def phrase_candidates(boxes: Iterable[Sequence[float]] | None, phrases: Sequence[str], positive: Sequence[str],
                      competitor: Sequence[str] = (), thr: float = 0.10, merge_iou: float = MERGE_IOU) -> list[dict]:
    """Candidate boxes from detailed proposer boxes ``[x1, y1, x2, y2, s_1, ..., s_n]``.

    ``phrases[i]`` names score ``s_(i+1)``. A box's ``person`` score is its highest positive-phrase
    score, its ``animal`` score the highest competitor-phrase score (0 without competitors).
    Boxes with ``person >= thr`` are kept best first; a box overlapping a better kept box at
    IoU >= ``merge_iou`` is dropped. The ``animal`` score of a kept box is then raised to the
    highest competitor score of any proposer box at the same place (IoU >= ``merge_iou``): the
    proposer often returns one box scored as a person and a near-identical one scored as an
    animal. Kept boxes are numbered from 1 (``id``).
    """
    boxes = list(boxes or [])
    pi = [i for i, p in enumerate(phrases) if p in set(positive)]
    ai = [i for i, p in enumerate(phrases) if p in set(competitor)]
    if not pi:
        raise ValueError(f"none of the positive phrases {list(positive)} is in {list(phrases)}")
    rows = []
    for b in boxes:
        sc = list(b[4:])
        s = max(sc[i] for i in pi)
        a = max((sc[i] for i in ai), default=0.0)
        if s >= thr:
            rows.append(dict(box=list(b[:4]), person=round(s, 3), animal=round(a, 3), scores=dict(zip(phrases, sc))))
    rows.sort(key=lambda r: -r["person"])
    keep: list[dict] = []
    for r in rows:
        if all(iou(r["box"], k["box"]) < merge_iou for k in keep):
            keep.append(r)
    for n, r in enumerate(keep, 1):
        r["id"] = n
        if ai:
            near = [max(b[4 + j] for j in ai) for b in boxes if iou(b[:4], r["box"]) >= merge_iou]
            r["animal"] = round(max([r["animal"]] + near), 3)
    return keep


def bucket(cands: Sequence[Mapping[str, float]]) -> str:
    """Sampling bucket of a frame from its candidates.

    ``strong``: a box with person >= 0.35 and at least 0.10 above its animal score;
    ``medium``: a box with person >= 0.12 not below its animal score;
    ``animal_like``: a box with person >= 0.12 (but below its animal score);
    ``random``: anything else.
    """
    if any(c["person"] >= 0.35 and c["person"] - c["animal"] >= 0.10 for c in cands):
        return "strong"
    if any(c["person"] >= 0.12 and c["person"] - c["animal"] >= 0.0 for c in cands):
        return "medium"
    if any(c["person"] >= 0.12 for c in cands):
        return "animal_like"
    return "random"


def label_candidate(box: Sequence[float], persons: Sequence[Sequence[float]], tight: float = 0.0) -> dict:
    """Label one candidate box against the labelled ``persons`` of its frame.

    Returns ``{label, kind, covers, iou_max}``: ``covers`` lists the persons of which the box
    covers at least half (and, with ``tight`` > 0, which fill at least ``tight`` of the box);
    ``label`` is 1 when ``covers`` is not empty; ``kind`` is ``"person"`` (IoU >= 0.5 with a
    person), ``"mixed"`` (covers a person but IoU < 0.5: a person plus something else, or a
    loose box) or ``"none"``.
    """
    ious = [inter(box, p) / (area(box) + area(p) - inter(box, p) + 1e-9) for p in persons]
    cov = [inter(box, p) / (area(p) + 1e-9) for p in persons]
    covers = [i for i, v in enumerate(cov) if v >= 0.5 and area(persons[i]) >= tight * area(box)]
    kind = "person" if any(v >= 0.5 for v in ious) else ("mixed" if covers else "none")
    return dict(label=int(bool(covers)), kind=kind, covers=covers, iou_max=round(max(ious, default=0.0), 4))


def assign_folds(groups_with_n: Mapping[str, Sequence[int]], k: int) -> tuple[dict[str, int], list[list[int]]]:
    """Greedy balanced assignment of groups to ``k`` folds.

    ``groups_with_n`` maps a group name to ``[positives, rows]``. Groups are placed largest
    first (by positives, then rows, then name) into the fold with the fewest positives (then
    rows, then the lowest index), so the result is deterministic. Returns ``(fold_of_group,
    load)`` with ``load[j] = [positives, rows]``.
    """
    order = sorted(groups_with_n, key=lambda g: (-groups_with_n[g][0], -groups_with_n[g][1], g))
    load = [[0, 0] for _ in range(k)]
    fold: dict[str, int] = {}
    for g in order:
        j = min(range(k), key=lambda i: (load[i][0], load[i][1], i))
        fold[g] = j
        load[j][0] += groups_with_n[g][0]
        load[j][1] += groups_with_n[g][1]
    return fold, load


SPLITS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "video": lambda r: str(r["video"]),
    "day": lambda r: f"{r.get('camera', '')}|{r.get('day', '')}",
    "camera": lambda r: str(r.get("camera", "")),
}


def add_folds(rows: list[dict], k: int = 5, split: str = "video", field: str = "fold") -> list[list[int]]:
    """Write a fold number into every row (``row[field]``), grouping rows by ``split``.

    ``split`` is ``"video"`` (default: no video in two folds), ``"day"`` (camera + recording
    day) or ``"camera"`` (leave one camera out; ``k`` is then the number of cameras). Rows that
    already carry ``field = -1`` stay in training for every fold. Returns the fold loads.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, not {split!r}")
    key = SPLITS[split]
    use = [r for r in rows if r.get(field, 0) != -1]
    if split == "camera":
        cams = sorted({key(r) for r in use})
        for r in use:
            r[field] = cams.index(key(r))
        load = [[sum(r["label"] for r in use if r[field] == j), sum(1 for r in use if r[field] == j)] for j in range(len(cams))]
        return load
    groups: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in use:
        g = key(r)
        groups[g][0] += int(r.get("label", 0))
        groups[g][1] += 1
    fold, load = assign_folds(groups, k)
    for r in use:
        r[field] = fold[key(r)]
    return load


def holdout_videos(videos: Sequence[Mapping[str, Any]], share: float = 0.10, seed: int = 13) -> list[str]:
    """A random ``share`` of the videos of every camera (at least one per camera), sorted.

    Held-out videos give unlabelled everyday frames to measure the false-mosaic burden on
    footage the verifier never saw. ``videos`` are dicts with ``name`` and ``camera``.
    """
    rng = random.Random(seed)
    out: list[str] = []
    for cam in sorted({str(v.get("camera", "")) for v in videos}):
        names = sorted(str(v["name"]) for v in videos if str(v.get("camera", "")) == cam)
        out += rng.sample(names, max(1, int(round(len(names) * share))))
    return sorted(out)


def colourfulness(rgb_or_bgr: np.ndarray) -> float:
    """Mean ``|R-G| + |G-B|`` of the image at 320 x 180 (0 for a grey infrared frame)."""
    from ..detect.verifier import _resize  # noqa: PLC0415

    s = _resize(np.ascontiguousarray(rgb_or_bgr), 320, 180).astype(np.int16)
    return float((np.abs(s[:, :, 2] - s[:, :, 1]) + np.abs(s[:, :, 1] - s[:, :, 0])).mean())


def is_grey(img: np.ndarray, limit: float = GREY_COLOURFULNESS) -> bool:
    """True for a grey (infrared mode) frame."""
    return colourfulness(img) < limit


def build_crop_table(frames: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
                     phrases: Sequence[str], positive: Sequence[str], competitor: Sequence[str] = (), thr: float = 0.05,
                     tight: float = 0.0, k: int = 5, split: str = "video", grey: Callable[[Mapping[str, Any]], bool] | None = None) -> dict:
    """One training row per candidate box of every labelled frame.

    ``frames`` and ``labels`` are described in the module docstring; frames without a label
    are skipped (counted in ``dropped``). ``grey(frame)`` may return whether a frame is in
    infrared mode (for example ``lambda f: is_grey(load_image(f["image"]))``); without it every
    frame is treated as colour.

    Returns ``{candidates, people, frames, bucket_share, dropped, split, folds}``:

    * ``candidates``: rows ``{id, key, video, camera, day, night, ir, bucket, image, width,
      height, box, person, animal, scores, label, kind, covers, fold}``; ``id`` is the row index;
    * ``people``: one entry per labelled person with ``cand`` = ids of the rows covering it
      (a person no candidate covers can never be found);
    * ``frames``: one entry per labelled frame with ``n_cands`` and ``n_people``;
    * ``bucket_share``: share of the frames per sampling bucket (input of the re-weighting in
      :mod:`openveil.train.evaluate`).
    """
    lab = labels if isinstance(labels, Mapping) else {str(r["key"]): r for r in labels}
    rows: list[dict] = []
    people: list[dict] = []
    out_frames: list[dict] = []
    dropped = 0
    for fr in frames:
        key = str(fr["key"])
        if key not in lab:
            dropped += 1
            continue
        persons = [list(p[:4]) for p in lab[key].get("persons", [])]
        cands = phrase_candidates(fr.get("boxes"), phrases, positive, competitor, thr)
        ir = bool(grey(fr)) if grey is not None else bool(fr.get("ir", False))
        bk = fr.get("bucket") or bucket(cands)
        common = dict(key=key, video=str(fr["video"]), camera=str(fr.get("camera", "")), day=str(fr.get("day", "")),
                      night=bool(fr.get("night", False)), ir=ir, bucket=bk)
        out_frames.append(dict(common, n_cands=len(cands), n_people=len(persons)))
        first = len(rows)
        for c in cands:
            lb = label_candidate(c["box"], persons, tight)
            rows.append(dict(common, id=len(rows), image=fr.get("image"), width=fr.get("width"), height=fr.get("height"),
                             box=c["box"], person=c["person"], animal=c["animal"], scores=c["scores"],
                             label=lb["label"], kind=lb["kind"], covers=lb["covers"]))
        for pi, p in enumerate(persons):
            people.append(dict(common, person=p, cand=[r["id"] for r in rows[first:] if pi in r["covers"]]))
    load = add_folds(rows, k, split) if rows else []
    # frames and people inherit the fold of their video / day / camera
    fold_of = {SPLITS[split](r): r["fold"] for r in rows}
    for item in out_frames + people:
        item["fold"] = fold_of.get(SPLITS[split](item), -1)
    n = len(out_frames)
    share = {b: c / n for b, c in sorted(collections.Counter(f["bucket"] for f in out_frames).items())} if n else {}
    return dict(candidates=rows, people=people, frames=out_frames, bucket_share=share, dropped=dropped, split=split, folds=load)
