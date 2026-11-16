# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Choose which frames to label for verifier training.

Run the proposer (``detailed=True``) on a sparse sample of every video, for example one frame
every 10 s, then call :func:`select_frames`. Frames are sorted into buckets by their best
candidate (:func:`openveil.train.labels.bucket`):

* ``strong``: clearly a person (40 % of the frames by default);
* ``medium``: an uncertain person-like box (33 %), the cases the verifier must learn;
* ``animal_like``: a person-like box that scores higher as a competitor phrase (17 %), the hard
  negatives;
* ``random``: anything else (10 %), ordinary footage by day and night.

Quotas are per camera. A bucket that runs short passes its shortfall to the next bucket in that
order; if ``random`` runs short as well, the rest is drawn from ``medium``, ``animal_like`` and
``strong`` again. Within a video, two frames of the same bucket are at least ``spacing_s``
apart. Frames are drawn round-robin over the videos of a camera so that no single video
dominates. Standard library only.

Mined records are dicts ``{video, camera, t, boxes, ...}`` (``t`` in seconds, ``boxes`` the
detailed proposer boxes); other keys are copied to the output.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from .labels import bucket, phrase_candidates

__all__ = ["QUOTA", "ORDER", "select_frames"]

QUOTA = dict(strong=0.40, medium=0.33, animal_like=0.17, random=0.10)
ORDER = ("strong", "medium", "animal_like", "random")
REFILL = ("medium", "animal_like", "strong")


def select_frames(mined: Iterable[Mapping[str, Any]], phrases: Sequence[str], positive: Sequence[str],
                  competitor: Sequence[str] = (), per_camera: int = 750, spacing_s: float = 30.0,
                  quota: Mapping[str, float] = QUOTA, holdout: Iterable[str] = (),
                  exclude: Callable[[Mapping[str, Any]], bool] | None = None, thr: float = 0.10,
                  seed: int = 7) -> tuple[list[dict], dict]:
    """Pick label frames; returns ``(chosen, report)``.

    ``holdout``: video names to leave out entirely (see
    :func:`openveil.train.labels.holdout_videos`). ``exclude(record)``: True for frames that must
    not be labelled (for example frames near a validation clip). Each chosen frame gets
    ``bucket``, ``cands`` (numbered candidates for the labeller, person score >= ``thr``) and
    ``key`` = ``"<video>:<t with 2 decimals>"``. ``report`` gives, per camera and bucket,
    ``[chosen, available]``.
    """
    rng = random.Random(seed)
    held = set(holdout)
    pool: dict[tuple[str, str], list[dict]] = {}
    recs = sorted((dict(r) for r in mined), key=lambda r: (str(r["video"]), float(r["t"])))
    last: dict[tuple[str, str], float] = {}
    n_excluded = 0
    for r in recs:
        if r["video"] in held or r.get("boxes") is None:
            continue
        if exclude is not None and exclude(r):
            n_excluded += 1
            continue
        cands = phrase_candidates(r["boxes"], phrases, positive, competitor, thr)
        bk = bucket(cands)
        lk = (str(r["video"]), bk)
        if float(r["t"]) - last.get(lk, -1e9) < spacing_s:
            continue
        last[lk] = float(r["t"])
        item = {k: v for k, v in r.items() if k != "boxes"}
        item.update(bucket=bk, cands=cands)
        pool.setdefault((str(r.get("camera", "")), bk), []).append(item)
    chosen: list[dict] = []
    report: dict[str, dict[str, list[int]]] = {}
    for cam in sorted({k[0] for k in pool}):
        left = {}
        for bk in ORDER:
            byv: dict[str, list[dict]] = {}
            for it in pool.get((cam, bk), []):
                byv.setdefault(str(it["video"]), []).append(it)
            for lst in byv.values():
                rng.shuffle(lst)
            vids = list(byv)
            rng.shuffle(vids)
            left[bk] = (vids, byv)

        def draw(bk: str, want: int) -> list[dict]:
            vids, byv = left[bk]
            pick: list[dict] = []
            while len(pick) < want and any(byv.values()):
                for vv in vids:
                    if byv[vv] and len(pick) < want:
                        pick.append(byv[vv].pop())
            return pick

        got = {bk: 0 for bk in ORDER}
        carry = 0
        for bk in ORDER:
            want = int(round(per_camera * quota.get(bk, 0.0))) + carry
            pick = draw(bk, want)
            chosen += pick
            got[bk] += len(pick)
            carry = want - len(pick)
        for bk in REFILL:
            if carry <= 0:
                break
            pick = draw(bk, carry)
            chosen += pick
            got[bk] += len(pick)
            carry -= len(pick)
        report[cam] = {bk: [got[bk], len(pool.get((cam, bk), []))] for bk in ORDER}
    for c in chosen:
        c["key"] = f"{c['video']}:{float(c['t']):.2f}"
    return chosen, dict(per_camera=report, excluded=n_excluded, held_out_videos=len(held))
