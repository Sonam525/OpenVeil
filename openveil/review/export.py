# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Turn the decisions of a review folder into human labels, a summary and corrections for the apply step.

    python -m openveil.review.export REVIEW_DIR [--overrides OUT.json] [--keep-unsure]

Reads ``REVIEW_DIR/manifest.json`` and ``decisions.jsonl``. The rule for every reader is the same:
**the latest decision of a frame wins, then unsure frames are filtered** (a frame confirmed and later
marked unsure is unsure). Writes into REVIEW_DIR:

``human_labels.json``
    ``{frame key: {kind, persons, status, queue, note, ts, fid}}`` for the confirmed frames
    (``--keep-unsure`` also keeps the unsure ones).
``unsure_frames.json``
    ``[{fid, key, note}, ...]``.
``review_summary.json``
    per queue: frames reviewed, median seconds per frame, and the proposal -> human box counts
    kept / removed / added / moved (greedy IoU matching: IoU >= 0.5 is the same box, below 0.8 it
    counts as moved or resized), frames changed, person-free frames where people were found and
    frames whose people were all removed.
``overrides.json`` (or the ``--overrides`` path)
    corrections for ``openveil apply video --overrides``, for frames cut from videos:
    ``{video: {"add": [[t0_s, t1_s, [x1, y1, x2, y2]], ...], "delete": [[t_s, [x1, y1, x2, y2]], ...],
    "keep": [[t0_s, t1_s], ...]}}`` in source pixels and seconds of the box table. ``keep`` covers a
    giant-rule episode whose big box the reviewer kept: the rule drops no box there.
``reviewed_boxes.json``
    for image-folder reviews: ``{image name: [[x1, y1, x2, y2(, "manual")], ...]}`` with the human
    boxes of every confirmed image (marked ``"manual"``, so the giant rule of ``apply images`` keeps
    them) and the proposal elsewhere, ready for ``openveil apply images --boxes``.

An unsure frame never removes masking: it contributes additions only (overrides) and the union of the
proposal and the human boxes (reviewed boxes). Only the standard library is used.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from typing import Iterable

from .server import load_manifest, read_decisions, write_json_atomic

KEEP_IOU = 0.5          # a human box with IoU >= this to a proposal box is the same box
MOVED_IOU = 0.8         # ... and below this it was moved or resized


def _area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def iou(a, b) -> float:
    """Intersection over union of two boxes [x1, y1, x2, y2, ...]."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0.0) * max(h, 0.0)
    return i / (_area(a) + _area(b) - i + 1e-9)


def match(before: list, after: list, thr: float = KEEP_IOU) -> tuple[list, list, list]:
    """Greedy one-to-one matching in the order of ``before``: each box takes the free ``after`` box of highest IoU.

    Returns (pairs [(i, j, iou)], unmatched before indices, unmatched after indices)."""
    left = list(range(len(after)))
    pairs, removed = [], []
    for i, b in enumerate(before):
        best = max(left, key=lambda j: iou(b, after[j]), default=None)
        if best is not None and iou(b, after[best]) >= thr:
            left.remove(best)
            pairs.append((i, best, iou(b, after[best])))
        else:
            removed.append(i)
    return pairs, removed, left


def compare(before: list, after: list, thr: float = KEEP_IOU) -> tuple[int, int, int, int]:
    """(kept, removed, added, moved) person boxes from the proposal ``before`` to the human ``after``."""
    pairs, removed, added = match(before, after, thr)
    return len(pairs), len(removed), len(added), sum(1 for _, _, v in pairs if v < MOVED_IOU)


def latest_decisions(review_dir: str, manifest: dict | None = None, include_unsure: bool = False) -> dict[str, dict]:
    """Latest decision per frame id; unsure frames are dropped unless ``include_unsure`` (latest wins, then filter)."""
    manifest = manifest or load_manifest(review_dir)
    latest = read_decisions(review_dir, manifest["frames"])
    if include_unsure:
        return latest
    return {fid: d for fid, d in latest.items() if d["status"] == "confirmed"}


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def queue_summary(manifest: dict, latest: dict[str, dict]) -> dict[str, dict]:
    """Per queue, over every frame of the queue that has a decision (whichever queue it was saved from)."""
    frames = manifest["frames"]
    out = {}
    for q in manifest["queues"]:
        s = collections.Counter()
        secs = []
        for e in q["entries"]:
            r = latest.get(e["fid"])
            if r is None:
                continue
            f = frames[e["fid"]]
            k, rem, add, mv = compare(f["persons"], r["persons"])
            s["frames"] += 1
            s["unsure"] += r["status"] == "unsure"
            s["proposal_people"] += len(f["persons"])
            s["human_people"] += len(r["persons"])
            s["kept"] += k
            s["removed"] += rem
            s["added"] += add
            s["moved"] += mv
            s["frames_changed"] += bool(rem or add or mv)
            s["free_to_people"] += (not f["persons"]) and bool(r["persons"])
            s["people_to_none"] += bool(f["persons"]) and not r["persons"]
            secs.append(float(r.get("ms", 0)) / 1000.0)
        keys = ("frames", "unsure", "proposal_people", "human_people", "kept", "removed", "added", "moved",
                "frames_changed", "free_to_people", "people_to_none")
        out[q["id"]] = dict({k: int(s[k]) for k in keys}, total=len(q["entries"]), title=q.get("title", q["id"]),
                            median_s=round(_median(secs), 2))
    return out


def _to_source(box, f: dict) -> list[float]:
    """Frame pixels -> source pixels (frames cut from a video may have been downscaled for review)."""
    sx = float(f.get("src_width") or f["width"]) / f["width"]
    sy = float(f.get("src_height") or f["height"]) / f["height"]
    return [round(box[0] * sx, 1), round(box[1] * sy, 1), round(box[2] * sx, 1), round(box[3] * sy, 1)]


def build_overrides(manifest: dict, latest_all: dict[str, dict]) -> dict[str, dict]:
    """Corrections for the apply step, from frames that carry ``video`` and ``t`` (seconds in the box table).

    The human boxes are compared with what the apply step would mask at that frame (``applied`` when
    the builder stored it, i.e. after the giant rule, else the proposal): an apply box the human
    removed becomes a ``delete`` at the time of the table row it came from (``table_t``, else ``t``);
    a human box with no apply box at IoU >= 0.5, or one moved below IoU 0.8, becomes an ``add`` at
    ``t`` (``[t, t, box]``; the hold of the apply step spreads it). A moved or resized box is not
    deleted: the old box keeps its time span and the new one is added (more mosaic, never less).
    On a giant-rule frame, a human box that matches a box the rule drops (the reviewer kept it)
    also gives a ``keep`` range over the whole episode (``giant.start_s`` to ``giant.end_s``), so
    the rule drops no box anywhere in it. Unsure frames add and keep only."""
    out: dict[str, dict] = {}
    for fid, r in sorted(latest_all.items()):
        f = manifest["frames"][fid]
        video, t = f.get("video"), f.get("t")
        if not isinstance(video, str) or not isinstance(t, (int, float)) or f.get("image"):
            continue
        applied = f.get("applied", f["persons"])
        pairs, removed, added = match(applied, r["persons"])
        adds = [j for j in added] + [j for _, j, v in pairs if v < MOVED_IOU]
        dels = [] if r["status"] == "unsure" else list(removed)
        keep = None
        g = f.get("giant")
        if isinstance(g, dict) and isinstance(g.get("start_s"), (int, float)) and isinstance(g.get("end_s"), (int, float)):
            dropped = [b for b in f["persons"] if not any(iou(b, a) >= 0.999 for a in applied)]
            if any(iou(r["persons"][j], d) >= KEEP_IOU for j in added for d in dropped):
                keep = [round(float(g["start_s"]), 3), round(float(g["end_s"]), 3)]
        if not adds and not dels and keep is None:
            continue
        o = out.setdefault(video, {"add": [], "delete": []})
        tt = f.get("table_t", t)
        tt = t if not isinstance(tt, (int, float)) else tt
        for j in sorted(adds):
            o["add"].append([round(float(t), 3), round(float(t), 3), _to_source(r["persons"][j], f)])
        for i in sorted(dels):
            o["delete"].append([round(float(tt), 3), _to_source(applied[i], f)])
        if keep is not None:
            o.setdefault("keep", []).append(keep)
    for o in out.values():
        o["add"].sort(key=lambda x: (x[0], x[2]))
        o["delete"].sort(key=lambda x: (x[0], x[1]))
        if "keep" in o:
            o["keep"] = sorted({tuple(k) for k in o["keep"]})
            o["keep"] = [list(k) for k in o["keep"]]
    return out


def build_reviewed_boxes(manifest: dict, latest_all: dict[str, dict]) -> dict[str, list]:
    """For image-folder reviews: corrected boxes per image name (frames that carry ``image``).

    Confirmed: the human boxes, marked ``"manual"`` (a box a reviewer confirmed is never dropped
    again by the giant rule of ``apply images``). Unsure: the proposal plus every human box that
    matches no proposal box (never fewer boxes than the proposal; the human boxes marked
    ``"manual"``). Not reviewed: the proposal."""
    out: dict[str, list] = {}
    for fid, f in manifest["frames"].items():
        name = f.get("image")
        if not isinstance(name, str):
            continue
        r = latest_all.get(fid)
        if r is None:
            boxes = [_to_source(b, f) for b in f["persons"]]
        elif r["status"] == "confirmed":
            boxes = [_to_source(b, f) + ["manual"] for b in r["persons"]]
        else:
            _, _, added = match(f["persons"], r["persons"])
            boxes = [_to_source(b, f) for b in f["persons"]] + [_to_source(r["persons"][j], f) + ["manual"] for j in added]
        out[name] = boxes
    return dict(sorted(out.items()))


def export_review(review_dir: str, overrides_path: str | None = None, keep_unsure: bool = False,
                  quiet: bool = False) -> dict:
    """Write the export files (see the module docstring) and return ``{labels, unsure, summary, overrides, reviewed_boxes, paths}``."""
    manifest = load_manifest(review_dir)
    frames = manifest["frames"]
    latest_all = latest_decisions(review_dir, manifest, include_unsure=True)
    labels, unsure = {}, []
    for fid, r in sorted(latest_all.items()):
        f = frames[fid]
        if r["status"] == "unsure":
            unsure.append(dict(fid=fid, key=f["key"], note=r.get("note", "")))
            if not keep_unsure:
                continue
        labels[f["key"]] = dict(kind=f["kind"], persons=r["persons"], status=r["status"], queue=r.get("queue", ""),
                                note=r.get("note", ""), ts=r["ts"], fid=fid)
    summary = queue_summary(manifest, latest_all)
    overrides = build_overrides(manifest, latest_all)
    reviewed = build_reviewed_boxes(manifest, latest_all)
    paths = dict(labels=os.path.join(review_dir, "human_labels.json"), unsure=os.path.join(review_dir, "unsure_frames.json"),
                 summary=os.path.join(review_dir, "review_summary.json"))
    write_json_atomic(paths["labels"], labels, indent=1)
    write_json_atomic(paths["unsure"], unsure, indent=1)
    write_json_atomic(paths["summary"], summary, indent=1)
    if overrides or overrides_path:
        paths["overrides"] = overrides_path or os.path.join(review_dir, "overrides.json")
        write_json_atomic(paths["overrides"], overrides, indent=1)
    if reviewed:
        paths["reviewed_boxes"] = os.path.join(review_dir, "reviewed_boxes.json")
        write_json_atomic(paths["reviewed_boxes"], reviewed, indent=1)
    if not quiet:
        for line in format_summary(latest_all, labels, unsure, summary, keep_unsure, overrides, paths):
            print(line)
    return dict(labels=labels, unsure=unsure, summary=summary, overrides=overrides, reviewed_boxes=reviewed, paths=paths)


def format_summary(latest_all, labels, unsure, summary, keep_unsure, overrides, paths) -> Iterable[str]:
    yield (f"{len(latest_all)} frames reviewed -> {len(labels)} human labels "
           f"({len(unsure)} unsure, {'kept' if keep_unsure else 'left out'})")
    for qid, s in summary.items():
        if not s["frames"]:
            continue
        yield ""
        yield f"{qid}: {s['frames']} of {s['total']} frames reviewed, median {s['median_s']:.1f} s per frame (unsure {s['unsure']})"
        yield (f"  people: proposal {s['proposal_people']} -> human {s['human_people']} (kept {s['kept']}, removed {s['removed']}, "
               f"added {s['added']}, moved/resized {s['moved']})")
        yield (f"  frames changed {s['frames_changed']} ({s['frames_changed'] / s['frames']:.0%}); person-free in the proposal but people "
               f"found {s['free_to_people']}; people in the proposal but none left {s['people_to_none']}")
    if overrides:
        n_add = sum(len(o["add"]) for o in overrides.values())
        n_del = sum(len(o["delete"]) for o in overrides.values())
        n_keep = sum(len(o.get("keep", [])) for o in overrides.values())
        yield ""
        yield (f"overrides for {len(overrides)} video(s): {n_add} additions, {n_del} deletions, {n_keep} kept giant-rule "
               f"episodes -> {paths['overrides']}")
    if "reviewed_boxes" in paths:
        yield f"reviewed boxes for the image folder -> {paths['reviewed_boxes']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openveil review export", description="Export the decisions of a review folder.")
    ap.add_argument("review_dir")
    ap.add_argument("--overrides", default=None, help="where to write the overrides for the apply step (default REVIEW_DIR/overrides.json)")
    ap.add_argument("--keep-unsure", action="store_true", help="keep unsure frames in human_labels.json")
    a = ap.parse_args(argv)
    export_review(a.review_dir, overrides_path=a.overrides, keep_unsure=a.keep_unsure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
