# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Build a review folder (``manifest.json`` + frame images) for the review page.

Two inputs:

**Box tables + videos** (``openveil review build --tables DIR --videos DIR --out REVIEW_DIR``)
    Reads every box table (``*.json.gz``, schema ``openveil.boxes/1``) and cuts the chosen frames
    out of the matching videos on this computer (PyAV), downscaled to at most 1280 px wide. Queues:

    ``mosaic``  frames with the largest mosaics (share of the image the 20%-enlarged boxes cover
                after the giant rule), largest first, at most one frame per video every ``spacing_s``;
    ``giant``   the worst frame of every episode in which the giant rule fires (least of the dropped
                box still covered by the other boxes first): does the dropped box hide a person?
    ``random``  frames drawn uniformly over the duration of all videos: the only way to find people
                the pipeline missed; any prefix is an unbiased sample;
    ``below``   (only when the tables store scores) frames whose best candidate scores below the
                masking threshold, highest first.

**Image folder + boxes JSON** (``openveil review build --images DIR --boxes BOXES.json --out REVIEW_DIR``)
    One frame per image; the images stay where they are (the manifest refers to them by a path
    relative to the review folder) unless ``copy=True``. Queues ``mosaic``, ``giant``, ``below`` (if
    scores are given) and ``random``.

The green boxes ("persons") are the table's boxes before the giant rule; ``applied`` holds what the
apply step would mask at that frame (after the giant rule) and is what ``review export`` compares the
human boxes with when it writes overrides.

Building into an existing review folder keeps every frame id whose key is unchanged (so saved
decisions stay valid), never reuses an old id, replaces the queues it builds and keeps the other
queues. ``manifest.json`` is written to a temporary file and renamed.

Optional dependencies, imported only when needed: PyAV (``av``) to cut frames from videos, and
Pillow (or OpenCV) to read image sizes and write JPEGs.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Iterable

from .server import MANIFEST, REAL_DATA_BANNER, load_manifest, validate_manifest, write_json_atomic

# production values (preset 'production')
PAD = 0.20                 # the apply step enlarges every box by this share of its width / height on each side
MIN_SHARE = 0.55           # giant rule: a box covering >= this share of the image ...
MIN_INNER = 2              # ... with >= this many smaller boxes ...
INNER_FRAC = 0.8           # ... at least this much inside it is dropped
EPISODE_GAP_S = 1.0        # frames where the giant rule fires belong to one episode when <= this far apart
MASK_THR = 0.10            # a box is masked when its score is >= this (used when a table does not say)

MAX_WIDTH = 1280           # review frames are downscaled to at most this width
JPEG_QUALITY = 90
PER_QUEUE = 100
SPACING_S = 10.0           # at most one frame per video in this many seconds, per queue
MATCH_S = 0.5              # random frames: the proposal is the nearest table row at most this far away
IR_COLOURFULNESS = 0.05    # mean |R-G| + |G-B| below this on a 320x180 thumbnail = a grey (infrared) frame

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".ts", ".mts", ".webm")

KINDS = {
    "mosaic": dict(label="large-mosaic frame",
                   hint="Green = the boxes of the table. Is every green box a person (a hand or a boot counts)? Remove a box that is not, "
                        "fix a loose box, draw a missed person; press Enter when it is right."),
    "giant": dict(label="giant-rule frame", warn=True,
                  hint="The giant rule would drop the biggest box here and keep the boxes inside it. If the big box is itself a person, or "
                       "dropping it would leave part of a person without a box, keep it; otherwise delete it."),
    "random": dict(label="random audit frame",
                   hint="A random frame. Look for any person, also small or at the edge, and make sure each has one box and nothing else does."),
    "below": dict(label="below-threshold frame",
                  hint="The best candidate here scores below the masking threshold. Click a dashed box that is a person to adopt it."),
    "image": dict(label="image",
                  hint="Green = the boxes given for this image. Fix them so that every visible person has one box and nothing else does."),
}
PREFIX = {"mosaic": "M", "giant": "G", "random": "A", "below": "B", "image": "I"}


# ---------------------------------------------------------------------------------------------------------------------
# geometry and the giant rule (same maths as openveil.rules / the production box rules)
# ---------------------------------------------------------------------------------------------------------------------
def _area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def _clip(b, w, h) -> list[float]:
    return [min(max(b[0], 0.0), w), min(max(b[1], 0.0), h), min(max(b[2], 0.0), w), min(max(b[3], 0.0), h)]


def giant_split(boxes: list, w: float, h: float, min_share: float = MIN_SHARE, min_inner: int = MIN_INNER,
                inner_frac: float = INNER_FRAC) -> tuple[list, list]:
    """(kept, dropped): a box covering >= min_share of the image with >= min_inner smaller boxes
    (>= inner_frac of their area) inside it is dropped. Every box is judged against the original list.

    The rule of the apply step (:func:`openveil.rules.giant_rule`, which never drops a reviewer's
    ``manual`` box), so that the review shows exactly what will be masked."""
    from ..rules import DEFAULT_KEEP_KINDS, giant_rule

    return giant_rule(boxes, w, h, min_share, min_inner, inner_frac, DEFAULT_KEEP_KINDS)


def enlarge(b, w: float, h: float, pad: float = PAD) -> list[float]:
    bw, bh = b[2] - b[0], b[3] - b[1]
    return [max(b[0] - pad * bw, 0.0), max(b[1] - pad * bh, 0.0), min(b[2] + pad * bw, w), min(b[3] + pad * bh, h)]


def union_area(boxes: list) -> float:
    """Exact area of the union of boxes (coordinate compression)."""
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        return _area(boxes[0])
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    total = 0.0
    for xa, xb in zip(xs, xs[1:]):
        ivs = sorted((b[1], b[3]) for b in boxes if b[0] <= xa and b[2] >= xb)
        length, cur0, cur1 = 0.0, None, None
        for y0, y1 in ivs:
            if cur1 is None or y0 > cur1:
                if cur1 is not None:
                    length += cur1 - cur0
                cur0, cur1 = y0, y1
            else:
                cur1 = max(cur1, y1)
        if cur1 is not None:
            length += cur1 - cur0
        total += (xb - xa) * length
    return total


def mosaic_share(boxes: list, w: float, h: float, pad: float = PAD) -> float:
    """Share of the image the enlarged boxes cover."""
    return union_area([enlarge(b, w, h, pad) for b in boxes]) / float(w * h)


def covered_share(box, others: list, w: float, h: float, pad: float = PAD, grid: int = 24) -> float:
    """Share of ``box`` inside the union of the enlarged ``others`` (sampled on a grid x grid raster)."""
    en = [enlarge(o, w, h, pad) for o in others]
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


# ---------------------------------------------------------------------------------------------------------------------
# box tables
# ---------------------------------------------------------------------------------------------------------------------
def read_table(path: str) -> dict:
    """Read a box table (``.json.gz`` or ``.json``). Production tables without ``schema`` are accepted as v1."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        table = json.load(fh)
    for k in ("video", "width", "height", "frames"):
        if k not in table:
            raise ValueError(f"{os.path.basename(path)}: not a box table (no '{k}')")
    return table


def seconds(table: dict, pts: float) -> float:
    """Seconds from the first frame of the table."""
    tb = table.get("time_base")
    if tb:
        return (pts - (table.get("first_pts") or 0)) * tb[0] / tb[1]
    return (pts - (table.get("first_pts") or 0)) / float(table["fps"])


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def box_score(b) -> float | None:
    """The score of a stored box: ``[x1, y1, x2, y2, kind, score]`` or ``[x1, y1, x2, y2, score]``; None when not stored."""
    for v in b[4:6]:
        if _num(v):
            return float(v)
    return None


def table_rows(table: dict) -> tuple[dict, dict]:
    """(boxes by pts, sub-threshold candidates by pts) of a table.

    Candidates are optional: the third element of a frame row, or a top-level
    ``candidates: [[pts, [[x1, y1, x2, y2, score], ...]], ...]``."""
    rows, cands = {}, collections.defaultdict(list)
    for fr in table["frames"]:
        pts, boxes = fr[0], [list(b) for b in fr[1] if _num(b[0]) and _num(b[3])]
        if boxes:
            rows[pts] = boxes
        if len(fr) > 2 and isinstance(fr[2], list) and all(isinstance(c, list) for c in fr[2]):
            cands[pts].extend(list(c) for c in fr[2])
    for item in table.get("candidates") or []:
        cands[item[0]].extend(list(c) for c in item[1])
    return rows, dict(cands)


def table_duration(table: dict, rows: dict) -> float:
    """Duration in seconds: from the chunk list, else the last row."""
    fps = float(table.get("fps") or 25.0)
    end = 0.0
    for ch in table.get("chunks") or []:
        try:
            end = max(end, float(ch[0]) + float(ch[2]) / fps)
        except (TypeError, ValueError, IndexError):
            continue
    if end <= 0 and rows:
        end = max(seconds(table, p) for p in rows) + 1.0 / fps
    return end


def firing_episodes(table: dict, rows: dict, min_share: float = MIN_SHARE, min_inner: int = MIN_INNER,
                    inner_frac: float = INNER_FRAC, gap_s: float = EPISODE_GAP_S) -> list[dict]:
    """Episodes in which the giant rule fires (frames <= gap_s apart), each with the frame to look at:
    the one where the dropped box is least covered by the enlarged kept boxes."""
    w, h = table["width"], table["height"]
    episodes: list[list[dict]] = []
    for pts in sorted(rows, key=lambda p: seconds(table, p)):
        boxes = rows[pts]
        kept, dropped = giant_split(boxes, w, h, min_share, min_inner, inner_frac)
        if not dropped:
            continue
        t = seconds(table, pts)
        for g in dropped:
            gc = _clip(g, w, h)
            item = dict(t=t, pts=pts, share=_area(gc) / (w * h), cov=covered_share(gc, kept, w, h))
            if episodes and t - episodes[-1][-1]["t"] <= gap_s:
                episodes[-1].append(item)
            else:
                episodes.append([item])
    out = []
    for it in episodes:
        worst = min(it, key=lambda x: x["cov"])
        out.append(dict(start_s=round(it[0]["t"], 3), end_s=round(it[-1]["t"], 3), frames=len({x["pts"] for x in it}),
                        max_share=round(max(x["share"] for x in it), 4), min_covered_after=round(worst["cov"], 3),
                        pts_at_min=worst["pts"], t_at_min_s=round(worst["t"], 3)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------------------------------------------------
def _pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def image_size(path: str) -> tuple[int, int]:
    """(width, height) of an image file as displayed, i.e. after its EXIF orientation (the frame
    in which boxes are drawn in the review page and applied by ``openveil apply images``)."""
    Image = _pil()
    if Image is not None:
        with Image.open(path) as im:
            w, h = im.size
            try:
                orient = im.getexif().get(0x0112, 1)
            except Exception:
                orient = 1
            return (h, w) if orient in (5, 6, 7, 8) else (w, h)
    import cv2
    import numpy as np
    a = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)       # applies the EXIF orientation
    if a is None:
        raise ValueError(f"cannot read image {os.path.basename(path)}")
    return a.shape[1], a.shape[0]


def save_frame(rgb, path: str, max_width: int = MAX_WIDTH, quality: int = JPEG_QUALITY) -> tuple[int, int, bool]:
    """Write an RGB uint8 array as JPEG, downscaled to at most ``max_width``. Returns (width, height, is_grey)."""
    import numpy as np
    h, w = rgb.shape[:2]
    nw, nh = (w, h) if w <= max_width else (max_width, max(1, round(h * max_width / w)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image = _pil()
    if Image is not None:
        im = Image.fromarray(rgb)
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.LANCZOS)
        im.save(path, "JPEG", quality=quality)
        thumb = np.asarray(im.resize((320, 180), Image.BOX), dtype=np.int16)
    else:
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if (nw, nh) != (w, h):
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise OSError(f"could not encode {os.path.basename(path)}")
        buf.tofile(path)
        thumb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA).astype(np.int16)
    col = float((np.abs(thumb[:, :, 0] - thumb[:, :, 1]) + np.abs(thumb[:, :, 1] - thumb[:, :, 2])).mean())
    return nw, nh, col < IR_COLOURFULNESS


def extract_frames(video_path: str, targets: Iterable[int], reseek_s: float = 3.0) -> dict[int, tuple[int, Any]]:
    """Decode the frames at the given pts (stream time base) with PyAV.

    Returns ``{target pts: (decoded pts, RGB array)}``: the first decoded frame whose pts is >= the
    target (the exact frame when the target is a frame's pts). Targets past the end are missing.
    The decoder seeks to the keyframe before a target only when the target is more than ``reseek_s``
    ahead of the current position, so neighbouring targets are decoded in one pass."""
    import av
    todo = sorted(set(int(t) for t in targets))
    out: dict[int, tuple[int, Any]] = {}
    if not todo:
        return out
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        gap = max(1, int(reseek_s / tb))
        i, seeked_for, last_pts = 0, -1, None
        while i < len(todo):
            if seeked_for != i and (last_pts is None or todo[i] - last_pts > gap):
                container.seek(todo[i], stream=stream, backward=True, any_frame=False)
                seeked_for = i                            # at most one seek per target: a long GOP is decoded through
            ended = True
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                last_pts = frame.pts
                arr = None
                while i < len(todo) and frame.pts >= todo[i]:
                    if arr is None:
                        arr = frame.to_ndarray(format="rgb24")
                    out[todo[i]] = (frame.pts, arr)
                    i += 1
                if i >= len(todo) or (seeked_for != i and todo[i] - frame.pts > gap):
                    ended = False
                    break
            if ended:
                break                                     # end of the stream: the remaining targets are missing
    return out


def _index_videos(videos_dir: str) -> dict[str, str]:
    """name / stem -> path of every video file under a folder."""
    idx: dict[str, str] = {}
    for root, _, files in os.walk(videos_dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                p = os.path.join(root, fn)
                idx.setdefault(fn, p)
                idx.setdefault(os.path.splitext(fn)[0], p)
    return idx


# ---------------------------------------------------------------------------------------------------------------------
# manifest assembly
# ---------------------------------------------------------------------------------------------------------------------
class _Ids:
    """Frame ids: keep the id of a known key, never reuse an old id."""

    def __init__(self, old_frames: dict):
        self.by_key = {f["key"]: fid for fid, f in old_frames.items() if isinstance(f, dict) and "key" in f}
        self.next = collections.Counter()
        for fid in old_frames:
            m = re.match(r"^([A-Za-z]+)(\d+)$", fid)
            if m:
                self.next[m.group(1)] = max(self.next[m.group(1)], int(m.group(2)) + 1)

    def get(self, key: str, prefix: str) -> str:
        if key in self.by_key:
            return self.by_key[key]
        fid = f"{prefix}{self.next[prefix]:04d}"
        self.next[prefix] += 1
        self.by_key[key] = fid
        return fid


def _old_manifest(out_dir: str) -> dict | None:
    if not os.path.exists(os.path.join(out_dir, MANIFEST)):
        return None
    m = load_manifest(out_dir)
    if m["meta"].get("demo"):
        raise SystemExit(f"{out_dir} holds a demo manifest: build into another folder")
    return m


def _decided_fids(out_dir: str) -> set:
    """Frame ids that have a line in ``decisions.jsonl`` (whatever the line says)."""
    path = os.path.join(out_dir, "decisions.jsonl")
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and isinstance(d.get("fid"), str):
                out.add(d["fid"])
    return out


def _merge_and_write(out_dir: str, old: dict | None, frames: dict, queues: list, meta: dict) -> dict:
    """Keep the old queues that were not rebuilt (and their frames), add the new ones, write atomically.

    A frame of the old manifest that has a decision is never dropped, even when its queue was
    rebuilt without it: it goes into the queue ``earlier`` (reviewed in an earlier build), so its
    decision still reaches the export."""
    new_ids = {q["id"] for q in queues}
    kept_queues = [q for q in (old or {}).get("queues", []) if q["id"] not in new_ids and q["id"] != "earlier"]
    all_frames = {}
    for q in kept_queues:
        for e in q["entries"]:
            all_frames[e["fid"]] = old["frames"][e["fid"]]
    all_frames.update(frames)
    if old:
        in_queues = {e["fid"] for q in kept_queues + queues for e in q["entries"]}
        decided = _decided_fids(out_dir)
        earlier = sorted(f for f in old["frames"] if f in decided and f not in in_queues)
        for f in earlier:
            all_frames.setdefault(f, old["frames"][f])
        if earlier:
            queues = queues + [dict(id="earlier", title="Reviewed in an earlier build", kind="mixed",
                                    desc="Frames reviewed before this folder was rebuilt that no rebuilt queue drew again. "
                                         "Their decisions still count.",
                                    entries=[dict(fid=f, np=len(all_frames[f]["persons"]), note="Reviewed in an earlier build.")
                                             for f in earlier])]
    old_meta = dict((old or {}).get("meta", {}))
    old_kinds = old_meta.get("kinds", {}) if isinstance(old_meta.get("kinds"), dict) else {}
    meta = dict(old_meta, **meta)
    meta["kinds"] = dict(old_kinds, **meta.get("kinds", {}))
    manifest = dict(meta=meta, frames=all_frames, queues=kept_queues + queues)
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("the built manifest is invalid: " + "; ".join(problems[:5]))
    write_json_atomic(os.path.join(out_dir, MANIFEST), manifest)
    _write_gitignore(out_dir)
    return manifest


def _write_gitignore(out_dir: str) -> None:
    """A review folder holds unmasked frames: a ``.gitignore`` with ``*`` keeps git from adding it."""
    p = os.path.join(out_dir, ".gitignore")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# review folder: unmasked frames and decisions, never commit\n*\n")


def _rel(path: str, start: str) -> str:
    """Path relative to the review folder (posix separators); absolute when on another drive."""
    try:
        return os.path.relpath(path, start).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def _r1(v: float) -> float:
    return round(float(v), 1)


def _scaled(b, sx: float, sy: float) -> list[float]:
    return [_r1(b[0] * sx), _r1(b[1] * sy), _r1(b[2] * sx), _r1(b[3] * sy)]


def _spaced(cands: list, n: int | None, spacing_s: float) -> list:
    """Pick from ``cands`` (dicts with video, t; already in priority order) at most one per video every spacing_s."""
    picked, by_video = [], collections.defaultdict(list)
    for c in cands:
        if n is not None and len(picked) >= n:
            break
        ts = by_video[c["video"]]
        if any(abs(c["t"] - x) < spacing_s for x in ts):
            continue
        ts.append(c["t"])
        picked.append(c)
    return picked


def build_from_tables(tables_dir: str, videos_dir: str, out_dir: str, per_queue: int = PER_QUEUE, seed: int = 7,
                      min_share: float = MIN_SHARE, spacing_s: float = SPACING_S, max_giant: int | None = None,
                      max_width: int = MAX_WIDTH, match_s: float = MATCH_S, title: str | None = None,
                      banner: str | None = None, quiet: bool = False) -> dict:
    """Build (or extend) a review folder from box tables and their videos. Returns the manifest."""
    log = (lambda *a: None) if quiet else (lambda *a: print(*a))
    os.makedirs(out_dir, exist_ok=True)
    old = _old_manifest(out_dir)
    rng = random.Random(seed)
    # the same files as `openveil episodes`: not detect's summary.json or a review's overrides
    paths = sorted(os.path.join(tables_dir, f) for f in os.listdir(tables_dir)
                   if f.endswith((".json.gz", ".json")) and not f.startswith(("summary", "overrides")))
    vids = _index_videos(videos_dir)
    tables, mosaic_c, below_c, giant_c, durations = {}, [], [], [], {}
    have_scores = False
    for p in paths:
        try:
            T = read_table(p)
        except (ValueError, OSError) as e:
            log(f"skipped {os.path.basename(p)}: {e}")
            continue
        name = T["video"]
        if T.get("kind", "video") != "video":
            log(f"skipped {name}: a table of a frame sequence (use --images)")
            continue
        vp = vids.get(name) or vids.get(os.path.splitext(name)[0]) or vids.get(os.path.basename(name))
        if vp is None:
            log(f"skipped {name}: no video of that name under the videos folder")
            continue
        rows, cands = table_rows(T)
        w, h = T["width"], T["height"]
        thr = float((T.get("params") or {}).get("mask_thr", MASK_THR))
        tables[name] = dict(table=T, rows=rows, cands=cands, video_path=vp, thr=thr)
        durations[name] = table_duration(T, rows)
        for pts, boxes in rows.items():
            kept, _ = giant_split(boxes, w, h, min_share)
            share = mosaic_share(kept, w, h)
            if share > 0:
                mosaic_c.append(dict(video=name, pts=pts, t=seconds(T, pts), share=share))
            have_scores = have_scores or any(box_score(b) is not None for b in boxes)
        for pts, cs in cands.items():
            sub = [box_score(c) for c in cs if box_score(c) is not None and box_score(c) < thr]
            if sub:
                have_scores = True
                below_c.append(dict(video=name, pts=pts, t=seconds(T, pts), best=max(sub)))
        for ep in firing_episodes(T, rows, min_share):
            giant_c.append(dict(video=name, pts=ep["pts_at_min"], t=ep["t_at_min_s"], ep=ep))
    if not tables:
        raise SystemExit("no usable box table with a matching video was found")

    # --- choose the frames of each queue
    mosaic_c.sort(key=lambda c: (-c["share"], c["video"], c["t"]))
    below_c.sort(key=lambda c: (-c["best"], c["video"], c["t"]))
    giant_c.sort(key=lambda c: (c["ep"]["min_covered_after"], -c["ep"]["max_share"], c["video"], c["t"]))
    chosen = {"mosaic": _spaced(mosaic_c, per_queue, spacing_s),
              "giant": giant_c if max_giant is None else giant_c[:max_giant],
              "below": _spaced(below_c, per_queue, spacing_s) if have_scores else []}
    names = sorted(durations)
    weights = [max(durations[n], 0.0) for n in names]
    rand = []
    if sum(weights) > 0:
        for _ in range(per_queue):
            name = rng.choices(names, weights=weights)[0]
            t = rng.uniform(0.0, durations[name])
            T = tables[name]["table"]
            tb = T.get("time_base") or [1, int(round(float(T.get("fps") or 25)))]
            pts = int((T.get("first_pts") or 0) + math.floor(t * tb[1] / tb[0]))
            rand.append(dict(video=name, pts=pts, t=t, random=True))
    chosen["random"] = rand

    # --- cut the frames
    wanted = collections.defaultdict(set)
    for items in chosen.values():
        for c in items:
            wanted[c["video"]].add(int(c["pts"]))
    decoded: dict[str, dict] = {}
    for name in sorted(wanted):
        try:
            decoded[name] = extract_frames(tables[name]["video_path"], wanted[name])
        except Exception as e:                            # a broken video must not stop the build
            log(f"could not decode {name}: {e}")
            decoded[name] = {}

    ids = _Ids(old["frames"] if old else {})
    frames_dir = os.path.join(out_dir, "frames")
    frames: dict[str, dict] = {}
    queues = []
    spec = [
        ("mosaic", "mosaic", "Largest mosaics: are the big boxes real people?",
         "Frames whose enlarged boxes cover the largest share of the image, largest first (one per video every "
         f"{spacing_s:g} s). Green = the table's boxes."),
        ("giant", "giant", f"Giant rule: does the dropped box hide a person? (min share {min_share:g})",
         "The worst frame of every episode in which the giant rule fires, least covered first. Green = the table's boxes "
         "before the rule."),
        ("below", "below", "Below the threshold: missed people?",
         "Frames whose best candidate scores below the masking threshold, highest first."),
        ("random", "random", "Random audit",
         "Frames drawn at random over all videos. Look for any person: this finds the people the pipeline missed. "
         "Any prefix is an unbiased sample."),
    ]
    for qid, kind, qtitle, desc in spec:
        entries, seen = [], set()
        for c in chosen[qid]:
            name = c["video"]
            info = tables[name]
            got = decoded.get(name, {}).get(int(c["pts"]))
            if got is None:
                continue
            pts, rgb = got
            key = f"{name}:{pts}"
            if key in seen:
                continue
            seen.add(key)
            T = info["table"]
            t = seconds(T, pts)
            if c.get("random"):                            # the proposal: the nearest table row within match_s
                near = min(info["rows"], key=lambda p: abs(seconds(T, p) - t), default=None)
                row_pts = near if near is not None and abs(seconds(T, near) - t) <= match_s else None
            else:
                row_pts = c["pts"]
            fid = ids.get(key, PREFIX[kind])
            if fid not in frames:
                frames[fid] = _video_frame(fid, kind, key, name, T, info, pts, t, row_pts, rgb, frames_dir, out_dir,
                                           max_width, min_share, c.get("ep"))
            elif c.get("ep") is not None:                  # a frame already chosen for another queue is also a giant-rule frame
                frames[fid]["kind"] = kind
                frames[fid]["giant"] = _giant_info(c["ep"])
            f = frames[fid]
            entries.append(dict(fid=fid, np=len(f["persons"]), note=_note(qid, f, c, min_share, match_s)))
        if entries:
            queues.append(dict(id=qid, title=qtitle, desc=desc, kind=kind, entries=entries))
    meta = dict(title=title or "OpenVeil review", banner=banner or REAL_DATA_BANNER, demo=False,
                kinds={k: KINDS[k] for k in ("mosaic", "giant", "random", "below")},
                params=dict(pad=PAD, min_share=min_share, min_inner=MIN_INNER, inner_frac=INNER_FRAC, max_width=max_width,
                            per_queue=per_queue, spacing_s=spacing_s, match_s=match_s, seed=seed),
                source=dict(type="tables", tables=len(tables)))
    manifest = _merge_and_write(out_dir, old, frames, queues, meta)
    log(f"review folder {out_dir}: {len(manifest['frames'])} frames; queues "
        + ", ".join(f"{q['id']} {len(q['entries'])}" for q in manifest["queues"]))
    return manifest


def _clock(T: dict, t: float) -> str:
    start = T.get("start_local")
    if isinstance(start, str):
        try:
            return f"{datetime.fromisoformat(start) + timedelta(seconds=t):%Y-%m-%d %H:%M:%S} (video t = {t:.1f} s)"
        except ValueError:
            pass
    return f"video t = {t:.1f} s"


def _video_frame(fid, kind, key, name, T, info, pts, t, row_pts, rgb, frames_dir, out_dir, max_width, min_share, ep):
    W, H = T["width"], T["height"]
    path = os.path.join(frames_dir, f"{fid}.jpg")
    w, h, grey = save_frame(rgb, path, max_width)
    sx, sy = w / float(W), h / float(H)
    boxes = info["rows"].get(row_pts, []) if row_pts is not None else []
    kept, _ = giant_split(boxes, W, H, min_share)
    scored = [b for b in boxes if box_score(b) is not None] + list(info["cands"].get(row_pts, []) if row_pts is not None else [])
    cands = sorted((_scaled(b, sx, sy) + [round(box_score(b), 4), -1] for b in scored if box_score(b) is not None),
                   key=lambda c: -c[4])
    rec = dict(id=fid, kind=kind, key=key, video=name, clock=_clock(T, t), ir=grey, width=w, height=h,
               src_width=W, src_height=H, path=_rel(path, out_dir), source="table",
               persons=[_scaled(b, sx, sy) for b in boxes], applied=[_scaled(b, sx, sy) for b in kept], cands=cands,
               pts=pts, t=round(t, 3), table_t=None if row_pts is None else round(seconds(T, row_pts), 3))
    if isinstance(T.get("visit"), str):
        rec["visit"] = T["visit"]
    if ep is not None:
        rec["giant"] = _giant_info(ep)
    return rec


def _giant_info(ep: dict) -> dict:
    return dict(share=ep["max_share"], cov_after=ep["min_covered_after"], ep_frames=ep["frames"],
                start_s=ep["start_s"], end_s=ep["end_s"])


def _note(qid: str, f: dict, c: dict, min_share: float, match_s: float) -> str:
    n = len(f["persons"])
    if qid == "mosaic":
        return (f"The table's boxes are green ({n}). With the {PAD:.0%} enlargement the mosaic would cover {c['share']:.0%} of this "
                "image. Is every green box a person (a hand or a boot counts)? Remove a box that is not, fix a loose box, draw a "
                "missed person; press Enter when it is right.")
    if qid == "giant":
        ep = c["ep"]
        return (f"The giant rule would drop the biggest box here ({ep['max_share']:.0%} of the image; after dropping it "
                f"{ep['min_covered_after']:.0%} of it is still covered by the other boxes), the boxes inside it stay. If the big box is "
                "itself a person, or dropping it would leave part of a person without a box, KEEP it; otherwise delete it (and any box "
                "that is not a person), fix a loose box, draw a missed person.")
    if qid == "below":
        return (f"The best candidate here scores {c['best']:.2f}, below the masking threshold. Click a dashed box that is a person to "
                "adopt it; press Enter when every person has a box.")
    if f.get("table_t") is None:
        return (f"Random frame. No table row within {match_s:g} s, so nothing would be masked here. Look for any person "
                "(also small or at the edge) and draw a box around each.")
    return (f"Random frame. Green = the boxes of the nearest table row ({abs(f['t'] - f['table_t']):.2f} s away). Look for any "
            "person (also small or at the edge); every person needs one box and nothing else.")


# ---------------------------------------------------------------------------------------------------------------------
# image folder + boxes JSON
# ---------------------------------------------------------------------------------------------------------------------
def _parse_box(b) -> tuple[list[float], float | None] | None:
    """[x1, y1, x2, y2(, kind)(, score)] or {"box": [...], "score": s} -> (box, score or None); None if malformed."""
    from ..apply.images import parse_box
    p = parse_box(b)
    return None if p is None else (p[0], p[1])


def read_boxes_json(path: str) -> dict[str, dict]:
    """Read a boxes file for an image folder: ``{name: {"boxes": [(box, score, kind)], "width"?, "height"?}}``.

    The same reader as ``openveil apply images`` (:func:`openveil.apply.images.read_boxes_file`):
    ``{name: [box, ...]}``, ``{name: {"boxes": [...], "width": w, "height": h}}``,
    ``{"images": <one of these>}`` and ``[{"image" | "name" | "file": name, "boxes": [...]}, ...]``.
    A box is ``[x1, y1, x2, y2]`` with optional kind and score, or ``{"box": [...], "score": s}``.
    A malformed box raises ``ValueError``."""
    from ..apply.images import read_boxes_file
    return read_boxes_file(path)


def _list_images(images_dir: str) -> list[str]:
    names = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                names.append(os.path.relpath(os.path.join(root, fn), images_dir).replace("\\", "/"))
    return sorted(names)


def build_from_images(images_dir: str, boxes_json: str, out_dir: str, per_queue: int = PER_QUEUE, accept: float = MASK_THR,
                      seed: int = 7, min_share: float = MIN_SHARE, copy: bool = False, title: str | None = None,
                      banner: str | None = None, quiet: bool = False) -> dict:
    """Build (or extend) a review folder from an image folder and a boxes file. Returns the manifest.

    Boxes without a score, or with a score >= ``accept``, are the proposal (green); every scored box is
    also a candidate (dashed). Images of the folder missing from the boxes file have no boxes."""
    import shutil
    log = (lambda *a: None) if quiet else (lambda *a: print(*a))
    os.makedirs(out_dir, exist_ok=True)
    old = _old_manifest(out_dir)
    rng = random.Random(seed)
    from ..apply.images import resolve_names
    raw = read_boxes_json(boxes_json)
    listed = _list_images(images_dir)
    # the same name matching as `openveil apply images` (leading ./, letter case, leading folders)
    found, missing = resolve_names(list(raw), listed + [n for n in raw if n not in listed
                                                         and os.path.isfile(os.path.join(images_dir, n))])
    given: dict[str, dict] = {}
    for key, rec in raw.items():
        if key in found:
            g = given.setdefault(found[key], dict(boxes=[], width=rec.get("width"), height=rec.get("height")))
            g["boxes"].extend(rec["boxes"])
    names = sorted(set(listed) | set(given))
    with_boxes = [n for n in missing if raw[n]["boxes"]]
    if with_boxes:
        raise ValueError(f"{len(with_boxes)} image(s) of the boxes file have boxes but match no image of the folder "
                         f"(or more than one), e.g. {with_boxes[0]!r}")
    if missing:
        log(f"{len(missing)} image(s) of the boxes file without boxes match no image of the folder (e.g. {missing[0]})")
    if not names:
        raise SystemExit("no image found")
    ids = _Ids(old["frames"] if old else {})
    frames, info = {}, {}
    for name in names:
        src = os.path.join(images_dir, name)
        g = given.get(name, dict(boxes=[], width=None, height=None))
        if isinstance(g.get("width"), int) and isinstance(g.get("height"), int) and g["width"] > 0 and g["height"] > 0:
            w, h = g["width"], g["height"]
        else:
            w, h = image_size(src)
        key = f"image:{name}"
        fid = ids.get(key, PREFIX["image"])
        if copy:
            dst = os.path.join(out_dir, "frames", fid + os.path.splitext(name)[1].lower())
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            path = _rel(dst, out_dir)
        else:
            path = _rel(src, out_dir)
        persons = [[_r1(v) for v in b] for b, s, k in g["boxes"] if s is None or s >= accept or k == "manual"]
        cands = sorted(([_r1(v) for v in b] + [round(s, 4), -1] for b, s, _k in g["boxes"] if s is not None), key=lambda c: -c[4])
        kept, dropped = giant_split(persons, w, h, min_share)
        frames[fid] = dict(id=fid, kind="image", key=key, image=name, width=int(w), height=int(h), path=path,
                           source="boxes file", persons=persons, applied=kept, cands=cands)
        sub = [c[4] for c in cands if c[4] < accept]
        info[fid] = dict(share=mosaic_share(kept, w, h), dropped=dropped, kept=kept, best_below=max(sub, default=None), w=w, h=h)
    fids = sorted(frames, key=lambda f: frames[f]["image"])
    queues = []
    mosaic = sorted((f for f in fids if info[f]["share"] > 0), key=lambda f: (-info[f]["share"], frames[f]["image"]))[:per_queue]
    if mosaic:
        queues.append(dict(id="mosaic", title="Largest mosaics: are the big boxes real people?", kind="image",
                           desc="Images whose enlarged boxes cover the largest share of the image, largest first.",
                           entries=[dict(fid=f, np=len(frames[f]["persons"]),
                                         note=f"With the {PAD:.0%} enlargement the mosaic would cover {info[f]['share']:.0%} of this image. "
                                              "Is every green box a person? Remove a box that is not, fix a loose box, draw a missed person.")
                                    for f in mosaic]))
    giant = []
    for f in fids:
        i = info[f]
        if i["dropped"]:
            cov = min(covered_share(_clip(d, i["w"], i["h"]), i["kept"], i["w"], i["h"]) for d in i["dropped"])
            share = max(_area(_clip(d, i["w"], i["h"])) for d in i["dropped"]) / float(i["w"] * i["h"])
            frames[f]["giant"] = dict(share=round(share, 4), cov_after=round(cov, 3), ep_frames=1)
            giant.append((cov, -share, frames[f]["image"], f))
    if giant:
        queues.append(dict(id="giant", title=f"Giant rule: does the dropped box hide a person? (min share {min_share:g})", kind="image",
                           desc="Images where the giant rule drops a box, least covered first.",
                           entries=[dict(fid=f, np=len(frames[f]["persons"]),
                                         note=f"The giant rule would drop a box of {-s:.0%} of the image ({c:.0%} of it stays covered). "
                                              "Keep it if it is a person; otherwise delete it.")
                                    for c, s, _, f in sorted(giant)]))
    below = sorted((f for f in fids if info[f]["best_below"] is not None), key=lambda f: (-info[f]["best_below"], frames[f]["image"]))[:per_queue]
    if below:
        queues.append(dict(id="below", title="Below the threshold: missed people?", kind="image",
                           desc="Images whose best candidate scores below the threshold, highest first.",
                           entries=[dict(fid=f, np=len(frames[f]["persons"]),
                                         note=f"The best candidate below the threshold scores {info[f]['best_below']:.2f}. "
                                              "Click a dashed box that is a person to adopt it.")
                                    for f in below]))
    shuffled = fids[:]
    rng.shuffle(shuffled)
    queues.append(dict(id="random", title="Random audit", kind="image",
                       desc="Images in random order: any prefix is an unbiased sample. Look for any person.",
                       entries=[dict(fid=f, np=len(frames[f]["persons"]), note="") for f in shuffled[:per_queue]]))
    meta = dict(title=title or "OpenVeil review", banner=banner or REAL_DATA_BANNER, demo=False,
                kinds={"image": KINDS["image"]},
                params=dict(pad=PAD, min_share=min_share, min_inner=MIN_INNER, inner_frac=INNER_FRAC, accept=accept,
                            per_queue=per_queue, seed=seed),
                source=dict(type="images", images=len(fids), copied=bool(copy)))
    manifest = _merge_and_write(out_dir, old, frames, queues, meta)
    log(f"review folder {out_dir}: {len(manifest['frames'])} frames; queues "
        + ", ".join(f"{q['id']} {len(q['entries'])}" for q in manifest["queues"]))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openveil review build", description="Build a review folder for the review page.")
    ap.add_argument("--out", required=True, help="the review folder (created; an existing one is extended)")
    ap.add_argument("--tables", help="folder of box tables (*.json.gz)")
    ap.add_argument("--videos", help="folder with the source videos of the tables")
    ap.add_argument("--images", help="image folder")
    ap.add_argument("--boxes", help="boxes JSON for the image folder")
    ap.add_argument("--per-queue", type=int, default=PER_QUEUE, help=f"frames per queue (default {PER_QUEUE}; the giant queue holds every episode)")
    ap.add_argument("--min-share", type=float, default=MIN_SHARE, help=f"giant rule share (default {MIN_SHARE})")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--accept", type=float, default=MASK_THR, help="images: scored boxes >= this are the proposal")
    ap.add_argument("--copy", action="store_true", help="images: copy the images into the review folder")
    ap.add_argument("--max-width", type=int, default=MAX_WIDTH, help="videos: downscale frames to at most this width")
    ap.add_argument("--title", default=None)
    ap.add_argument("--banner", default=None, help=f"banner text (default '{REAL_DATA_BANNER}')")
    a = ap.parse_args(argv)
    if a.tables and a.videos and not (a.images or a.boxes):
        build_from_tables(a.tables, a.videos, a.out, per_queue=a.per_queue, seed=a.seed, min_share=a.min_share,
                          max_width=a.max_width, title=a.title, banner=a.banner)
    elif a.images and a.boxes and not (a.tables or a.videos):
        build_from_images(a.images, a.boxes, a.out, per_queue=a.per_queue, accept=a.accept, seed=a.seed,
                          min_share=a.min_share, copy=a.copy, title=a.title, banner=a.banner)
    else:
        ap.error("give either --tables and --videos, or --images and --boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
