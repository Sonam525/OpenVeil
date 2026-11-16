# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Box tables: the per-video record of every box to mosaic.

A box table is written once by detection and read by review and apply. It is a gzip JSON file
``<name>.json.gz`` with schema ``openveil.boxes/1``::

    {
      "schema": "openveil.boxes/1",
      "video": "clip01",                 # name (no path)
      "kind": "video",                   # "video", or "frames" for an image sequence
      "fps": 25.0,
      "width": 1280, "height": 720,
      "time_base": [1, 90000],           # stream time base, or null for a frames table
      "first_pts": 0,                    # pts of the first decoded frame
      "start_local": null,               # optional ISO camera time of the first frame
      "params": {...},                   # the settings that produced the table
      "chunks": [[t0_s, mode, n_frames, n_detector_calls], ...],
      "frames": [[pts, [[x1, y1, x2, y2, kind(, score)], ...]], ...]
    }

``frames`` lists only frames that carry a box, sorted by pts. Coordinates are source pixels
before the giant rule, the enlargement and the temporal hold of the apply step. ``kind`` is
``det`` (a detection), ``interp`` (interpolated between detections) or ``manual`` (added by a
reviewer); ``score`` is optional. Times are ``(pts - first_pts) * time_base`` seconds; in a
frames table (no time base) ``(pts - first_pts) / fps``.

Tables written before the schema was versioned (no ``schema`` key, no scores, extra ``visit``
and ``source`` fields) load as version 1. ``source`` (a path) is never written unless asked for.

Overrides from a human review (``overrides.json``)::

    {"<video>": {"add": [[t0_s, t1_s, [x1, y1, x2, y2]], ...],
                 "delete": [[t_s, [x1, y1, x2, y2]], ...]}}

are merged into a table with :func:`apply_overrides`.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .geometry import iou

__all__ = [
    "SCHEMA", "KINDS", "TableError", "BoxTable", "seconds", "load_table", "read_table_dict",
    "save_table", "list_tables", "load_overrides", "save_overrides", "overrides_for",
    "apply_overrides",
]

SCHEMA = "openveil.boxes/1"
KINDS = ("det", "interp", "manual")
TABLE_KINDS = ("video", "frames")
_KNOWN = ("schema", "video", "kind", "visit", "source", "fps", "width", "height", "time_base",
          "first_pts", "start_local", "params", "chunks", "frames")

TableLike = Union["BoxTable", Dict[str, Any]]


class TableError(ValueError):
    """A box table or an overrides file is malformed."""


@dataclass
class BoxTable:
    """One box table (see the module docstring for the format).

    ``frames`` and ``chunks`` are plain lists, exactly as stored, so that a table of a long video
    stays cheap to load. :meth:`from_dict` and :meth:`to_dict` do not copy them; use
    :meth:`copy` for an independent table.
    """
    video: str
    width: int
    height: int
    fps: float
    kind: str = "video"
    time_base: Optional[Tuple[int, int]] = None
    first_pts: Optional[int] = 0
    start_local: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    chunks: List[list] = field(default_factory=list)
    frames: List[list] = field(default_factory=list)
    visit: Optional[str] = None
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    # --- conversion ---------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BoxTable":
        """Table from its JSON dict. A dict without ``schema`` is a legacy table (version 1)."""
        if not isinstance(d, Mapping):
            raise TableError(f"a box table must be a JSON object, got {type(d).__name__}")
        schema = d.get("schema", SCHEMA)
        if not (isinstance(schema, str) and schema.split("/")[0] == SCHEMA.split("/")[0]):
            raise TableError(f"not an OpenVeil box table (schema {schema!r})")
        if schema != SCHEMA:
            raise TableError(f"box table schema {schema!r} is not supported by this version (expects {SCHEMA!r})")
        missing = [k for k in ("video", "width", "height", "frames") if k not in d]
        if missing:
            raise TableError(f"box table lacks {', '.join(missing)}")
        tb = d.get("time_base")
        return cls(
            video=d["video"], width=d["width"], height=d["height"], fps=d.get("fps"),
            kind=d.get("kind") or "video",
            time_base=(int(tb[0]), int(tb[1])) if tb else None,
            first_pts=d.get("first_pts", 0), start_local=d.get("start_local"),
            params=d.get("params") or {}, chunks=d.get("chunks") or [], frames=d["frames"],
            visit=d.get("visit"), source=d.get("source"),
            extra={k: v for k, v in d.items() if k not in _KNOWN}, schema=SCHEMA)

    def to_dict(self, include_source: bool = False) -> Dict[str, Any]:
        """JSON dict of the table. ``source`` (a path) is written only with ``include_source``."""
        out: Dict[str, Any] = {"schema": self.schema, "video": self.video, "kind": self.kind}
        if self.visit is not None:
            out["visit"] = self.visit
        if include_source and self.source is not None:
            out["source"] = self.source
        out.update(fps=self.fps, width=self.width, height=self.height,
                   time_base=list(self.time_base) if self.time_base else None,
                   first_pts=self.first_pts, start_local=self.start_local, params=self.params,
                   chunks=self.chunks, frames=self.frames)
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out

    def copy(self) -> "BoxTable":
        """Independent deep copy."""
        return BoxTable.from_dict(json.loads(json.dumps(self.to_dict(include_source=True))))

    # --- time ---------------------------------------------------------------------------------
    def seconds(self, pts: int) -> float:
        """Seconds of ``pts`` from the first frame of the table."""
        return seconds(self, pts)

    def pts_at(self, t: float) -> int:
        """The pts of the frame time ``t`` seconds (rounded to the nearest time-base tick or frame)."""
        first = self.first_pts or 0
        if self.time_base:
            num, den = self.time_base
            return int(first + round(t * den / num))
        return int(first + round(t * float(self.fps)))

    # --- content ------------------------------------------------------------------------------
    @property
    def n_boxes(self) -> int:
        """Number of boxes over all frames."""
        return sum(len(r[1]) for r in self.frames)

    def problems(self) -> List[str]:
        """Every problem found in the table (empty list = valid)."""
        p: List[str] = []
        if self.kind not in TABLE_KINDS:
            p.append(f"kind must be one of {TABLE_KINDS}, got {self.kind!r}")
        for name in ("width", "height"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                p.append(f"{name} must be a positive integer, got {v!r}")
        if self.time_base is not None:
            if len(self.time_base) != 2 or not all(isinstance(x, int) and x > 0 for x in self.time_base):
                p.append(f"time_base must be two positive integers, got {self.time_base!r}")
        if not self.time_base and not (isinstance(self.fps, (int, float)) and self.fps > 0):
            p.append("a table without a time base needs fps > 0")
        if self.first_pts is not None and not isinstance(self.first_pts, int):
            p.append(f"first_pts must be an integer, got {self.first_pts!r}")
        if not isinstance(self.params, dict):
            p.append("params must be an object")
        for i, c in enumerate(self.chunks):
            if not (isinstance(c, (list, tuple)) and len(c) >= 4):
                p.append(f"chunks[{i}] must be [t0_s, mode, n_frames, n_calls]")
        prev = None
        for i, row in enumerate(self.frames):
            if len(p) >= 50:
                p.append("... (more problems not listed)")
                break
            if not (isinstance(row, (list, tuple)) and len(row) >= 2 and isinstance(row[1], list)):
                p.append(f"frames[{i}] must be [pts, [boxes]]")
                continue
            pts = row[0]
            if isinstance(pts, bool) or not isinstance(pts, int):
                p.append(f"frames[{i}]: pts must be an integer, got {pts!r}")
            elif prev is not None and pts <= prev:
                p.append(f"frames[{i}]: pts {pts} is not after {prev} (frames must be sorted, one row per pts)")
            else:
                prev = pts
            for j, b in enumerate(row[1]):
                why = _box_problem(b)
                if why:
                    p.append(f"frames[{i}] box {j}: {why}")
        return p

    def validate(self) -> None:
        """Raise :class:`TableError` listing every problem of the table."""
        p = self.problems()
        if p:
            raise TableError(f"invalid box table {self.video!r}:\n  " + "\n  ".join(p))


def _box_problem(b: Any) -> Optional[str]:
    if not isinstance(b, (list, tuple)) or len(b) not in (5, 6):
        return "must be [x1, y1, x2, y2, kind] or [x1, y1, x2, y2, kind, score]"
    xs = b[:4]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs):
        return f"coordinates must be finite numbers, got {xs!r}"
    if b[2] < b[0] or b[3] < b[1]:
        return f"x2 < x1 or y2 < y1 in {xs!r}"
    if b[4] not in KINDS:
        return f"kind must be one of {KINDS}, got {b[4]!r}"
    if len(b) == 6 and b[5] is not None and not (isinstance(b[5], (int, float)) and 0.0 <= b[5] <= 1.0):
        return f"score must be in [0, 1], got {b[5]!r}"
    return None


# ---------------------------------------------------------------------------------------------
# helpers that accept a BoxTable or a plain dict
# ---------------------------------------------------------------------------------------------
def _get(table: TableLike, key: str, default: Any = None) -> Any:
    if isinstance(table, BoxTable):
        return getattr(table, key, default)
    return table.get(key, default)


def seconds(table: TableLike, pts: int) -> float:
    """Seconds of ``pts`` from the first frame of ``table`` (a :class:`BoxTable` or a dict).

    With a time base: ``(pts - first_pts) * num / den``; in a frames table (no time base, pts =
    image number): ``(pts - first_pts) / fps``. For integer pts the result is the correctly
    rounded float of the exact value.
    """
    tb = _get(table, "time_base")
    first = _get(table, "first_pts") or 0
    if tb:
        return (pts - first) * tb[0] / tb[1]
    return (pts - first) / float(_get(table, "fps"))


# ---------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------
def read_table_dict(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """The raw JSON dict of a table file (``.json.gz`` or plain ``.json``)."""
    p = os.fspath(path)
    with open(p, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: Union[str, os.PathLike], validate: bool = False) -> BoxTable:
    """Load a box table (current or legacy). With ``validate`` the table is checked as well."""
    t = BoxTable.from_dict(read_table_dict(path))
    if validate:
        t.validate()
    return t


def save_table(table: TableLike, path: Union[str, os.PathLike], include_source: bool = False) -> None:
    """Write a table as gzip JSON (plain JSON if ``path`` does not end with ``.gz``).

    The file is written to ``<path>.tmp`` and renamed, so a half-written table never carries
    the final name. The gzip header has no timestamp, so equal tables give equal bytes.
    A dict is converted to the current schema first.
    """
    t = table if isinstance(table, BoxTable) else BoxTable.from_dict(table)
    data = t.to_dict(include_source=include_source)
    p = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(p))
    os.makedirs(folder, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "wb") as raw:
        if p.endswith(".gz"):
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8") as text:
                    json.dump(data, text)
        else:
            raw.write(json.dumps(data).encode("utf-8"))
    os.replace(tmp, p)


def list_tables(directory: Union[str, os.PathLike]) -> List[str]:
    """Sorted paths of the ``*.json.gz`` box tables in ``directory`` (not recursive)."""
    d = os.fspath(directory)
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json.gz"))


# ---------------------------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------------------------
def load_overrides(path: Union[str, os.PathLike]) -> Dict[str, Dict[str, list]]:
    """Read and check an overrides file (``{video: {"add": [...], "delete": [...], "keep": [...]}}``)."""
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TableError("an overrides file must be a JSON object {video: {add, delete}}")
    for video, entry in data.items():
        _check_entry(entry, str(video))
    return data


def save_overrides(overrides: Mapping[str, Mapping[str, list]], path: Union[str, os.PathLike]) -> None:
    """Write an overrides file (checked first)."""
    for video, entry in overrides.items():
        _check_entry(entry, str(video))
    p = os.fspath(path)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=1)
    os.replace(tmp, p)


def overrides_for(overrides: Optional[Mapping[str, Any]], video: str) -> Dict[str, list]:
    """The ``{"add", "delete", "keep"}`` entry of ``video``.

    ``overrides`` may be a whole overrides file (keyed by video name) or already one entry.
    A video without an entry gets empty lists.
    """
    if not overrides:
        return {"add": [], "delete": [], "keep": []}
    if "add" in overrides or "delete" in overrides or "keep" in overrides:
        entry = overrides
    else:
        entry = overrides.get(video) or {}
    _check_entry(entry, video)
    return {"add": list(entry.get("add") or []), "delete": list(entry.get("delete") or []),
            "keep": list(entry.get("keep") or [])}


def _check_box4(b: Any, where: str) -> List[float]:
    if not (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in b)):
        raise TableError(f"{where}: a box must be [x1, y1, x2, y2], got {b!r}")
    if not (b[2] > b[0] and b[3] > b[1]):
        raise TableError(f"{where}: empty box {b!r} (x2 must be > x1 and y2 > y1)")
    return [float(x) for x in b]


def _check_time(t: Any, where: str) -> float:
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        raise TableError(f"{where}: a time must be a number of seconds, got {t!r}")
    return float(t)


def _parse_add(item: Any, where: str) -> Tuple[float, float, List[float]]:
    if isinstance(item, (list, tuple)) and len(item) == 3:
        t0, t1, box = _check_time(item[0], where), _check_time(item[1], where), _check_box4(item[2], where)
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        t0 = t1 = _check_time(item[0], where)
        box = _check_box4(item[1], where)
    else:
        raise TableError(f"{where}: an addition must be [t0_s, t1_s, [x1, y1, x2, y2]], got {item!r}")
    if t1 < t0:
        raise TableError(f"{where}: t1 {t1} is before t0 {t0}")
    return t0, t1, box


def _parse_delete(item: Any, where: str) -> Tuple[float, float, List[float]]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        t = _check_time(item[0], where)
        return t, t, _check_box4(item[1], where)
    if isinstance(item, (list, tuple)) and len(item) == 3:       # a range [t0_s, t1_s, box]
        t0, t1 = _check_time(item[0], where), _check_time(item[1], where)
        if t1 < t0:
            raise TableError(f"{where}: t1 {t1} is before t0 {t0}")
        return t0, t1, _check_box4(item[2], where)
    raise TableError(f"{where}: a deletion must be [t_s, [x1, y1, x2, y2]], got {item!r}")


def _parse_keep(item: Any, where: str) -> Tuple[float, float]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        t0, t1 = _check_time(item[0], where), _check_time(item[1], where)
        if t1 < t0:
            raise TableError(f"{where}: t1 {t1} is before t0 {t0}")
        return t0, t1
    raise TableError(f"{where}: a keep range must be [t0_s, t1_s], got {item!r}")


def _check_entry(entry: Any, video: str) -> None:
    if not isinstance(entry, Mapping):
        raise TableError(f"overrides of {video!r} must be an object with 'add', 'delete' and/or 'keep'")
    unknown = [k for k in entry if k not in ("add", "delete", "keep")]
    if unknown:
        raise TableError(f"overrides of {video!r}: unknown key(s) {unknown}")
    for i, it in enumerate(entry.get("add") or []):
        _parse_add(it, f"{video} add[{i}]")
    for i, it in enumerate(entry.get("delete") or []):
        _parse_delete(it, f"{video} delete[{i}]")
    for i, it in enumerate(entry.get("keep") or []):
        _parse_keep(it, f"{video} keep[{i}]")


def apply_overrides(table: TableLike, overrides: Optional[Mapping[str, Any]], cfg: Any = None, *,
                    step_s: Optional[float] = None, match_iou: Optional[float] = None,
                    delete_window_s: Optional[float] = None) -> Tuple[TableLike, Dict[str, int]]:
    """Merge a reviewer's additions and deletions into a table.

    Returns ``(new_table, stats)`` of the same type as ``table`` (``BoxTable`` or dict); the input
    is not modified. ``overrides`` is a whole overrides file or the entry of this video.

    Deletions run first. ``[t_s, box]`` removes, from every row whose time is within
    ``t_s +- delete_window_s``, the boxes with IoU >= ``match_iou`` against ``box`` (raw table
    coordinates). With the window equal to the temporal hold of the apply step (2 s), the deleted
    box no longer reaches the frame at ``t_s`` from any neighbouring row. ``[t0_s, t1_s, box]``
    deletes over a time range (plus the window on both sides).

    Keep ranges ``[t0_s, t1_s]`` (a reviewer kept a box the giant rule drops) turn every box of
    the rows from ``t0_s`` to ``t1_s`` into a ``manual`` box, which the giant rule never drops.

    Additions ``[t0_s, t1_s, box]`` (or ``[t_s, box]``) put a ``manual`` box into rows every
    ``step_s`` seconds from ``t0_s`` to ``t1_s`` (both included), creating rows where needed.
    Between the rows the temporal hold carries the box, so ``step_s`` must not exceed twice the
    hold. An identical manual box already in a row is not added twice, so applying the same
    overrides again changes nothing. Rows left without boxes are removed.

    ``params['overrides']`` records what was applied.

    Defaults come from ``cfg`` (an ``OpenVeilConfig`` or ``ApplyConfig``; default: the production
    preset): ``override_step_s`` (0.5 s), ``override_match_iou`` (0.5) and
    ``override_delete_window_s`` (``None`` = ``hold_s``, 2 s). Keyword arguments win.
    """
    if cfg is None:
        from .config import ApplyConfig
        a = ApplyConfig()
    else:
        a = getattr(cfg, "apply", cfg)
    step_s = a.override_step_s if step_s is None else step_s
    match_iou = a.override_match_iou if match_iou is None else match_iou
    if delete_window_s is None:
        delete_window_s = a.hold_s if a.override_delete_window_s is None else a.override_delete_window_s
    if step_s <= 0:
        raise ValueError("step_s must be > 0")
    if delete_window_s < 0:
        raise ValueError("delete_window_s must be >= 0")
    is_obj = isinstance(table, BoxTable)
    d = table.to_dict(include_source=True) if is_obj else table
    video = str(d.get("video", ""))
    entry = overrides_for(overrides, video)
    adds = [_parse_add(it, f"{video} add[{i}]") for i, it in enumerate(entry["add"])]
    dels = [_parse_delete(it, f"{video} delete[{i}]") for i, it in enumerate(entry["delete"])]
    keeps = [_parse_keep(it, f"{video} keep[{i}]") for i, it in enumerate(entry["keep"])]

    rows = {r[0]: [r[0], [list(b) for b in r[1]]] + list(r[2:]) for r in d["frames"]}
    stats = dict(deleted_boxes=0, added_boxes=0, added_rows=0, removed_rows=0, kept_boxes=0)
    if dels:
        times = {pts: seconds(d, pts) for pts in rows}
        for t0, t1, box in dels:
            lo, hi = t0 - delete_window_s - 1e-9, t1 + delete_window_s + 1e-9
            for pts, row in rows.items():
                if lo <= times[pts] <= hi:
                    keep = [b for b in row[1] if iou(b, box) < match_iou]
                    stats["deleted_boxes"] += len(row[1]) - len(keep)
                    row[1] = keep
    if keeps:
        times = {pts: seconds(d, pts) for pts in rows}
        for t0, t1 in keeps:
            for pts, row in rows.items():
                if t0 - 1e-9 <= times[pts] <= t1 + 1e-9:
                    for b in row[1]:
                        if len(b) >= 5 and b[4] == "manual":
                            continue
                        if len(b) >= 5:
                            b[4] = "manual"
                        else:
                            b.append("manual")
                        stats["kept_boxes"] += 1
    tmp = BoxTable(video=video, width=d["width"], height=d["height"], fps=d.get("fps"),
                   time_base=tuple(d["time_base"]) if d.get("time_base") else None, first_pts=d.get("first_pts", 0))
    for t0, t1, box in adds:
        n = int(math.floor((t1 - t0) / step_s + 1e-9))
        ts = [t0 + k * step_s for k in range(n + 1)]
        if t1 - ts[-1] > 1e-9:
            ts.append(t1)
        for t in ts:
            pts = tmp.pts_at(t)
            new_box = box + ["manual"]
            row = rows.get(pts)
            if row is None:
                row = rows[pts] = [pts, []]
                stats["added_rows"] += 1
            if not any(len(b) >= 5 and b[4] == "manual" and list(b[:4]) == box for b in row[1]):
                row[1].append(new_box)
                stats["added_boxes"] += 1
    frames = []
    for pts in sorted(rows):
        if rows[pts][1]:
            frames.append(rows[pts])
        else:
            stats["removed_rows"] += 1
    new = dict(d)
    new["frames"] = frames
    params = dict(d.get("params") or {})
    params["overrides"] = dict(n_add=len(adds), n_delete=len(dels), n_keep=len(keeps), step_s=step_s, match_iou=match_iou,
                               delete_window_s=delete_window_s)
    new["params"] = params
    return (BoxTable.from_dict(new) if is_obj else new), stats
