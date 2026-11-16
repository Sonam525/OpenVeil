# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Mosaic the people in an image folder or in a tar archive of JPEG/PNG images.

Only the images that get a mosaic are decoded and written again (JPEG at
``images.jpeg_quality``, 95 by default; PNG losslessly); every other file or tar member is
copied byte for byte, and a tar keeps its member order and headers.

Where the boxes come from
-------------------------
* A **boxes JSON** in pixels of the upright image (:func:`read_boxes_file`, the same reader as
  ``openveil review build --images``). Layouts: ``{name: [box, ...]}``, ``{name: {"boxes":
  [...], "width": w, "height": h}}``, ``{"images": <one of these>}`` or ``[{"image": name,
  "boxes": [...]}, ...]``. A box is ``[x1, y1, x2, y2]``, optionally followed by a kind
  (``"det"``, ``"manual"``) and/or a score, or ``{"box": [...], "score": s}``. A scored box
  below ``accept`` (0.10) is a candidate, not a person, and is not masked. These boxes are
  treated as the detections of one video frame: the giant rule and the enlargement of the
  apply step (``apply.min_share``, ``apply.pad``, ...) are applied before the mosaic, except
  that a ``"manual"`` box (one a reviewer confirmed) is never dropped by the giant rule. Use
  :func:`load_boxes_json`.
* Frames **extracted from videos** that already have box tables (Python API only): a CSV maps
  every image to its video and time, and the boxes come from that video's
  :class:`~openveil.apply.plan.MaskPlan` (giant rule, enlargement and temporal hold, exactly as
  in the masked video). Use :func:`load_image_map` and :func:`boxes_from_map`.

Image names are member names inside a tar, or paths relative to the folder (with ``/``). A
name that does not match exactly is matched, if the match is unique, ignoring a leading
``./``, letter case and leading folders (``frames/cam1/a.jpg`` matches the member
``cam1/a.jpg``, and ``a.jpg`` matches ``cam1/a.jpg``). Boxes for the same image under two
names are merged. A name that matches no image (or more than one) makes :func:`mask_images`
fail unless ``allow_absent`` is set.

Masked images are written without their metadata (EXIF and similar), so the camera, time or
location stored in a masked image do not travel with it. A JPEG with an EXIF orientation is
decoded upright (the box coordinates must refer to the upright image) and written upright.

Needs numpy and either OpenCV (``cv2``, preferred) or Pillow.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "IMAGE_EXTS", "ACCEPT", "read_boxes_file", "resolve_names", "load_boxes_json", "load_image_map", "boxes_from_map", "image_rules",
    "mask_image_bytes", "mask_jpeg", "rewrite_tar", "check_tar", "mask_folder", "check_folder",
    "mask_images",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
ACCEPT = 0.10              # a scored box is a person when its score is >= this (the verifier threshold)
_JPEG_EXTS = (".jpg", ".jpeg")

BoxList = List[List[float]]
Plan = Mapping[str, Sequence[Sequence[float]]]


def _cfg(cfg: Any):
    from ..config import load_config
    return load_config(cfg)


def _block(height: int, cfg: Any) -> int:
    from .mosaic import block_size
    return block_size(height, cfg.apply.block_divisor, cfg.apply.block_min)


# ---------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------
def _num(v: Any) -> bool:
    import math
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_box(b: Any) -> Optional[Tuple[List[float], Optional[float], Optional[str]]]:
    """``[x1, y1, x2, y2(, kind)(, score)]`` or ``{"box": [...], "score": s, "kind": k}`` ->
    ``(box, score or None, kind or None)`` with the corners put in order; ``None`` if malformed."""
    score, kind = None, None
    if isinstance(b, dict):
        score = b.get("score")
        kind = b.get("kind")
        b = b.get("box") or b.get("bbox") or [b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")]
        score = float(score) if _num(score) else None
    if not (isinstance(b, (list, tuple)) and len(b) >= 4 and all(_num(v) for v in b[:4])):
        return None
    for v in list(b[4:6]):
        if isinstance(v, str) and kind is None:
            kind = v
        elif _num(v) and score is None:
            score = float(v)
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], score, (kind if isinstance(kind, str) else None)


def read_boxes_file(path: Union[str, os.PathLike]) -> Dict[str, Dict[str, Any]]:
    """Read a boxes file for images: ``{name: {"boxes": [(box, score, kind)], "width", "height"}}``.

    See the module docstring for the layouts. Raises ``ValueError`` on a file that holds no
    images in a known layout, a malformed box or a box list that is not a list."""
    with open(os.fspath(path), "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("images"), (dict, list)):
        data = data["images"]
    items: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        items = [(str(k), v) for k, v in data.items()]
    elif isinstance(data, list):
        for e in data:
            name = (e.get("image") or e.get("name") or e.get("file")) if isinstance(e, dict) else None
            if not isinstance(name, str):
                raise ValueError(f"a list entry of the boxes file needs 'image' (or 'name', 'file'): {e!r}"[:200])
            items.append((name, e))
    else:
        raise ValueError("a boxes file must hold an object {image_name: [[x1, y1, x2, y2], ...]} or a list")
    out: Dict[str, Dict[str, Any]] = {}
    for name, v in items:
        rec = v if isinstance(v, dict) else {"boxes": v}
        raw = rec.get("boxes") if rec.get("boxes") is not None else []
        if not isinstance(raw, list):
            raise ValueError(f"{name}: expected a list of boxes")
        boxes = []
        for b in raw:
            pb = parse_box(b)
            if pb is None:
                raise ValueError(f"{name}: a box must start with four numbers x1, y1, x2, y2, got {b!r}"[:300])
            boxes.append(pb)
        key = name.replace("\\", "/")
        if key in out:
            out[key]["boxes"].extend(boxes)
        else:
            out[key] = dict(boxes=boxes, width=rec.get("width"), height=rec.get("height"))
    return out


def load_boxes_json(path: Union[str, os.PathLike], accept: float = ACCEPT) -> Dict[str, BoxList]:
    """The boxes to mask per image: ``{image_name: [[x1, y1, x2, y2(, "manual")], ...]}``.

    Reads any layout of :func:`read_boxes_file`. Boxes without a score, with a score >=
    ``accept`` or of kind ``manual`` are kept; a ``manual`` box keeps its kind (the giant rule
    never drops it). Images with an empty list are kept (no mosaic). Raises ``ValueError`` on a
    malformed file."""
    out: Dict[str, BoxList] = {}
    for name, rec in read_boxes_file(path).items():
        bl = []
        for box, score, kind in rec["boxes"]:
            if kind == "manual":
                bl.append(list(box) + ["manual"])
            elif score is None or score >= accept:
                bl.append(list(box))
        out[name] = bl
    return out


def load_image_map(path: Union[str, os.PathLike]) -> Dict[str, Dict[str, Any]]:
    """Read a CSV that maps images to video times.

    Columns: ``image`` (name), ``video`` (the video's table name) and either ``t_s`` (seconds
    on the table clock, i.e. from the video's first frame) or ``pts`` (the frame's pts in the
    video's time base). Returns ``{image: {"video": ..., "t_s": float|None, "pts": int|None}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with open(os.fspath(path), "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = set(rd.fieldnames or [])
        if not {"image", "video"} <= cols or not ({"t_s", "pts"} & cols):
            raise ValueError("the image map CSV needs the columns image, video and t_s (or pts)")
        for i, row in enumerate(rd, start=2):
            img, vid = (row.get("image") or "").strip(), (row.get("video") or "").strip()
            if not img or not vid:
                raise ValueError(f"line {i}: image and video must not be empty")
            ts, pts = (row.get("t_s") or "").strip(), (row.get("pts") or "").strip()
            if not ts and not pts:
                raise ValueError(f"line {i}: give t_s or pts")
            out[img] = dict(video=vid, t_s=float(ts) if ts else None, pts=int(pts) if pts else None)
    return out


def _table_for(tables: Any, video: str):
    from ..table import load_table

    if isinstance(tables, (str, os.PathLike)):
        d = os.fspath(tables)
        stem = os.path.splitext(video)[0]
        for name in (video, stem):
            p = os.path.join(d, name + ".json.gz")
            if os.path.exists(p):
                return load_table(p)
        raise KeyError(f"no box table for video {video!r} in the tables folder")
    if video in tables:
        t = tables[video]
    elif os.path.splitext(video)[0] in tables:
        t = tables[os.path.splitext(video)[0]]
    else:
        raise KeyError(f"no box table for video {video!r}")
    return load_table(t) if isinstance(t, (str, os.PathLike)) else t


def boxes_from_map(mapping: Mapping[str, Mapping[str, Any]], tables: Any, cfg: Any = None) -> Dict[str, BoxList]:
    """Mosaic boxes of every mapped image, from its video's :class:`MaskPlan`.

    ``tables``: ``{video: BoxTable | dict | path}`` or a folder of ``<video>.json.gz`` tables.
    Images without a box are left out. The returned boxes are final (already enlarged): pass
    them on with ``rules=False``.
    """
    from ..table import seconds
    from .plan import MaskPlan

    cfg = _cfg(cfg)
    plans: Dict[str, Tuple[Any, Any]] = {}
    out: Dict[str, BoxList] = {}
    for img, m in mapping.items():
        v = m["video"]
        if v not in plans:
            t = _table_for(tables, v)
            plans[v] = (t, MaskPlan(t, cfg))
        t, plan = plans[v]
        ts = m.get("t_s")
        if ts is None:
            ts = seconds(t, int(m["pts"]))
        b = plan.boxes_at(float(ts))
        if b:
            out[img] = [list(map(float, x[:4])) for x in b]
    return out


def image_rules(boxes: Sequence[Sequence[float]], width: int, height: int, cfg: Any = None) -> BoxList:
    """The apply-step rules for the raw boxes of one image: giant rule, then enlargement
    (same values as for a video frame; there is no temporal hold for a single image). A box
    whose fifth field is ``"manual"`` is never dropped by the giant rule."""
    from ..table import BoxTable
    from .plan import MaskPlan

    cfg = _cfg(cfg)
    if not boxes:
        return []
    def kind(b):
        return "manual" if len(b) > 4 and b[4] == "manual" else "det"

    t = BoxTable(video="image", width=int(width), height=int(height), fps=1.0, kind="frames",
                 time_base=None, first_pts=0,
                 frames=[[0, [[float(x) for x in b[:4]] + [kind(b)] for b in boxes]]])
    return MaskPlan(t, cfg).boxes_at(0.0)


# ---------------------------------------------------------------------------------------
# one image
# ---------------------------------------------------------------------------------------
def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _decode(data: bytes, kind: str):
    """uint8/uint16 array of an encoded image (JPEG as 3-channel BGR or RGB, upright; PNG as
    stored, alpha kept), and the library used."""
    import numpy as np

    if _have_cv2():
        import cv2
        flag = cv2.IMREAD_COLOR if kind == "jpeg" else cv2.IMREAD_UNCHANGED
        img = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        if img is None:
            raise ValueError(f"{kind.upper()} does not decode")
        return img, "cv2"
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ImportError("masking images needs opencv-python-headless or Pillow "
                          "(pip install 'openveil[images]')") from None
    im = Image.open(io.BytesIO(data))
    im.load()
    if kind == "jpeg":
        im = ImageOps.exif_transpose(im).convert("RGB")
    elif im.mode == "P":
        im = im.convert("RGBA")
    return np.array(im), "pil"


def _encode(img, kind: str, lib: str, quality: int) -> bytes:
    if lib == "cv2":
        import cv2
        if kind == "jpeg":
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        else:
            ok, enc = cv2.imencode(".png", img)
        if not ok:
            raise ValueError(f"{kind.upper()} encode failed")
        return enc.tobytes()
    from PIL import Image
    buf = io.BytesIO()
    im = Image.fromarray(img)
    if kind == "jpeg":
        im.save(buf, format="JPEG", quality=int(quality))
    else:
        im.save(buf, format="PNG")
    return buf.getvalue()


def _kind_of(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in _JPEG_EXTS:
        return "jpeg"
    if ext == ".png":
        return "png"
    raise ValueError(f"{name}: only JPEG and PNG images can be masked")


def mask_image_bytes(data: bytes, boxes: Sequence[Sequence[float]], name: str, cfg: Any = None,
                     rules: bool = True, quality: Optional[int] = None) -> bytes:
    """Mosaic one encoded image (JPEG or PNG, by the extension of ``name``) and return the new
    encoded bytes. With ``rules`` the boxes are raw detections (giant rule and enlargement are
    applied first); without, they are final mosaic boxes."""
    from .mosaic import mosaic_array

    cfg = _cfg(cfg)
    kind = _kind_of(name)
    img, lib = _decode(data, kind)
    h, w = img.shape[:2]
    outside = [list(b[:4]) for b in boxes if b[0] >= w or b[1] >= h or b[2] <= 0 or b[3] <= 0]
    if outside:
        # most likely boxes in another coordinate frame (for example a JPEG with an EXIF
        # orientation, whose boxes must refer to the upright image)
        raise ValueError(f"{name}: box {outside[0]} lies outside the {w} x {h} image (boxes must refer to the "
                         "upright image)")
    final = image_rules(boxes, w, h, cfg) if rules else [list(b[:4]) for b in boxes]
    mosaic_array(img, final, _block(h, cfg))
    return _encode(img, kind, lib, cfg.images.jpeg_quality if quality is None else quality)


def mask_jpeg(data: bytes, boxes: Sequence[Sequence[float]], quality: int = 95, block: Optional[int] = None) -> bytes:
    """Mosaic a JPEG over the final ``boxes`` and re-encode it at ``quality`` (production
    behaviour of the frame-tar step; ``block`` defaults to the production grid)."""
    from .mosaic import block_size, mosaic_array

    img, lib = _decode(data, "jpeg")
    mosaic_array(img, [list(b[:4]) for b in boxes], block or block_size(img.shape[0]))
    return _encode(img, "jpeg", lib, quality)


def _decodes(data: bytes, name: str) -> Optional[Tuple[int, int]]:
    """(height, width) of an encoded image, or None if it does not decode."""
    try:
        img, _ = _decode(data, _kind_of(name))
        return img.shape[:2]
    except Exception:
        return None


# ---------------------------------------------------------------------------------------
# name matching
# ---------------------------------------------------------------------------------------
def _norm_name(n: str) -> str:
    n = n.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    return n


def resolve_names(keys: Iterable[str], names: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    """``({key: image name}, [keys that match no image or more than one])``.

    A key matches an image when the names are equal; otherwise, if exactly one image fits, when
    they are equal ignoring a leading ``./`` and letter case, or when one is the other with
    leading folders removed (``frames/cam1/a.jpg`` and ``cam1/a.jpg``; ``a.jpg`` and
    ``cam1/a.jpg``)."""
    names = list(names)
    exact = set(names)
    by_norm: Dict[str, List[str]] = {}
    by_base: Dict[str, List[str]] = {}
    for n in names:
        nn = _norm_name(n).casefold()
        by_norm.setdefault(nn, []).append(n)
        by_base.setdefault(nn.rsplit("/", 1)[-1], []).append(n)
    found: Dict[str, str] = {}
    absent: List[str] = []
    for key in keys:
        if key in exact:
            found[key] = key
            continue
        k = _norm_name(key).casefold()
        cand = by_norm.get(k, [])
        if len(cand) != 1:
            cand = [n for n in by_base.get(k.rsplit("/", 1)[-1], [])
                    if (lambda nn: nn == k or nn.endswith("/" + k) or k.endswith("/" + nn))(_norm_name(n).casefold())]
        if len(cand) == 1:
            found[key] = cand[0]
        else:
            absent.append(key)
    return found, sorted(absent)


def _resolve(plan: Plan, names: Iterable[str]) -> Tuple[Dict[str, List[Sequence[float]]], List[str]]:
    """``({member name: boxes}, [plan keys that match no member])`` (see :func:`resolve_names`);
    boxes given for the same member under several names are merged."""
    found, absent = resolve_names(list(plan), names)
    out: Dict[str, List[Sequence[float]]] = {}
    for key, boxes in plan.items():
        if key in found:
            out.setdefault(found[key], []).extend(list(boxes))
    return out, absent


# ---------------------------------------------------------------------------------------
# tar archives
# ---------------------------------------------------------------------------------------
class _HashWriter:
    """File wrapper that hashes and counts what is written."""

    def __init__(self, f):
        self.f, self.n, self.h = f, 0, hashlib.sha256()

    def write(self, b) -> int:
        self.h.update(b)
        self.f.write(b)
        self.n += len(b)
        return len(b)


def rewrite_tar(src_tar: str, dst_tar: str, plan: Plan, cfg: Any = None, rules: bool = True,
                quality: Optional[int] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Copy ``src_tar`` to ``dst_tar`` member by member, mosaicking the members in ``plan``
    (``{member name: boxes}``, names already resolved).

    Every member keeps its header fields (name, mode, user and group, mtime, type, link
    target); untouched members keep their bytes. The archive is written in POSIX (pax) tar
    format, which equals plain ustar for members that need no extended header.

    Returns ``(stats, member names in order)``; stats has ``members``, ``masked``,
    ``tar_bytes`` and ``sha256`` of the written archive.
    """
    cfg = _cfg(cfg)
    n_all, n_masked, names = 0, 0, []
    with open(dst_tar, "wb") as raw:
        hw = _HashWriter(raw)
        with tarfile.open(src_tar, "r:*") as ti, \
                tarfile.open(fileobj=hw, mode="w|", format=tarfile.PAX_FORMAT) as to:
            for m in ti:
                info = copy.copy(m)
                # the size is rewritten below; any other extended header (long name, fractional
                # mtime, ...) is kept as it was
                info.pax_headers = {k: v for k, v in m.pax_headers.items() if k != "size"}
                if m.isfile():
                    data = ti.extractfile(m).read()
                    if m.name in plan:
                        data = mask_image_bytes(data, plan[m.name], m.name, cfg, rules=rules, quality=quality)
                        n_masked += 1
                    info.size = len(data)
                    to.addfile(info, io.BytesIO(data))
                else:
                    to.addfile(info)
                names.append(m.name)
                n_all += 1
    return dict(members=n_all, masked=n_masked, tar_bytes=hw.n, sha256=hw.h.hexdigest()), names


def check_tar(src_tar: str, dst_tar: str, plan: Plan) -> Tuple[bool, str]:
    """Same member names in the same order; untouched members byte-identical with the same
    mtime and mode; masked members decode to an image of the original size."""
    with tarfile.open(src_tar, "r:*") as a, tarfile.open(dst_tar, "r:*") as b:
        ma, mb = a.getmembers(), b.getmembers()
        if [m.name for m in ma] != [m.name for m in mb]:
            return False, "member names differ"
        for x, y in zip(ma, mb):
            if x.type != y.type or (x.mtime, x.mode) != (y.mtime, y.mode):
                return False, f"{x.name}: header changed"
            if not x.isfile():
                if x.linkname != y.linkname:
                    return False, f"{x.name}: link target changed"
                continue
            da, db = a.extractfile(x).read(), b.extractfile(y).read()
            if x.name in plan:
                sa, sb = _decodes(da, x.name), _decodes(db, x.name)
                if sb is None:
                    return False, f"{x.name} does not decode"
                if sa is not None and sa != sb:
                    return False, f"{x.name} changed size"
            elif da != db:
                return False, f"{x.name} changed but is not masked"
    return True, ""


# ---------------------------------------------------------------------------------------
# folders
# ---------------------------------------------------------------------------------------
def _walk(root: str) -> List[str]:
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return out


def mask_folder(src_dir: str, dst_dir: str, plan: Plan, cfg: Any = None, rules: bool = True,
                quality: Optional[int] = None) -> Dict[str, Any]:
    """Copy the folder ``src_dir`` to ``dst_dir`` (recursively), mosaicking the images in
    ``plan`` (``{relative name: boxes}``, names already resolved); every other file is copied
    byte for byte (with its timestamps). Returns ``files`` and ``masked`` counts."""
    cfg = _cfg(cfg)
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise ValueError("the output folder must differ from the input folder")
    n, n_masked = 0, 0
    for rel in _walk(src_dir):
        s, d = os.path.join(src_dir, rel), os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        if rel in plan:
            with open(s, "rb") as f:
                data = mask_image_bytes(f.read(), plan[rel], rel, cfg, rules=rules, quality=quality)
            with open(d + ".tmp", "wb") as f:
                f.write(data)
            os.replace(d + ".tmp", d)
            n_masked += 1
        else:
            shutil.copy2(s, d)
        n += 1
    return dict(files=n, masked=n_masked)


def check_folder(src_dir: str, dst_dir: str, plan: Plan) -> Tuple[bool, str]:
    """Same relative file names; untouched files byte-identical; masked images decode to an
    image of the original size."""
    a, b = _walk(src_dir), _walk(dst_dir)
    if a != b:
        return False, "file names differ"
    for rel in a:
        with open(os.path.join(src_dir, rel), "rb") as f:
            da = f.read()
        with open(os.path.join(dst_dir, rel), "rb") as f:
            db = f.read()
        if rel in plan:
            sa, sb = _decodes(da, rel), _decodes(db, rel)
            if sb is None:
                return False, f"{rel} does not decode"
            if sa is not None and sa != sb:
                return False, f"{rel} changed size"
        elif da != db:
            return False, f"{rel} changed but is not masked"
    return True, ""


# ---------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------
def _tar_names(path: str) -> List[str]:
    with tarfile.open(path, "r:*") as t:
        return [m.name for m in t.getmembers() if m.isfile()]


def _absent_failure(r: Dict[str, Any], absent: List[str], t0: float) -> Dict[str, Any]:
    r.update(ok=False, stats={}, seconds=round(time.time() - t0, 1),
             why=f"boxes for {len(absent)} image(s) match no image (or more than one), e.g. {absent[0]!r}: nothing was "
                 "written (check the names, or pass allow_absent / --allow-missing)")
    return r


def mask_images(src: str, boxes: Plan, dst: str, cfg: Any = None, rules: bool = True,
                allow_absent: bool = False) -> Dict[str, Any]:
    """Mask an image folder or a tar archive and check the result.

    ``boxes``: ``{image name: boxes}`` (see the module docstring for ``rules`` and name
    matching). ``src`` is a folder or a ``.tar`` file; ``dst`` is the output folder or tar file.
    Images whose boxes are empty (or become empty under the rules) are copied unchanged. A tar
    output is written to ``<dst>.part`` and renamed only when the check passes. When boxes are
    given for a name that matches no image, nothing is written and the result fails, unless
    ``allow_absent`` is set. Returns a JSON-ready result with ``ok``, ``why``, ``kind``,
    ``stats``, ``masked_planned`` and ``planned_absent`` (names in ``boxes`` that match no
    image).
    """
    cfg = _cfg(cfg)
    t0 = time.time()
    plan_in = {k: v for k, v in boxes.items() if v}
    r: Dict[str, Any] = dict(input=os.path.basename(os.path.normpath(str(src))), ok=False)
    if os.path.isdir(src):
        if os.path.isdir(dst) and os.listdir(dst):
            raise ValueError("the output folder must be new or empty")
        names = _walk(src)
        plan, absent = _resolve(plan_in, names)
        bad = [n for n in plan if os.path.splitext(n)[1].lower() not in IMAGE_EXTS]
        if bad:
            raise ValueError(f"only JPEG and PNG images can be masked: {bad[:3]}")
        r.update(kind="folder", masked_planned=len(plan), planned_absent=absent)
        if absent and not allow_absent:
            return _absent_failure(r, absent, t0)
        st = mask_folder(src, dst, plan, cfg, rules=rules)
        ok, why = check_folder(src, dst, plan)
    else:
        if not tarfile.is_tarfile(src):
            raise ValueError(f"{os.path.basename(str(src))} is neither a folder nor a tar archive")
        plan, absent = _resolve(plan_in, _tar_names(src))
        bad = [n for n in plan if os.path.splitext(n)[1].lower() not in IMAGE_EXTS]
        if bad:
            raise ValueError(f"only JPEG and PNG images can be masked: {bad[:3]}")
        r.update(kind="tar", masked_planned=len(plan), planned_absent=absent)
        if absent and not allow_absent:
            return _absent_failure(r, absent, t0)
        part = str(dst) + ".part"
        try:
            st, _names = rewrite_tar(src, part, plan, cfg, rules=rules)
            ok, why = check_tar(src, part, plan)
            if ok and st["masked"] != len(plan):
                ok, why = False, "masked count differs"
            if ok:
                os.replace(part, dst)
                r.update(sha256=st["sha256"], bytes=st["tar_bytes"])
        finally:
            if os.path.exists(part):
                os.remove(part)
    r.update(stats=st, ok=bool(ok), why=why, seconds=round(time.time() - t0, 1))
    return r
