# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Second stage of the detector: a crop classifier that keeps or drops each proposed box.

The proposer (Grounding DINO) finds almost every person, but it also boxes animals, feeders and
other objects that look a little like people. The verifier cuts each candidate box out of the
frame with some context around it, letterboxes it to a square and asks a fine-tuned image
classifier (a DINOv3 backbone with LoRA adapters and a small MLP head) whether it shows a person.
A box is kept when the classifier's person probability is at least ``threshold``.

Preprocessing must match the training data exactly:

* the frame is resized to ``resize_width`` pixels wide and round-tripped through JPEG at
  ``jpeg_quality`` (the training frames were stored that way); set either to ``None`` to skip it;
* the crop is the box plus ``pad`` of its width/height on every side, clipped to the frame;
* the crop is scaled to fit ``size`` x ``size`` and centred on a grey (114) canvas;
* pixels are scaled to [0, 1] and normalised with the ImageNet mean and standard deviation.

Model weights are loaded from a local folder (or a Hugging Face id for the backbone):
``adapter_dir`` holds the PEFT adapter (``adapter_config.json`` + ``adapter_model.safetensors``),
the head (``head.safetensors``, or the legacy ``head.pt``) and an optional ``meta.json`` whose
``size`` and ``pad`` must match the verifier settings. The released weights
(``Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16`` on the Hugging Face Hub, DINOv3 License) use exactly
this layout, so ``adapter_dir`` can be the downloaded folder. The LoRA weights may be merged into
the backbone in memory for speed; merged backbone weights are never written to disk. An adapter
whose weights do not reach the backbone (PEFT only warns when the saved keys do not match the
module names, as with transformers 5 and the DINOv3 adapter) is refused.

Two verifier objects share one interface (``threshold``, ``mode`` and ``score_frames``):

* :class:`CropVerifier` - the model described above;
* :class:`PassThroughVerifier` - verifier mode ``"none"``: no model; a candidate is kept when its
  proposer score is at least the threshold (0 keeps every candidate).

Optional dependencies (torch, transformers, peft, safetensors, opencv or Pillow) are imported
inside the functions that need them, so this module imports with numpy only.
"""
from __future__ import annotations

import json
import math
import os
import struct
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LETTERBOX_GREY",
    "crop_window",
    "letterbox",
    "make_crop",
    "prepare_frame",
    "normalize_crops",
    "build_head",
    "load_head",
    "save_head",
    "save_state_dict",
    "load_state_dict",
    "check_adapter_loaded",
    "CropVerifier",
    "PassThroughVerifier",
    "make_verifier",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
LETTERBOX_GREY = 114
MIN_CROP_PX = 4


# ---------------------------------------------------------------------------------------------
# image helpers (OpenCV when installed, Pillow otherwise)
# ---------------------------------------------------------------------------------------------
def _cv2():
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except ImportError:
        return None


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an HxWx3 uint8 image to ``width`` x ``height``.

    With OpenCV: area interpolation when shrinking, bilinear when enlarging (the training code
    used exactly this). Without OpenCV, Pillow's box filter and bilinear filter are used; they
    give very similar but not bit-identical pixels.
    """
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    shrink = width * height < w * h
    cv2 = _cv2()
    if cv2 is not None:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    from PIL import Image  # noqa: PLC0415

    flt = Image.Resampling.BOX if shrink else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((width, height), flt))


def _jpeg_round_trip_bgr(bgr: np.ndarray, quality: int) -> np.ndarray:
    """Encode a BGR image as JPEG at ``quality`` and decode it again (OpenCV, or Pillow)."""
    cv2 = _cv2()
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    bio = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1])).save(bio, format="JPEG", quality=int(quality))
    bio.seek(0)
    return np.asarray(Image.open(bio).convert("RGB"))[:, :, ::-1].copy()


# ---------------------------------------------------------------------------------------------
# crop geometry
# ---------------------------------------------------------------------------------------------
def crop_window(box: Sequence[float], pad: float, width: int, height: int,
                corner: Sequence[float] | None = None) -> tuple[int, int, int, int]:
    """Integer pixel window ``(x1, y1, x2, y2)`` cut out for a box.

    The box is enlarged by ``pad`` of its width and height on every side, rounded outwards and
    clipped to the image. A window smaller than 4 pixels in either direction is replaced by a
    4 x 4 window at ``corner`` (default: the box's top-left corner), clamped into the image.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = x2 - x1, y2 - y1
    ex1, ex2 = x1 - pad * bw, x2 + pad * bw
    ey1, ey2 = y1 - pad * bh, y2 + pad * bh
    wx1, wy1 = int(max(0, math.floor(ex1))), int(max(0, math.floor(ey1)))
    wx2, wy2 = int(min(width, math.ceil(ex2))), int(min(height, math.ceil(ey2)))
    if wx2 - wx1 < MIN_CROP_PX or wy2 - wy1 < MIN_CROP_PX:
        cx, cy = (box[0], box[1]) if corner is None else (corner[0], corner[1])
        wx1 = max(0, min(width - MIN_CROP_PX, int(cx)))
        wy1 = max(0, min(height - MIN_CROP_PX, int(cy)))
        wx2, wy2 = wx1 + MIN_CROP_PX, wy1 + MIN_CROP_PX
    return wx1, wy1, wx2, wy2


def letterbox(crop: np.ndarray, size: int) -> np.ndarray:
    """Scale ``crop`` so that its longer side is ``size`` and centre it on a grey square canvas."""
    h, w = crop.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = _resize(crop, nw, nh)
    canvas = np.full((size, size, 3), LETTERBOX_GREY, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def make_crop(img: np.ndarray, box: Sequence[float], pad: float = 0.10, size: int = 224,
              rng: np.random.Generator | None = None, jitter: float = 0.0) -> np.ndarray:
    """Cut ``box`` (plus ``pad`` context per side) out of an RGB image and letterbox it.

    With ``rng`` and ``jitter`` (training augmentation) each box edge is moved by up to
    ``jitter`` of the box width/height first; a jitter that would leave less than 4 pixels is
    undone.
    """
    height, width = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if rng is not None and jitter:
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x1 += rng.uniform(-jitter, jitter) * bw
        x2 += rng.uniform(-jitter, jitter) * bw
        y1 += rng.uniform(-jitter, jitter) * bh
        y2 += rng.uniform(-jitter, jitter) * bh
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
    # the fallback 4 x 4 window sits at the corner of the original (not jittered) box, as in training
    wx1, wy1, wx2, wy2 = crop_window([x1, y1, x2, y2], pad, width, height, corner=box[:2])
    return letterbox(img[wy1:wy2, wx1:wx2], size)


def prepare_frame(bgr: np.ndarray, resize_width: int | None = 1280, jpeg_quality: int | None = 90) -> tuple[np.ndarray, float]:
    """Frame preprocessing of the verifier: ``(rgb, scale)``.

    The BGR frame is resized to ``resize_width`` pixels wide (aspect ratio kept, height rounded)
    and round-tripped through JPEG at ``jpeg_quality``, then converted to RGB. ``scale`` maps
    source pixel coordinates to the returned image (multiply box coordinates by it).
    """
    h, w = bgr.shape[:2]
    scale = 1.0
    small = bgr
    if resize_width:
        scale = float(resize_width) / w
        if w != resize_width:
            # area interpolation in both directions, exactly as the production code
            cv2 = _cv2()
            size = (int(resize_width), int(round(h * scale)))
            if cv2 is not None:
                small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            else:
                small = _resize(bgr, *size)
    if jpeg_quality:
        small = _jpeg_round_trip_bgr(np.ascontiguousarray(small), int(jpeg_quality))
    return np.ascontiguousarray(small[:, :, ::-1]), scale


def normalize_crops(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Stack uint8 RGB crops into a float32 N x 3 x H x W array with ImageNet normalisation."""
    x = (np.stack(crops).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------------------------
# safetensors (the library when installed, a small reader/writer for plain tensors otherwise)
# ---------------------------------------------------------------------------------------------
_ST_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
    "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}
_ST_NAMES = {np.dtype(v): k for k, v in _ST_DTYPES.items()}


def save_state_dict(tensors: Mapping[str, Any], path: str, metadata: Mapping[str, str] | None = None) -> None:
    """Write a dict of tensors (torch tensors or numpy arrays) in the safetensors format.

    Uses the ``safetensors`` package when it is installed; otherwise a minimal writer that
    produces a standard file (little-endian, header JSON padded to 8 bytes).
    """
    arrays = {}
    for k, v in tensors.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu()
            if str(v.dtype) == "torch.bfloat16":
                v = v.float()
            v = v.numpy()
        arrays[k] = np.ascontiguousarray(np.asarray(v))
    try:
        from safetensors.numpy import save_file  # noqa: PLC0415

        save_file(arrays, path, metadata=dict(metadata) if metadata else None)
        return
    except ImportError:
        pass
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    order = sorted(arrays)
    for k in order:
        a = arrays[k]
        if a.dtype not in _ST_NAMES:
            raise TypeError(f"unsupported dtype {a.dtype} for {k}")
        n = a.nbytes
        header[k] = {"dtype": _ST_NAMES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in order:
            f.write(arrays[k].astype(arrays[k].dtype.newbyteorder("<"), copy=False).tobytes())
    os.replace(tmp, path)


def load_state_dict(path: str) -> dict[str, np.ndarray]:
    """Read a safetensors file into a dict of numpy arrays (BF16 tensors become float32)."""
    try:
        from safetensors.numpy import load_file  # noqa: PLC0415

        return dict(load_file(path))
    except ImportError:
        pass
    except Exception:  # pragma: no cover - e.g. BF16, which safetensors.numpy cannot load
        pass
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        data = f.read()
    out = {}
    for k, spec in header.items():
        if k == "__metadata__":
            continue
        a, b = spec["data_offsets"]
        raw = data[a:b]
        if spec["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif spec["dtype"] in _ST_DTYPES:
            arr = np.frombuffer(raw, dtype=np.dtype(_ST_DTYPES[spec["dtype"]]).newbyteorder("<"))
        else:
            raise TypeError(f"unsupported safetensors dtype {spec['dtype']} for {k}")
        out[k] = arr.reshape(spec["shape"]).copy()
    return out


# ---------------------------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------------------------
def build_head(dim: int, hidden: int = 1024):
    """The classifier head: ``dim -> hidden -> hidden/2 -> 2`` with ReLU and dropout 0.5 / 0.25.

    Output 1 is "person", output 0 is "not person".
    """
    import torch.nn as nn  # noqa: PLC0415

    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                         nn.Dropout(0.25), nn.Linear(hidden // 2, 2))


def load_head(path: str, device: str = "cpu"):
    """Load the MLP head from ``head.safetensors`` or a legacy ``head.pt`` state dict.

    The layer sizes are read from the weights, so no extra metadata is needed. Legacy ``.pt``
    files are read with ``weights_only=True`` (plain tensors only, no pickled code). Any
    failure of that restricted load is a refusal: the file is never read again with the full
    unpickler, which could run code stored in it.
    """
    import torch  # noqa: PLC0415

    if path.endswith(".safetensors"):
        state = {k: torch.from_numpy(np.array(v)) for k, v in load_state_dict(path).items()}
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 - never fall back to an unrestricted load
            raise ValueError(f"{path}: cannot be read as plain tensors ({type(e).__name__}: {e}). "
                             "Convert the head to head.safetensors with a trusted copy of the weights.") from e
    if not isinstance(state, dict) or "0.weight" not in state or "6.weight" not in state:
        raise ValueError(f"{path} does not look like a verifier head (keys {sorted(state)[:6] if isinstance(state, dict) else type(state).__name__})")
    dim, hidden = int(state["0.weight"].shape[1]), int(state["0.weight"].shape[0])
    head = build_head(dim, hidden)
    head.load_state_dict({k: v.float() for k, v in state.items()})
    return head.to(device).eval()


def save_head(head, path: str) -> None:
    """Save the head's state dict as safetensors (``path`` should end in ``.safetensors``)."""
    save_state_dict(head.state_dict(), path, metadata={"format": "pt"})


def check_adapter_loaded(model, where: str = "the adapter") -> int:
    """Refuse a LoRA adapter whose weights did not reach the model; return the number of LoRA layers.

    When the keys saved in an adapter do not match the module names of the backbone (for example
    after a transformers release renames them), PEFT only warns and creates fresh LoRA layers,
    whose B matrices start at zero: the model then silently runs without the adapter. A trained
    adapter has no all-zero B matrix, so any such matrix (or no LoRA layer at all) is an error.
    """
    n = zero = 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            n += 1
            if not bool(p.detach().abs().sum() > 0):
                zero += 1
    if n == 0:
        raise ValueError(f"no LoRA layers were created from {where}")
    if zero:
        raise ValueError(f"{zero} of {n} LoRA layers from {where} have no weights loaded: the adapter keys do not match "
                         "this backbone. Use transformers>=4.56,<5 (tested: 4.57.1 with peft 0.13.2)")
    return n


def _find_head(adapter_dir: str | None, head_path: str | None) -> str:
    if head_path:
        return head_path
    if not adapter_dir:
        raise ValueError("the verifier needs head_path or adapter_dir")
    for name in ("head.safetensors", "head.pt"):
        p = os.path.join(adapter_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no head.safetensors or head.pt in {adapter_dir}")


# ---------------------------------------------------------------------------------------------
# verifiers
# ---------------------------------------------------------------------------------------------
class PassThroughVerifier:
    """Verifier mode ``"none"``: no model; a candidate's score is its proposer score.

    With ``threshold`` 0 every candidate is kept; otherwise candidates whose proposer score is
    below ``threshold`` are dropped.
    """

    mode = "none"

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)
        self.stats: Counter = Counter()

    def load(self) -> "PassThroughVerifier":
        return self

    def score_frames(self, frames_bgr: Sequence[np.ndarray], boxes_per_frame: Sequence[Sequence[Sequence[float]]]) -> list[np.ndarray]:
        """Per frame, an array with the proposer score of each box (1.0 when a box has none)."""
        return [np.array([float(b[4]) if len(b) > 4 else 1.0 for b in boxes], np.float32) for boxes in boxes_per_frame]


class CropVerifier:
    """Fine-tuned crop classifier (backbone + optional LoRA adapter + MLP head).

    Parameters mirror the production setup: ``size`` 224, ``pad`` 0.10, frames resized to
    ``resize_width`` 1280 and round-tripped through JPEG ``jpeg_quality`` 90, keep a box when
    p(person) >= ``threshold`` 0.10. ``dtype`` is the autocast dtype of the backbone; the head
    always runs in float32. ``batch`` crops are scored per forward pass; a CUDA out-of-memory
    error halves it.
    """

    mode = "crop"

    def __init__(self, backbone_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", adapter_dir: str | None = None,
                 head_path: str | None = None, size: int = 224, pad: float = 0.10, resize_width: int | None = 1280,
                 jpeg_quality: int | None = 90, threshold: float = 0.10, device: str = "cuda", dtype: str = "bfloat16",
                 batch: int = 128, merge_lora: bool = True, check_meta: bool = True):
        self.backbone_id, self.adapter_dir, self.head_path = backbone_id, adapter_dir, head_path
        self.size, self.pad = int(size), float(pad)
        self.resize_width, self.jpeg_quality = resize_width, jpeg_quality
        self.threshold = float(threshold)
        self.device, self.dtype, self.batch = device, dtype, int(batch)
        self.merge_lora, self.check_meta = merge_lora, check_meta
        self.model = None
        self.head = None
        self.stats: Counter = Counter()

    # -- loading -------------------------------------------------------------------------------
    def check_meta_file(self) -> dict | None:
        """Compare ``meta.json`` in the adapter folder (if any) with ``size`` and ``pad``."""
        if not self.adapter_dir:
            return None
        p = os.path.join(self.adapter_dir, "meta.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        if "size" in meta and int(meta["size"]) != self.size:
            raise ValueError(f"verifier size {self.size} differs from the training size {meta['size']} in {p}")
        if "pad" in meta and abs(float(meta["pad"]) - self.pad) > 1e-9:
            raise ValueError(f"verifier pad {self.pad} differs from the training pad {meta['pad']} in {p}")
        return meta

    def load(self) -> "CropVerifier":
        """Load backbone, adapter and head (needs torch, transformers and, for an adapter, peft)."""
        import torch  # noqa: PLC0415
        from transformers import AutoModel  # noqa: PLC0415

        if self.check_meta:
            self.check_meta_file()
        dtype = getattr(torch, self.dtype)
        base = AutoModel.from_pretrained(self.backbone_id, torch_dtype=dtype).to(self.device)
        model = base
        if self.adapter_dir and os.path.exists(os.path.join(self.adapter_dir, "adapter_config.json")):
            from peft import PeftModel  # noqa: PLC0415

            model = PeftModel.from_pretrained(base, self.adapter_dir).eval()
            check_adapter_loaded(model, self.adapter_dir)
            if self.merge_lora:
                # merged in memory only (faster inference); the merged backbone is never saved
                model = model.merge_and_unload()
        self.model = model.eval()
        self.head = load_head(_find_head(self.adapter_dir, self.head_path), self.device)
        return self

    # -- scoring -------------------------------------------------------------------------------
    def score_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """p(person) for a list of letterboxed RGB crops."""
        if not crops:
            return np.zeros(0, np.float32)
        if self.model is None:
            self.load()
        import torch  # noqa: PLC0415

        t0 = time.time()
        out = []
        bs = self.batch
        k = 0
        dev_type = torch.device(self.device).type
        autocast_dtype = getattr(torch, self.dtype)
        with torch.no_grad():
            while k < len(crops):
                try:
                    x = torch.from_numpy(normalize_crops(crops[k:k + bs])).to(self.device, non_blocking=True)
                    with torch.autocast(dev_type, dtype=autocast_dtype, enabled=self.dtype != "float32"):
                        f = self.model(pixel_values=x).pooler_output
                    out.append(torch.softmax(self.head(f.float()), 1)[:, 1].float().cpu())
                except torch.cuda.OutOfMemoryError:
                    import gc  # noqa: PLC0415

                    gc.collect()
                    torch.cuda.empty_cache()
                    self.stats["oom"] += 1
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    continue
                k += x.shape[0]
        self.stats["clf_s"] += time.time() - t0
        return torch.cat(out).numpy()

    def crops_for(self, frame_bgr: np.ndarray, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        """Letterboxed crops of ``boxes`` (source pixels) after the frame preprocessing."""
        if not boxes:
            return []
        rgb, s = prepare_frame(frame_bgr, self.resize_width, self.jpeg_quality)
        return [make_crop(rgb, [b[0] * s, b[1] * s, b[2] * s, b[3] * s], self.pad, self.size) for b in boxes]

    def score_frames(self, frames_bgr: Sequence[np.ndarray], boxes_per_frame: Sequence[Sequence[Sequence[float]]]) -> list[np.ndarray]:
        """Per frame, p(person) of each box; all crops of the call are scored in shared batches."""
        crops, counts = [], []
        for img, boxes in zip(frames_bgr, boxes_per_frame):
            c = self.crops_for(img, boxes)
            crops += c
            counts.append(len(c))
        self.stats["crops"] += len(crops)
        p = self.score_crops(crops)
        out, k = [], 0
        for n in counts:
            out.append(p[k:k + n])
            k += n
        return out


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def make_verifier(spec: Any = None, device: str | None = None):
    """Build a verifier from a config section (``openveil.config.VerifierConfig``, a mapping or
    any object with the same attribute names).

    Recognised keys: ``mode`` (``"crop"`` or ``"none"``), ``backbone_id``, ``adapter_dir``,
    ``head_path``, ``size``, ``pad``, ``resize_width``, ``jpeg_quality``, ``threshold``,
    ``dtype``, ``batch``. ``device`` comes from the run settings. Mode ``"none"`` returns a
    :class:`PassThroughVerifier` that keeps candidates whose proposer score is at least
    ``threshold``. Mode ``"crop"`` without ``adapter_dir`` or ``head_path`` is an error: no
    weights are loaded by default (the released ones are a separate download under the DINOv3
    License).
    """
    mode = str(_get(spec, "mode", "crop") or "crop").lower()
    if mode == "none":
        return PassThroughVerifier(float(_first(_get(spec, "threshold"), 0.0)))
    if mode not in ("crop", "model"):
        raise ValueError(f"verifier mode must be 'crop' or 'none', not {mode!r}")
    adapter_dir, head_path = _get(spec, "adapter_dir"), _get(spec, "head_path")
    if not adapter_dir and not head_path:
        raise ValueError("verifier mode 'crop' needs verifier.adapter_dir (a folder with the adapter, head and meta.json, "
                         "such as the downloaded Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16) or verifier.head_path; "
                         "use verifier mode 'none' to run without a verifier")
    kw = dict(
        backbone_id=_get(spec, "backbone_id") or _get(spec, "backbone") or "facebook/dinov3-vitl16-pretrain-lvd1689m",
        adapter_dir=adapter_dir, head_path=head_path,
        size=int(_get(spec, "size", 224)), pad=float(_get(spec, "pad", 0.10)),
        resize_width=_get(spec, "resize_width", 1280), jpeg_quality=_get(spec, "jpeg_quality", 90),
        threshold=float(_first(_get(spec, "threshold"), 0.10)),
        device=device or _get(spec, "device", "cuda") or "cuda", dtype=_get(spec, "dtype", "bfloat16") or "bfloat16",
        batch=int(_get(spec, "batch", 128) or 128),
    )
    return CropVerifier(**kw)


def _first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None
