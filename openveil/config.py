# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Configuration of an OpenVeil run.

Every tunable value of the pipeline lives in one :class:`OpenVeilConfig`, grouped by stage::

    cfg = load_config()                      # the 'production' preset
    cfg = load_config("production")          # the same, by name
    cfg = load_config("configs/my_barn.yaml")  # a file: only the values that differ are needed
    cfg.apply.min_share                      # 0.55

A config file (YAML or JSON) mirrors :func:`to_dict`. It may name the preset it starts from
(``preset: production``) and then lists only the values it changes, for example::

    preset: production
    proposer:
      prompt: "worker. person. horse."
      positive_phrases: [worker, person]
      competitor_phrases: [horse]
    schedule:
      day_night:
        policy: clock

Unknown keys are an error, so a typo never silently falls back to a default. YAML files need
PyYAML; without it a ``.yaml`` file that is written as JSON (YAML is a superset of JSON) is
still accepted.

The 'production' preset holds the values that produced the case-study release (detection
schedule, Grounding DINO prompt ``"worker. person. cow. calf."`` and proposer settings, crop
verifier recipe, giant rule 0.55, 20% enlargement, +-2 s hold, mosaic grid of 1/36 of the image
height, smart-cut encoder settings), so a run with the defaults uses the evaluated
configuration. One value differs on purpose from the case study and is documented where it is
defined: the day/night policy defaults to ``always_full`` (every chunk gets the full schedule;
the camera-clock policy of the case study is an option).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

__all__ = [
    "DayNightPolicy", "ScheduleConfig", "ProposerConfig", "VerifierConfig", "PostprocessConfig",
    "ApplyConfig", "VideoConfig", "DecodeConfig", "ImagesConfig", "RunConfig", "OpenVeilConfig",
    "PRESETS", "preset_names", "get_preset", "load_config", "config_from_dict", "to_dict",
    "save_config", "validate_config",
]


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
@dataclass
class DayNightPolicy:
    """Which scan mode every chunk of a video gets.

    policy:
        ``"always_full"`` (default): every chunk gets the full day schedule. Safe for any
        footage; it costs more detector calls on long dark recordings.
        ``"clock"``: the camera clock decides. A chunk is ``full`` when its start, or its end
        plus ``margin_s``, falls between ``day_start`` and ``day_end`` (hours, local camera
        time); otherwise it gets ``night_mode``. The clock is the video's ``start_local``
        (ISO time, for example parsed from the file name with ``clock_regex``) plus the offset
        into the video. The case study used this policy with 06:00-23:00 and a 300 s margin.
        A day that crosses midnight is allowed (``day_start`` > ``day_end``, for example a
        camera clock in UTC); ``day_start`` == ``day_end`` is refused.
    night_mode:
        ``"sparse"`` (a checkpoint every ``night_base_s``, a hit never switches to the dense
        mode), ``"skip"`` (no detection at all) or ``"full"``.
    """
    policy: str = "always_full"
    day_start: float = 6.0
    day_end: float = 23.0
    night_mode: str = "sparse"
    margin_s: float = 300.0
    clock_regex: str = r"(\d{8})[-_](\d{6})"


@dataclass
class ScheduleConfig:
    """Which frames the detector runs on, and how gaps between detections are bridged.

    The video is cut into chunks of ``chunk_s``. In the full (day) mode a checkpoint runs every
    ``base_s``; up to ``lookahead`` checkpoints go to the detector in one batch. A hit back-fills
    from the previous clear checkpoint and switches to the dense mode (a detector call every
    ``dense_s``), which ends after ``clear_gaps`` strides without a hit. With
    ``last_checkpoint`` the last frame of every chunk is a checkpoint too. In the sparse (night)
    mode the checkpoints are ``night_base_s`` apart and a hit never starts the dense mode.

    Gaps between detections of the same person are filled by linear interpolation when the gap
    is at most ``max_gap_s`` (at night ``max(max_gap_s, night_gap_factor * night_base_s)``) and the
    two boxes are within ``gate_base + gate_per_s * gap`` box sizes of each other.
    """
    chunk_s: float = 60.0
    base_s: float = 1.0
    dense_s: float = 0.2
    night_base_s: float = 5.0
    clear_gaps: int = 2
    lookahead: int = 4
    last_checkpoint: bool = True
    max_gap_s: float = 3.0
    night_gap_factor: float = 2.2
    gate_base: float = 1.0
    gate_per_s: float = 1.5
    day_night: DayNightPolicy = field(default_factory=DayNightPolicy)


@dataclass
class ProposerConfig:
    """Open-vocabulary box proposer (Grounding DINO).

    The prompt is a list of phrases separated by ``". "``. A box's score is the highest score
    over the token groups of ``positive_phrases``; ``competitor_phrases`` are in the prompt only
    so that the model has somewhere else to put animals or objects (they are not scored).
    Boxes with a score of at least ``cand_thr`` become candidates; candidates that overlap a
    better one at IoU >= ``merge_iou`` are merged into it.

    The default is the case-study prompt ``"worker. person. cow. calf."`` with positive phrases
    worker/person and competitors cow/calf: the prompt the released crop verifier was trained
    with. It names the animals of the case-study barns so that fewer of them become candidates,
    and the verifier learned the animal candidates of this prompt as negatives. For other
    footage, keep the positive phrases and replace the competitors with the animals or objects
    that are mistaken for people there (every phrase must occur in the prompt), for example
    ``"worker. person. horse."`` with ``competitor_phrases: [horse]``; then re-check the verifier
    on that footage, since its candidates now come from a different prompt.
    """
    model_id: str = "IDEA-Research/grounding-dino-base"
    prompt: str = "worker. person. cow. calf."
    positive_phrases: List[str] = field(default_factory=lambda: ["worker", "person"])
    competitor_phrases: List[str] = field(default_factory=lambda: ["cow", "calf"])
    shortest_edge: int = 800
    longest_edge: int = 1333
    fp16: bool = True
    batch: int = 4
    cand_thr: float = 0.05
    merge_iou: float = 0.6


@dataclass
class VerifierConfig:
    """Crop verifier that keeps or drops every candidate box.

    ``mode="crop"``: the candidate box plus ``pad`` of its size as context is cut out of the
    frame, letterboxed to ``size`` px on grey 114, normalised with the ImageNet mean/std and
    classified by a backbone with merged LoRA adapters and an MLP head. Before cropping, the
    frame is resized to ``resize_width`` px wide and round-tripped through JPEG at
    ``jpeg_quality`` (the training frames went through the same steps; set either to ``None`` to
    skip it). A box is kept when p(person) >= ``threshold``.

    ``mode="none"``: every candidate with a proposer score >= ``threshold`` is kept (no
    verifier model; useful for footage without animals or for a first look).
    """
    mode: str = "crop"
    backbone_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    adapter_dir: Optional[str] = None
    head_path: Optional[str] = None
    size: int = 224
    pad: float = 0.10
    resize_width: Optional[int] = 1280
    jpeg_quality: Optional[int] = 90
    threshold: float = 0.10
    batch: int = 128
    dtype: str = "bfloat16"
    lora_r: int = 32
    lora_alpha: int = 64
    head_hidden: List[int] = field(default_factory=lambda: [1024, 512])


@dataclass
class PostprocessConfig:
    """Clean-up of the kept boxes of one detector frame.

    Near duplicates (IoU >= ``nms``) of a better box are removed. With ``container`` on, a box
    that holds at least ``container_min_inner`` other kept boxes (each at least
    ``container_inner_frac`` inside it) and is at least ``container_ratio`` times the area of
    their union is replaced by that union.
    """
    nms: float = 0.85
    container: bool = True
    container_min_inner: int = 2
    container_ratio: float = 3.0
    container_inner_frac: float = 0.8


@dataclass
class ApplyConfig:
    """From a box table to the mosaic of every frame.

    1. Giant rule: a box covering at least ``min_share`` of the image with at least
       ``min_inner`` smaller boxes (each at least ``inner_frac`` inside it) is dropped; boxes
       whose kind is in ``keep_kinds`` (hand-added ``manual`` boxes) are never dropped.
    2. Every box is enlarged by ``pad`` of its width/height on each side.
    3. Temporal hold: the frame at time t gets every box of the table within t +- ``hold_s``.
    4. Mosaic: one grid anchored at the image origin with cells of
       ``max(block_min, round(height / block_divisor))`` px; every cell touching a box is filled
       with its mean.

    ``episode_gap_s`` groups giant-rule firings into episodes for review. The ``override_*``
    values control how a reviewer's overrides file is merged into a table
    (see :func:`openveil.table.apply_overrides`); ``override_delete_window_s=None`` means
    "the same as ``hold_s``", so a deleted box no longer reaches the frame it was deleted in.
    """
    min_share: float = 0.55
    min_inner: int = 2
    inner_frac: float = 0.8
    keep_kinds: List[str] = field(default_factory=lambda: ["manual"])
    pad: float = 0.20
    hold_s: float = 2.0
    block_divisor: float = 36.0
    block_min: int = 16
    episode_gap_s: float = 1.0
    override_step_s: float = 0.5
    override_match_iou: float = 0.5
    override_delete_window_s: Optional[float] = None


@dataclass
class VideoConfig:
    """Encoder settings of the smart cut (only GOPs that carry a mosaic are re-encoded).

    H.264 is re-encoded with ``h264_codec`` at ``h264_crf`` and ``h264_preset``; HEVC with
    ``hevc_codec`` at ``hevc_crf``. Both without B-frames (``bframes``), with parameter sets
    repeated in every keyframe (``repeat_headers``), without scene-cut keyframes (``scenecut``)
    and with a keyframe interval of the source's median GOP length (``keyint=None``) or a fixed
    number. Other codecs are re-encoded in full with ``other_codec`` at the source bitrate.
    ``verify``: ``"fast"`` (packet count, pts, byte-identical copies, re-encoded runs decode),
    ``"full"`` (decode everything) or ``"none"``. With ``fallback_full_reencode`` a smart cut
    that fails its check (or a source that is not eligible) is re-encoded in full instead.
    """
    h264_codec: str = "libx264"
    h264_preset: str = "medium"
    h264_crf: int = 20
    hevc_codec: str = "libx265"
    hevc_preset: str = "medium"
    hevc_crf: int = 24
    other_codec: str = "mpeg4"
    bframes: int = 0
    repeat_headers: bool = True
    scenecut: int = 0
    keyint: Optional[int] = None
    copy_packet_duration: bool = True
    verify: str = "fast"
    fallback_full_reencode: bool = True


@dataclass
class DecodeConfig:
    """Video decoding in a child process that is killed when it stalls.

    ``first_frame_timeout_s``: time allowed before the first frame; ``stall_timeout_s``: time
    allowed between frames afterwards. Up to ``max_bad_pts`` frames with a non-increasing pts are
    dropped. ``threads``: decoder threads (a retry uses ``retry_threads``).
    """
    first_frame_timeout_s: float = 300.0
    stall_timeout_s: float = 120.0
    max_bad_pts: int = 50
    threads: int = 6
    retry_threads: int = 2


@dataclass
class ImagesConfig:
    """Masking of image folders or tar archives: masked JPEGs are re-encoded at
    ``jpeg_quality``; untouched files are copied byte for byte."""
    jpeg_quality: int = 95


@dataclass
class RunConfig:
    """Running detection over many videos.

    ``workers`` processes share the device; a video that fails ``max_attempts`` times is
    abandoned (recorded in a failure ledger); a worker that prints nothing for ``stall_s`` is
    restarted after ``restart_wait_s``.
    """
    device: str = "cuda"
    workers: int = 1
    max_attempts: int = 3
    stall_s: float = 2400.0
    restart_wait_s: float = 30.0


@dataclass
class OpenVeilConfig:
    """All settings of an OpenVeil run, grouped by stage."""
    preset: str = "production"
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dict (JSON/YAML-ready) of this config."""
        return to_dict(self)


# ---------------------------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------------------------
def _production() -> OpenVeilConfig:
    return OpenVeilConfig(preset="production")


PRESETS = {"production": _production}


def preset_names() -> List[str]:
    """Names of the built-in presets."""
    return sorted(PRESETS)


def get_preset(name: str = "production") -> OpenVeilConfig:
    """A fresh copy of a built-in preset."""
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; known presets: {', '.join(preset_names())}") from None


# ---------------------------------------------------------------------------------------------
# dict <-> dataclass
# ---------------------------------------------------------------------------------------------
def to_dict(cfg: Any) -> Dict[str, Any]:
    """Nested plain dict of a config dataclass (lists stay lists, tuples become lists)."""
    def conv(v):
        if dataclasses.is_dataclass(v):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(cfg)


def _merge(obj: Any, values: Mapping[str, Any], where: str) -> None:
    """Set the fields of dataclass ``obj`` from ``values`` (recursively), refusing unknown keys."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{where or 'config'}: expected a mapping, got {type(values).__name__}")
    fields = {f.name: f for f in dataclasses.fields(obj)}
    unknown = [k for k in values if k not in fields]
    if unknown:
        raise ValueError(f"{where or 'config'}: unknown key(s) {', '.join(map(str, unknown))}; "
                         f"known: {', '.join(fields)}")
    for k, v in values.items():
        cur = getattr(obj, k)
        path = f"{where}.{k}" if where else k
        if dataclasses.is_dataclass(cur):
            _merge(cur, v, path)
        else:
            setattr(obj, k, _coerce(cur, v, path))


def _coerce(cur: Any, v: Any, path: str) -> Any:
    """Light type check: numbers stay numbers, booleans stay booleans, lists stay lists."""
    if v is None:
        return None
    if isinstance(cur, bool):
        if not isinstance(v, bool):
            raise ValueError(f"{path}: expected true/false, got {v!r}")
        return v
    if isinstance(cur, int) and not isinstance(cur, bool):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            raise ValueError(f"{path}: expected an integer, got {v!r}")
        return int(v)
    if isinstance(cur, float):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{path}: expected a number, got {v!r}")
        return float(v)
    if isinstance(cur, list):
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"{path}: expected a list, got {v!r}")
        return list(v)
    if isinstance(cur, str) and not isinstance(v, str):
        raise ValueError(f"{path}: expected a string, got {v!r}")
    return v


def config_from_dict(values: Optional[Mapping[str, Any]], base: Union[str, OpenVeilConfig, None] = None) -> OpenVeilConfig:
    """Build a config from a (partial) nested dict.

    The starting point is ``base`` (a preset name or a config, copied), else the preset named in
    ``values['preset']``, else 'production'. Only the keys present in ``values`` change.
    """
    values = dict(values or {})
    preset = values.pop("preset", None)
    if isinstance(base, OpenVeilConfig):
        cfg = copy.deepcopy(base)
    else:
        cfg = get_preset(base or preset or "production")
    if preset is not None:
        cfg.preset = str(preset)
    _merge(cfg, values, "")
    validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------
def _read_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # optional dependency
        except ImportError:
            try:
                data = json.loads(text)
            except ValueError:
                raise ImportError(f"reading {os.path.basename(path)} needs PyYAML (pip install pyyaml), "
                                  "or write the file as JSON") from None
        else:
            data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{os.path.basename(path)}: the top level must be a mapping")
    return data


def load_config(source: Union[str, os.PathLike, Mapping[str, Any], OpenVeilConfig, None] = None) -> OpenVeilConfig:
    """Load a config.

    ``source`` may be ``None`` (the 'production' preset), a preset name, a path to a YAML or JSON
    file, a (partial) dict, or an :class:`OpenVeilConfig` (returned as a validated copy).
    """
    if source is None:
        return get_preset("production")
    if isinstance(source, OpenVeilConfig):
        cfg = copy.deepcopy(source)
        validate_config(cfg)
        return cfg
    if isinstance(source, Mapping):
        return config_from_dict(source)
    s = os.fspath(source)
    if s in PRESETS:
        return get_preset(s)
    if not os.path.exists(s):
        raise FileNotFoundError(f"no preset or config file named {s!r} (presets: {', '.join(preset_names())})")
    return config_from_dict(_read_mapping(s))


def save_config(cfg: OpenVeilConfig, path: Union[str, os.PathLike]) -> None:
    """Write the full config to ``path``: YAML for ``.yaml``/``.yml`` (needs PyYAML), else JSON."""
    p = os.fspath(path)
    data = to_dict(cfg)
    if p.lower().endswith((".yaml", ".yml")):
        import yaml  # optional dependency
        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2) + "\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def validate_config(cfg: OpenVeilConfig) -> None:
    """Raise ``ValueError`` listing every value that is out of range."""
    errs: List[str] = []

    def need(ok: bool, msg: str) -> None:
        if not ok:
            errs.append(msg)

    s = cfg.schedule
    need(s.chunk_s > 0, "schedule.chunk_s must be > 0")
    need(s.base_s > 0 and s.night_base_s > 0, "schedule.base_s and night_base_s must be > 0")
    need(s.dense_s >= 0, "schedule.dense_s must be >= 0 (0 = every frame)")
    need(s.clear_gaps >= 1 and s.lookahead >= 1, "schedule.clear_gaps and lookahead must be >= 1")
    need(s.max_gap_s >= 0, "schedule.max_gap_s must be >= 0")
    d = s.day_night
    need(d.policy in ("always_full", "clock"), "schedule.day_night.policy must be 'always_full' or 'clock'")
    need(d.night_mode in ("sparse", "skip", "full"), "schedule.day_night.night_mode must be 'sparse', 'skip' or 'full'")
    need(0 <= d.day_start <= 24 and 0 <= d.day_end <= 24, "schedule.day_night.day_start/day_end must be hours 0-24")
    need(d.day_start % 24 != d.day_end % 24,
         "schedule.day_night.day_start and day_end must differ (use policy 'always_full' for no night)")
    p = cfg.proposer
    need(bool(p.positive_phrases), "proposer.positive_phrases must not be empty")
    need(0 <= p.cand_thr <= 1 and 0 < p.merge_iou <= 1, "proposer.cand_thr must be in [0, 1] and merge_iou in (0, 1]")
    need(p.batch >= 1, "proposer.batch must be >= 1")
    v = cfg.verifier
    need(v.mode in ("crop", "none"), "verifier.mode must be 'crop' or 'none'")
    need(0 <= v.threshold <= 1, "verifier.threshold must be in [0, 1]")
    need(v.size > 0 and v.pad >= 0, "verifier.size must be > 0 and pad >= 0")
    need(v.jpeg_quality is None or 1 <= v.jpeg_quality <= 100, "verifier.jpeg_quality must be 1-100 or null")
    pp = cfg.postprocess
    need(0 < pp.nms <= 1, "postprocess.nms must be in (0, 1]")
    need(pp.container_min_inner >= 1 and pp.container_ratio > 0, "postprocess.container_min_inner must be >= 1 and container_ratio > 0")
    a = cfg.apply
    need(a.min_share > 0, "apply.min_share must be > 0")
    need(isinstance(a.min_inner, int) and a.min_inner >= 1,
         "apply.min_inner must be an integer >= 1 (0 would drop a lone big box, e.g. a person close to the camera)")
    need(0 < a.inner_frac <= 1, "apply.inner_frac must be in (0, 1]")
    need(a.pad >= 0 and a.hold_s >= 0, "apply.pad and apply.hold_s must be >= 0")
    need(a.block_divisor > 0 and a.block_min >= 1, "apply.block_divisor must be > 0 and block_min >= 1")
    need(a.override_step_s > 0, "apply.override_step_s must be > 0")
    need(0 < a.override_match_iou <= 1, "apply.override_match_iou must be in (0, 1]")
    need(a.override_delete_window_s is None or a.override_delete_window_s >= 0,
         "apply.override_delete_window_s must be >= 0 or null (= hold_s)")
    vc = cfg.video
    need(vc.verify in ("fast", "full", "none"), "video.verify must be 'fast', 'full' or 'none'")
    need(0 <= vc.h264_crf <= 51 and 0 <= vc.hevc_crf <= 51, "video CRF values must be 0-51")
    need(1 <= cfg.images.jpeg_quality <= 100, "images.jpeg_quality must be 1-100")
    r = cfg.run
    need(r.workers >= 1 and r.max_attempts >= 1, "run.workers and run.max_attempts must be >= 1")
    if errs:
        raise ValueError("invalid config:\n  " + "\n  ".join(errs))
