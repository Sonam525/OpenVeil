# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Mosaic a video with as little re-encoding as possible (smart cut).

A masked copy of a video is written in which only the groups of pictures (GOPs, from one
keyframe to the next) that contain a frame with a mosaic are decoded, mosaicked and
re-encoded in the source codec; every other packet is copied byte for byte. The output keeps
the source pts of every frame (variable frame rates stay as they are) and the source packet
durations.

Why the splice is clean
-----------------------
The smart cut needs three properties of the source stream, which :func:`eligibility` checks
for every file before cutting (a file that lacks one is re-encoded in full instead):

* no frame reordering (no B-frames: pts order equals decode order), so that a decoded run
  maps one-to-one onto its packets;
* every keyframe carries the parameter sets in-band (H.264 SPS/PPS, HEVC VPS/SPS/PPS) and is
  a random-access picture (H.264 IDR, HEVC IRAP), so that a decoder can switch from a
  re-encoded run back to the copied camera GOP after it;
* length-prefixed samples (MP4/MOV/MKV style) with an ``avcC``/``hvcC`` header.

A re-encoded run starts with an IDR picture that carries its own parameter sets, and the copied
GOP after it starts with the camera's own keyframe and parameter sets, so a decoder switches
cleanly at both ends. Codecs other than H.264 and HEVC (for example MPEG-4 Part 2, which keeps
its header only out of band) are re-encoded in full: MPEG-4 Part 2 in the same codec, others
with ``video.other_codec``, at about the source bitrate.

Mosaic rules (giant rule, enlargement, temporal hold, grid) come from
:class:`openveil.apply.plan.MaskPlan`; the mosaic is drawn on the decoded I420 planes (Y cells
of B px, U/V cells of B/2 px), so unmasked pixels of a re-encoded GOP change only by the
re-encoding.

Checks
------
:func:`verify_fast` (used after every cut): the same number of packets and the same pts as
the source; copied packets byte-identical; every re-encoded run plus the GOP after it decodes
with a fresh decoder (as a player would after seeking) to exactly one frame per packet.
:func:`verify`: the same, but the whole output and the whole source are decoded.
:func:`verify_pair`: a check of two files without the statistics of the cut.

Entry points: :func:`mask_video` (cut, check, fall back to a full re-encode, write the result
dict) and :func:`mask_videos` (a batch with resume). Needs PyAV (``av``) and numpy.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SMART_CODECS", "Eligibility", "annexb_to_lp", "nal_types", "nal_length_size",
    "index_packets", "gops_of", "plan_runs", "eligibility", "probe_eligibility",
    "encoder_settings", "smart_cut", "verify_fast", "verify", "verify_pair", "mask_video", "mask_videos",
    "sha256_file", "sha256_copy", "result_ok", "write_json_atomic",
]

#: Source codecs that can be smart-cut (re-encoded GOP by GOP in the same codec).
SMART_CODECS = ("h264", "hevc")
SMART_PIX_FMTS = ("yuv420p", "yuvj420p")   # the re-encoded runs are 8-bit 4:2:0

# NAL unit types
_H264_SPS, _H264_PPS, _H264_IDR = 7, 8, 5
_HEVC_VPS, _HEVC_SPS, _HEVC_PPS = 32, 33, 34
_HEVC_IRAP = range(16, 22)          # BLA_W_LP .. CRA_NUT

_STAT_DETAIL_KEYS = ("run_ranges", "gop_starts", "run_frames", "rule")


# ---------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------
def _load_cfg(cfg: Any):
    from ..config import load_config
    return load_config(cfg)


def _load_table(table: Any):
    """A table given as a path is loaded; a BoxTable or dict is used as it is."""
    if isinstance(table, (str, os.PathLike)):
        from ..table import load_table
        return load_table(table)
    return table


def _tget(table: Any, key: str, default: Any = None) -> Any:
    if isinstance(table, dict):
        return table.get(key, default)
    return getattr(table, key, default)


def _seconds_fn(table: Any, stream_tb: Fraction) -> Callable[[int], float]:
    """pts -> seconds on the table's clock (``(pts - first_pts) * time_base``).

    A table without a time base is read with the stream time base (the table was made from
    the same stream).
    """
    tb = _tget(table, "time_base")
    if tb:
        from ..table import seconds

        def sec(pts: int) -> float:
            return seconds(table, pts)
        return sec
    first = _tget(table, "first_pts") or 0
    f_tb = Fraction(stream_tb.numerator, stream_tb.denominator)

    def sec2(pts: int) -> float:
        return float((pts - first) * f_tb)
    return sec2


def _check_table_clock(table: Any, pk: Sequence[Tuple], F: Dict[str, Any], name: str) -> None:
    """Refuse a table whose clock does not match the stream: its boxes would land on no frame or
    on the wrong frames (for example a table made from another container of the same video)."""
    tb = _tget(table, "time_base")
    if tb:
        t_tb = Fraction(int(tb[0]), int(tb[1]))
        s_tb = Fraction(F["tb"].numerator, F["tb"].denominator)
        if t_tb != s_tb:
            raise ValueError(f"the box table's time base {t_tb} differs from the stream's {s_tb} in {name}: the table was "
                             "made from another file (for example before a remux); run detection on this file")
    fp = _tget(table, "first_pts")
    if fp is not None and pk:
        if int(fp) not in {p[0] for p in pk if p[0] is not None}:
            raise ValueError(f"the box table's first_pts {int(fp)} is not a frame of {name} (its timestamps start at "
                             f"{min(p[0] for p in pk if p[0] is not None)}): the table was made from another file")


def _make_plan(table: Any, cfg: Any):
    from .plan import MaskPlan
    return MaskPlan(table, cfg)


def _block_size(height: int, plan: Any, cfg: Any) -> int:
    """Mosaic cell size for a frame ``height`` px high, with the plan's grid settings."""
    from .mosaic import block_size
    div = getattr(plan, "block_divisor", None) or cfg.apply.block_divisor
    mn = getattr(plan, "block_min", None) or cfg.apply.block_min
    return int(block_size(height, div, mn))


def _rule_stats(plan: Any) -> Any:
    for name in ("rule_stats", "stats"):
        v = getattr(plan, name, None)
        if v is not None and not callable(v):
            return v
    return None


# ---------------------------------------------------------------------------------------
# bitstream helpers
# ---------------------------------------------------------------------------------------
def annexb_to_lp(data: bytes, nal_len: int = 4) -> bytes:
    """Annex B (start codes) -> length-prefixed NAL units (the MP4 sample format).

    Zero bytes between NAL units (trailing zeros before the next start code) are dropped;
    the last NAL unit is kept as it is. Data without a start code is returned unchanged.
    """
    b = bytes(data)
    pos, starts = 0, []
    while True:
        k = b.find(b"\x00\x00\x01", pos)
        if k < 0:
            break
        s0 = k - 1 if k > 0 and b[k - 1] == 0 else k          # 4-byte start code
        starts.append((s0, k + 3))
        pos = k + 3
    if not starts:
        return b
    out = bytearray()
    for n, (_s0, e) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(b)
        nal = b[e:end].rstrip(b"\x00") if n + 1 < len(starts) else b[e:end]
        out += len(nal).to_bytes(nal_len, "big") + nal
    return bytes(out)


def nal_length_size(codec: str, extradata: bytes) -> Optional[int]:
    """NAL length field size of a length-prefixed stream, from its ``avcC``/``hvcC`` header;
    ``None`` when the header is missing or in Annex B form."""
    ex = bytes(extradata or b"")
    if codec == "h264" and len(ex) > 4 and ex[0] == 1:
        return (ex[4] & 3) + 1
    if codec == "hevc" and len(ex) > 21 and ex[0] == 1:
        return (ex[21] & 3) + 1
    return None


def nal_types(sample: bytes, codec: str, nal_len: Optional[int] = 4) -> List[int]:
    """NAL unit types in one sample (length-prefixed, or Annex B when ``nal_len`` is None).

    A malformed length-prefixed sample yields the types read so far followed by ``-1``.
    """
    b = bytes(sample)

    def typ(h: int) -> int:
        return (h >> 1) & 0x3F if codec == "hevc" else h & 0x1F

    out: List[int] = []
    if nal_len is None:
        i = 0
        while True:
            k = b.find(b"\x00\x00\x01", i)
            if k < 0 or k + 3 >= len(b):
                break
            out.append(typ(b[k + 3]))
            i = k + 3
        return out
    i = 0
    while i + nal_len <= len(b):
        n = int.from_bytes(b[i:i + nal_len], "big")
        if n <= 0 or i + nal_len + n > len(b):
            out.append(-1)
            break
        out.append(typ(b[i + nal_len]))
        i += nal_len + n
    return out


# ---------------------------------------------------------------------------------------
# stream index
# ---------------------------------------------------------------------------------------
def index_packets(src: str) -> Tuple[List[Tuple[int, Optional[int], bool, int]], Dict[str, Any]]:
    """Packets of the video stream as ``[(pts, dts, is_key, size)]`` in decode order (empty
    packets skipped), plus stream facts.

    Facts: ``codec``, ``tb`` (time base), ``W``, ``H``, ``rate`` (average frame rate or
    None), ``extradata``, ``container``, ``pix_fmt``, ``full_range``, ``color`` (FFmpeg enum
    values), ``nal_len`` and ``key_nals`` (NAL types of every keyframe, H.264/HEVC only).

    The file must hold exactly one stream, a video stream (audio would need its own privacy
    decision and is refused).
    """
    import av

    with av.open(str(src)) as c:
        if len(c.streams) != 1 or c.streams[0].type != "video":
            raise ValueError(f"expected one video stream, found {[s.type for s in c.streams]} "
                             "(remove other streams first, for example with ffmpeg -map 0:v:0 -c copy)")
        st = c.streams.video[0]
        cc = st.codec_context
        codec = cc.name
        ex = bytes(cc.extradata or b"")
        nal_len = nal_length_size(codec, ex)
        pk, key_nals = [], []
        for p in c.demux(st):
            if p.size <= 0:
                continue
            key = bool(p.is_keyframe)
            pk.append((p.pts, p.dts, key, p.size))
            if key and codec in SMART_CODECS:
                key_nals.append(nal_types(bytes(p), codec, nal_len))
        pix = cc.format.name if cc.format else None
        facts = dict(codec=codec, tb=st.time_base, W=cc.width, H=cc.height, rate=st.average_rate,
                     extradata=ex, container=c.format.name, pix_fmt=pix,
                     full_range=bool(pix == "yuvj420p" or int(cc.color_range or 0) == 2),
                     color=dict(colorspace=int(cc.colorspace or 2), color_primaries=int(cc.color_primaries or 2),
                                color_trc=int(cc.color_trc or 2)),
                     nal_len=nal_len, key_nals=key_nals)
    return pk, facts


def gops_of(pk: Sequence[Tuple]) -> List[Tuple[int, int]]:
    """Packet index ranges ``[(start, end), ...]`` of the GOPs (keyframe to keyframe)."""
    keys = [i for i, p in enumerate(pk) if p[2]]
    if not keys or keys[0] != 0:
        keys = [0] + keys
    return [(a, b) for a, b in zip(keys, keys[1:] + [len(pk)])]


def plan_runs(pk: Sequence[Tuple], gops: Sequence[Tuple[int, int]], plan: Any,
              sec: Callable[[int], float]) -> Tuple[List[List[int]], List[float]]:
    """Runs of consecutive GOPs (packet index ranges ``[a, b)``) that contain a frame whose time
    lies in a masked interval of ``plan``, and the time of every packet.

    A frame counts as masked when it lies in one of ``plan.intervals()`` (with a tolerance of
    1 microsecond on both ends) or when ``plan.boxes_at`` gives it a box, so a frame that will
    be mosaicked is never left in a copied GOP by float rounding at an interval edge.
    """
    iv = [tuple(x) for x in plan.intervals()]
    times = [sec(p[0]) for p in pk]
    starts = [a for a, _ in iv]
    eps = 1e-6
    masked = []
    for a, b in gops:
        hit = False
        for k in range(a, b):
            t = times[k]
            i = bisect.bisect_right(starts, t + eps) - 1
            if (i >= 0 and iv[i][0] - eps <= t <= iv[i][1] + eps) or plan.boxes_at(t):
                hit = True
                break
        masked.append(hit)
    runs: List[List[int]] = []
    for (a, b), m in zip(gops, masked):
        if not m:
            continue
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return runs, times


# ---------------------------------------------------------------------------------------
# eligibility probe
# ---------------------------------------------------------------------------------------
@dataclass
class Eligibility:
    """Whether a source can be smart-cut; ``reasons`` lists why not (empty when it can)."""
    smart_cut: bool
    codec: str
    reasons: List[str] = field(default_factory=list)
    packets: int = 0
    keyframes: int = 0
    reordered_packets: int = 0
    keyframes_without_parameter_sets: int = 0
    keyframes_not_random_access: int = 0
    nal_length_size: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def eligibility(pk: Sequence[Tuple], facts: Dict[str, Any], cfg: Any = None) -> Eligibility:
    """Check the smart-cut assumptions on an indexed stream (see the module docstring).

    Repeated pts (equal timestamps) are allowed: the muxer moves such a packet by one tick
    and records it. Decreasing pts (frame reordering) are not.
    """
    import av

    cfg = _load_cfg(cfg)
    codec = facts["codec"]
    e = Eligibility(smart_cut=False, codec=codec, packets=len(pk), keyframes=sum(1 for p in pk if p[2]),
                    nal_length_size=facts.get("nal_len"))
    if codec not in SMART_CODECS:
        e.reasons.append(f"codec {codec} is re-encoded in full (smart cut supports {', '.join(SMART_CODECS)})")
        return e
    if not pk:
        e.reasons.append("no video packets")
        return e
    pix = facts.get("pix_fmt")
    if pix not in SMART_PIX_FMTS:
        # the re-encoded runs are 8-bit 4:2:0: spliced into a stream of another format they would
        # change chroma format or bit depth in mid-stream, which many decoders cannot play
        e.reasons.append(f"pixel format {pix} is re-encoded in full (smart cut needs {' or '.join(SMART_PIX_FMTS)})")
    enc_name = cfg.video.h264_codec if codec == "h264" else cfg.video.hevc_codec
    if enc_name not in av.codecs_available:
        e.reasons.append(f"encoder {enc_name} is not available in this FFmpeg build")
    if facts.get("nal_len") is None:
        e.reasons.append("no avcC/hvcC header (Annex B samples): the stream cannot be spliced into MP4 as it is")
    prev = None
    for p in pk:
        if p[0] is None:
            e.reasons.append("a packet has no pts")
            break
        if prev is not None and p[0] < prev:
            e.reordered_packets += 1
        prev = p[0]
    if e.reordered_packets:
        e.reasons.append(f"frame reordering (B-frames): {e.reordered_packets} packets have a lower pts than the one before")
    if not pk[0][2]:
        e.reasons.append("the first packet is not a keyframe")
    need = {_H264_SPS, _H264_PPS} if codec == "h264" else {_HEVC_VPS, _HEVC_SPS, _HEVC_PPS}
    for types in facts.get("key_nals", []):
        ts = set(types)
        if not need <= ts:
            e.keyframes_without_parameter_sets += 1
        ra = (_H264_IDR in ts) if codec == "h264" else any(t in _HEVC_IRAP for t in ts)
        if not ra:
            e.keyframes_not_random_access += 1
    if e.keyframes_without_parameter_sets:
        e.reasons.append(f"{e.keyframes_without_parameter_sets} of {e.keyframes} keyframes carry no in-band "
                         "parameter sets (they are only in the file header)")
    if e.keyframes_not_random_access:
        e.reasons.append(f"{e.keyframes_not_random_access} of {e.keyframes} keyframes are not "
                         f"{'IDR' if codec == 'h264' else 'IRAP'} pictures")
    e.smart_cut = not e.reasons
    return e


def probe_eligibility(src: str, cfg: Any = None) -> Eligibility:
    """:func:`eligibility` of a file (reads every packet header and every keyframe's NAL headers)."""
    pk, facts = index_packets(src)
    return eligibility(pk, facts, cfg)


# ---------------------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------------------
def encoder_settings(codec: str, cfg: Any = None, keyint: int = 25) -> Tuple[str, Dict[str, str]]:
    """Encoder name and options used to re-encode ``codec`` (``"h264"`` or ``"hevc"``).

    No B-frames, parameter sets repeated in every keyframe, no scene-cut keyframes and a fixed
    keyframe interval ``keyint`` (``video.keyint`` when set).
    """
    vc = _load_cfg(cfg).video
    g = int(vc.keyint or keyint)
    rh = 1 if vc.repeat_headers else 0
    if codec == "h264":
        return vc.h264_codec, {
            "preset": vc.h264_preset, "crf": str(vc.h264_crf), "bf": str(vc.bframes),
            "x264-params": f"repeat-headers={rh}:bframes={vc.bframes}:scenecut={vc.scenecut}"
                           f":keyint={g}:min-keyint={g}"}
    if codec == "hevc":
        return vc.hevc_codec, {
            "preset": vc.hevc_preset,
            "x265-params": f"crf={vc.hevc_crf}:bframes={vc.bframes}:repeat-headers={rh}:scenecut={vc.scenecut}"
                           f":info=0:log-level=error:keyint={g}:min-keyint={g}"}
    raise ValueError(f"no smart-cut encoder for codec {codec!r}")


def _median_gop(gops: Sequence[Tuple[int, int]]) -> int:
    import numpy as np
    return max(1, int(np.median([b - a for a, b in gops]))) if gops else 25


def _set_color(ctx: Any, F: Dict[str, Any]) -> None:
    """Carry the source colour description over to an encoder (best effort)."""
    for k, v in F.get("color", {}).items():
        try:
            setattr(ctx, k, v)
        except Exception:
            pass
    if F.get("full_range"):
        try:
            ctx.color_range = 2
        except Exception:
            pass


def _side_encoder(F: Dict[str, Any], gops, cfg):
    """A stand-alone encoder for the re-encoded runs of a smart cut (Annex B output)."""
    import av

    name, opts = encoder_settings(F["codec"], cfg, _median_gop(gops))
    c = av.CodecContext.create(name, "w")
    c.width, c.height, c.pix_fmt = F["W"], F["H"], "yuv420p"
    c.time_base = F["tb"]
    c.framerate = F["rate"] or Fraction(25)
    c.gop_size = int(cfg.video.keyint or _median_gop(gops))
    c.max_b_frames = 0
    _set_color(c, F)
    c.options = opts
    c.open()
    return c


def _full_stream(oc, F: Dict[str, Any], pk, gops, cfg):
    """Output stream (with its own encoder and header) for a full re-encode."""
    codec = F["codec"]
    rate = Fraction(F["rate"] or 25)
    if rate.denominator > 1001:
        # the encoder only needs a nominal rate (the muxed pts are the source pts); MPEG-4
        # Part 2 refuses time bases with a denominator above 65535
        rate = rate.limit_denominator(1001)
    if codec in SMART_CODECS:
        name, opts = encoder_settings(codec, cfg, _median_gop(gops))
        ost = oc.add_stream(name, rate=rate)
        ost.codec_context.gop_size = int(cfg.video.keyint or _median_gop(gops))
        ost.codec_context.options = opts
    else:
        # MPEG-4 Part 2 stays MPEG-4 Part 2; anything else gets video.other_codec; both at about
        # the source bitrate, with the source's first GOP length.
        name = "mpeg4" if codec == "mpeg4" else cfg.video.other_codec
        ost = oc.add_stream(name, rate=rate)
        pts = [p[0] for p in pk if p[0] is not None]
        dur = float((max(pts) - min(pts)) * F["tb"]) or 1.0
        ost.bit_rate = int(sum(p[3] for p in pk) * 8 / dur)
        ost.codec_context.gop_size = max(1, gops[0][1] - gops[0][0]) if gops else 12
    ost.width, ost.height, ost.pix_fmt = F["W"], F["H"], "yuv420p"
    ost.time_base = F["tb"]
    ost.codec_context.max_b_frames = 0
    _set_color(ost.codec_context, F)
    return ost, name


# ---------------------------------------------------------------------------------------
# muxing
# ---------------------------------------------------------------------------------------
class _Mux:
    """Muxes packets in order with strictly increasing dts.

    Some camera files repeat a packet's timestamp (for example a duplicated keyframe in
    mid-file) and the MP4 muxer refuses equal dts, so such a packet gets dts = previous + 1
    tick (its pts is raised to that if lower). Each change is recorded as ``[output packet
    index, new pts]`` in ``fixes``. All packets passed in carry the source time base.
    """

    def __init__(self, oc):
        self.oc, self.last, self.n, self.fixes = oc, None, 0, []

    def mux(self, pkt) -> None:
        if self.last is not None and pkt.dts is not None and pkt.dts <= self.last:
            pkt.dts = self.last + 1
            if pkt.pts is not None and pkt.pts < pkt.dts:
                pkt.pts = pkt.dts
            self.fixes.append([self.n, int(pkt.pts)])
        if pkt.dts is not None:
            self.last = pkt.dts
        self.n += 1
        self.oc.mux(pkt)


class _Run:
    """One re-encoded run: a fresh decoder, an encoder, and the source pts of the frames sent
    to the encoder (encoder output order equals input order: no B-frames)."""

    def __init__(self, index: int, F: Dict[str, Any], ost, mx: _Mux, plan, sec, block: int, stats: Dict[str, Any],
                 durs: Dict[int, int], cfg, encoder=None):
        import av

        self.index, self.F, self.ost, self.mx, self.plan, self.sec = index, F, ost, mx, plan, sec
        self.block, self.stats, self.durs, self.cfg = block, stats, durs, cfg
        self.dec = av.CodecContext.create(F["codec"], "r")
        if F["extradata"]:
            self.dec.extradata = F["extradata"]
        self.dec.thread_type = "AUTO"
        self.side = encoder is not None          # smart mode: stand-alone encoder, LP conversion
        self.enc = encoder if encoder is not None else ost.codec_context
        self.pending: Deque[int] = deque()
        self.last_in: Optional[int] = None
        self.n_in = 0
        self.n_packets = 0
        if not self.side:
            self.enc_tb = ost.codec_context.time_base

    def feed(self, packet) -> None:
        self.n_packets += 1
        for fr in self.dec.decode(packet):
            self._encode(fr)

    def finish(self) -> None:
        for fr in self.dec.decode(None):
            self._encode(fr)
        for pkt in self._enc(None):
            self._mux(pkt)
        if self.pending:
            raise RuntimeError(f"run ended with {len(self.pending)} frames not encoded")
        self.stats.setdefault("run_frames", []).append(self.n_packets)

    def _enc(self, frame):
        if self.side:
            return self.enc.encode(frame)
        return self.ost.encode(frame)

    def _encode(self, fr) -> None:
        import av

        boxes = self.plan.boxes_at(self.sec(fr.pts))
        out = _prepare_frame(fr, boxes, self.block, self.F)
        if boxes:
            self.stats["frames_mosaicked"] += 1
        out.pict_type = av.video.frame.PictureType.NONE     # the encoder decides the frame types
        self.pending.append(fr.pts)
        self.n_in += 1
        if self.side:
            out.time_base = self.F["tb"]
            out.pts = fr.pts
            if self.last_in is not None and out.pts <= self.last_in:
                out.pts = self.last_in + 1      # repeated camera timestamp: the encoder needs increasing pts
            self.last_in = out.pts
        else:
            # full re-encode: the encoder sees a constant-rate frame counter; the muxed pts
            # come from `pending`, so variable frame rates and repeated timestamps survive.
            out.time_base = self.enc_tb
            out.pts = self.n_in - 1
        for pkt in self._enc(out):
            self._mux(pkt)

    def _mux(self, pkt) -> None:
        import av

        pts = self.pending.popleft()
        if self.side:
            q = av.Packet(annexb_to_lp(bytes(pkt), self.F["nal_len"] or 4))
            q.is_keyframe = pkt.is_keyframe
        else:
            q = pkt
        q.pts = q.dts = pts
        q.time_base = self.F["tb"]
        d = self.durs.get(pts)
        if d and self.cfg.video.copy_packet_duration:
            # the source duration: without it the MP4 edit list ends at the start of the last
            # sample and players drop the last frame
            q.duration = d
        q.stream = self.ost
        self.mx.mux(q)


def _prepare_frame(fr, boxes, block: int, F: Dict[str, Any]):
    """The frame to encode: mosaicked I420 planes when there are boxes, else the frame itself
    (converted to ``yuv420p`` if it has another layout). Full-range (``yuvj420p``) frames keep
    their sample values and are passed on as full-range ``yuv420p``."""
    import av
    from ..io.video import frame_to_i420
    from .mosaic import mosaic_yuv420

    name = fr.format.name
    if not boxes and name == "yuv420p":
        return fr
    if not boxes and name not in ("yuv420p", "yuvj420p"):
        return fr.reformat(format="yuv420p")
    buf, full = frame_to_i420(fr)
    if boxes:
        # flat plane offsets: with H % 4 == 2 a chroma plane does not end on a whole row of W
        mosaic_yuv420(buf, boxes, block, width=fr.width, height=fr.height)
    out = av.VideoFrame.from_ndarray(buf, format="yuv420p")
    if full:
        try:
            out.color_range = 2
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------------------
# smart cut
# ---------------------------------------------------------------------------------------
def smart_cut(src: str, table: Any, dst: str, cfg: Any = None, full: bool = False, plan: Any = None,
              log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Write ``dst`` (MP4): ``src`` with the mosaic of ``table`` applied.

    Only the GOP runs that hold a masked frame are re-encoded, unless ``full`` is set or the
    source is not eligible (:func:`eligibility`), in which case every frame is re-encoded.
    ``plan`` may be passed to reuse an existing :class:`MaskPlan`. Returns statistics
    (numbers only); pass them to :func:`verify_fast` or :func:`verify`.
    """
    import av
    import numpy as np

    cfg = _load_cfg(cfg)
    if cfg.video.bframes != 0:
        raise ValueError("video.bframes must be 0: the smart cut maps encoder output to source pts in order")
    t_start = time.time()
    table = _load_table(table)
    plan = plan if plan is not None else _make_plan(table, cfg)
    pk, F = index_packets(src)
    if not pk:
        raise ValueError(f"no video packets in {os.path.basename(str(src))}")
    tw, th = _tget(table, "width"), _tget(table, "height")
    if (tw, th) != (F["W"], F["H"]):
        # boxes of another frame size would land in the wrong place
        raise ValueError(f"the box table is for {tw} x {th} px but {os.path.basename(str(src))} "
                         f"is {F['W']} x {F['H']} px")
    _check_table_clock(table, pk, F, os.path.basename(str(src)))
    elig = eligibility(pk, F, cfg)
    requested_full = bool(full)
    if not full and not elig.smart_cut:
        full = True
    gops = gops_of(pk)
    sec = _seconds_fn(table, F["tb"])
    runs, _times = plan_runs(pk, gops, plan, sec)
    if full:
        runs = [[0, len(pk)]]
    in_run = np.full(len(pk), -1, np.int64)
    for r, (a, b) in enumerate(runs):
        in_run[a:b] = r
    B = _block_size(int(F["H"]), plan, cfg)
    stats: Dict[str, Any] = dict(
        packets=len(pk), gops=len(gops), runs=len(runs), packets_reencoded=int((in_run >= 0).sum()),
        frames_mosaicked=0, codec=F["codec"], mode="full" if full else "smart",
        full_reason=("requested" if requested_full else ("; ".join(elig.reasons) if full else None)),
        eligible=elig.smart_cut, block=B, rule=_rule_stats(plan),
        run_ranges=[[int(a), int(b)] for a, b in runs], gop_starts=[int(a) for a, _ in gops])
    if log:
        log(f"{os.path.basename(str(src))}: {stats['mode']} cut, {stats['runs']} runs, "
            f"{stats['packets_reencoded']}/{stats['packets']} packets re-encoded")
    tmp = str(dst) + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    try:
        with av.open(str(src)) as ic, av.open(tmp, "w", format="mp4") as oc:
            ist = ic.streams.video[0]
            if full:
                ost, enc_name = _full_stream(oc, F, pk, gops, cfg)
            else:
                ost = oc.add_stream_from_template(ist)
                ost.time_base = ist.time_base
                enc_name = encoder_settings(F["codec"], cfg)[0]
            stats["encoder"] = enc_name
            oc.start_encoding()
            mx = _Mux(oc)
            run: Optional[_Run] = None
            durs: Dict[int, int] = {}
            for k, p in enumerate(q for q in ic.demux(ist) if q.size > 0):
                r = int(in_run[k])
                durs[p.pts] = p.duration
                if r < 0:
                    if run is not None:
                        run.finish()
                        run = None
                    p.stream = ost
                    mx.mux(p)
                    continue
                if run is None or run.index != r:
                    if run is not None:
                        run.finish()
                    run = _Run(r, F, ost, mx, plan, sec, B, stats, durs, cfg,
                               encoder=None if full else _side_encoder(F, gops, cfg))
                run.feed(p)
            if run is not None:
                run.finish()
        stats["ts_fixes"] = mx.fixes
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    stats["seconds"] = round(time.time() - t_start, 1)
    return stats


# ---------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------
def _packets(path: str, src_tb: Optional[Fraction] = None):
    """``([(pts, md5 digest)], extradata, codec, time_base)`` of the video packets of ``path``;
    pts are converted to ``src_tb`` when it differs from the file's time base."""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb = st.time_base
        ex = bytes(st.codec_context.extradata or b"")
        scale = None
        if src_tb is not None:
            ratio = Fraction(tb.numerator, tb.denominator) / Fraction(src_tb.numerator, src_tb.denominator)
            scale = ratio if ratio != 1 else None
        out = []
        for p in c.demux(st):
            if p.size <= 0:
                continue
            pts = p.pts
            if scale is not None and pts is not None:
                pts = int(round(pts * scale))
            out.append((pts, hashlib.md5(bytes(p)).digest()))
        return out, ex, st.codec_context.name, tb


def _expected_pts(ps, stats) -> List[int]:
    """The pts the output should have, packet by packet.

    A smart cut keeps the source packet order. A full re-encode writes the frames in
    presentation order (a source with B-frames stores them in decode order), so the expected
    list is sorted. Repeated camera timestamps moved by one tick (``ts_fixes``) are applied.
    """
    exp = [a[0] for a in ps]
    if stats.get("mode") == "full":
        exp = sorted(exp, key=lambda x: (x is None, x))
    for i, q in stats.get("ts_fixes", []):
        if i < len(exp):
            exp[i] = q
    return exp


def verify_fast(src: str, dst: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap check of a cut (see the module docstring); ``result["ok"]`` is the verdict.

    A full re-encode decodes the whole output instead of the runs.
    """
    import av
    import numpy as np

    ps, _, _, src_tb = _packets(src)
    pd_, ex, codec, _ = _packets(dst, src_tb)
    exp = _expected_pts(ps, stats)
    res: Dict[str, Any] = dict(packets_src=len(ps), packets_dst=len(pd_),
                               pts_equal=exp == [a[0] for a in pd_],
                               ts_fixes=len(stats.get("ts_fixes", [])))
    rr = stats.get("run_ranges", [])
    inrun = np.zeros(len(ps), bool)
    for a, b in rr:
        inrun[a:b] = True
    res["copied_identical"] = bool(len(ps) == len(pd_) and all(
        x[1] == y[1] for x, y, m in zip(ps, pd_, inrun) if not m))
    starts = stats.get("gop_starts", [0])
    if stats["mode"] == "full":
        windows = [(0, len(pd_))]
    else:
        windows = []
        for a, b in rr:
            k = bisect.bisect_right(starts, b)              # the GOP after the run (if any)
            windows.append((a, starts[k] if k < len(starts) else len(pd_)))
    bad, err = [], None
    want: Dict[int, List[int]] = {}
    for w, (a, b) in enumerate(windows):
        for i in range(a, b):
            want.setdefault(i, []).append(w)
    decs: Dict[int, Any] = {}
    counts: Dict[int, int] = {}
    try:
        with av.open(str(dst)) as c:
            st = c.streams.video[0]
            for i, p in enumerate(q for q in c.demux(st) if q.size > 0):
                for w in want.get(i, []):
                    if w not in decs:
                        d = av.CodecContext.create(codec, "r")
                        if ex:
                            d.extradata = ex
                        d.thread_type = "AUTO"
                        decs[w], counts[w] = d, 0
                    counts[w] += len(decs[w].decode(p))
                    if i == windows[w][1] - 1:
                        counts[w] += len(decs[w].decode(None))
                        n = windows[w][1] - windows[w][0]
                        if counts[w] != n:
                            bad.append((windows[w][0], windows[w][1], counts[w]))
                        del decs[w]
    except Exception as e:
        err = repr(e)[:200]
    res.update(windows=len(windows), windows_bad=bad[:5], decode_error=err)
    res["ok"] = bool(res["packets_src"] == res["packets_dst"] and res["pts_equal"] and res["copied_identical"]
                     and not bad and err is None)
    return res


def _count_frames(path: str) -> Tuple[int, Optional[str]]:
    import av
    n, err = 0, None
    try:
        with av.open(str(path)) as c:
            c.streams.video[0].thread_type = "AUTO"
            for _ in c.decode(video=0):
                n += 1
    except Exception as e:
        err = repr(e)[:200]
    return n, err


def verify(src: str, dst: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Full check: same packet count and pts, the output decodes without error to as many
    frames as the source, and (smart mode) every copied packet is byte-identical."""
    ps, _, _, src_tb = _packets(src)
    pd_, _, _, _ = _packets(dst, src_tb)
    exp = _expected_pts(ps, stats)
    res: Dict[str, Any] = dict(packets_src=len(ps), packets_dst=len(pd_),
                               pts_equal=exp == [p[0] for p in pd_])
    same = sum(1 for a, b in zip(ps, pd_) if a[1] == b[1])
    res["packets_identical"] = same
    res["packets_reencoded_expected"] = stats.get("packets_reencoded")
    n_dec, err = _count_frames(dst)
    n_src, err_src = _count_frames(src)
    res.update(frames_decoded_dst=n_dec, decode_error=err, frames_decoded_src=n_src, source_decode_error=err_src)
    res["ok"] = bool(res["packets_src"] == res["packets_dst"] and res["pts_equal"] and err is None and n_dec == n_src
                     and (stats["mode"] == "full" or same == len(ps) - stats["packets_reencoded"]))
    return res


def verify_pair(src: str, dst: str) -> Dict[str, Any]:
    """Check a masked file against its source without the statistics of the cut.

    The output must have as many packets as the source, decode without error to as many
    frames, and carry the source pts: packet by packet (a smart cut, where a pts may be one
    tick above the previous one where the source repeats a timestamp) or as a sorted list (a
    full re-encode). Reports how many packets are byte-identical and the packet ranges that
    changed (``changed_runs``); in a smart cut every changed run must start at a source
    keyframe (``runs_start_at_keyframes``).
    """
    import av

    ps, _, _, src_tb = _packets(src)
    pd_, _, _, _ = _packets(dst, src_tb)
    with av.open(str(src)) as c:
        keys = [bool(p.is_keyframe) for p in c.demux(c.streams.video[0]) if p.size > 0]
    sp, dp = [a[0] for a in ps], [b[0] for b in pd_]
    res: Dict[str, Any] = dict(packets_src=len(ps), packets_dst=len(pd_))
    exact = len(sp) == len(dp) and all(
        x == y or (i > 0 and x is not None and sp[i - 1] is not None and x <= sp[i - 1] and y == dp[i - 1] + 1)
        for i, (x, y) in enumerate(zip(sp, dp)))
    res["pts_equal"] = exact
    res["pts_equal_sorted"] = sorted(sp, key=lambda x: (x is None, x)) == sorted(dp, key=lambda x: (x is None, x))
    same = [a[1] == b[1] for a, b in zip(ps, pd_)]
    res["packets_identical"] = sum(same)
    runs: List[List[int]] = []
    for i, s in enumerate(same):
        if s:
            continue
        if runs and runs[-1][1] == i:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    res["changed_runs"] = runs[:50]
    res["mode"] = "full" if res["packets_identical"] == 0 and ps else "smart"
    res["runs_start_at_keyframes"] = all(keys[a] for a, _ in runs) if res["mode"] == "smart" else None
    n_dec, err = _count_frames(dst)
    n_src, err_src = _count_frames(src)
    res.update(frames_decoded_dst=n_dec, decode_error=err, frames_decoded_src=n_src, source_decode_error=err_src)
    pts_ok = exact if res["mode"] == "smart" else (exact or res["pts_equal_sorted"])
    res["ok"] = bool(len(ps) == len(pd_) and pts_ok and err is None and n_dec == n_src
                     and res["runs_start_at_keyframes"] is not False)
    return res


# ---------------------------------------------------------------------------------------
# files, results, resume
# ---------------------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = 16 << 20) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_copy(src: str, dst: str, chunk: int = 16 << 20) -> Tuple[str, int]:
    """Copy ``src`` to ``dst`` (written in place) and return ``(sha256 hex, bytes)`` of what was
    written, hashed in the same pass."""
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(chunk)
            if not b:
                break
            h.update(b)
            fo.write(b)
    return h.hexdigest(), os.path.getsize(dst)


def write_json_atomic(path: str, obj: Any) -> None:
    """Write JSON to ``path`` through a temporary file and a rename."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def result_ok(path: str) -> bool:
    """Whether a result file exists and says ``ok`` (a finished item is skipped on resume)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("ok"))
    except (ValueError, OSError):
        return False


def _has_boxes(plan: Any) -> bool:
    return bool(list(plan.intervals()))


def mask_video(src: str, table: Any, dst: str, cfg: Any = None, full_reencode: bool = False,
               plan: Any = None, log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Mask one video and check the result.

    ``table`` is a :class:`~openveil.table.BoxTable`, a table dict or a path. A video without
    any box after the mask rules needs no change: it is copied byte for byte
    (``unchanged: true``; nothing is written when ``dst`` is ``src``). Otherwise it is smart-cut (or re-encoded in full when
    ``full_reencode`` is set or the source is not eligible) and checked
    (``video.verify``); if the cut fails or its check fails and
    ``video.fallback_full_reencode`` is on, the video is re-encoded in full and checked
    again. A result in which no frame was mosaicked although the plan has boxes fails (the
    table does not fit this video). ``dst`` is written only when the check passes. Returns a JSON-ready result dict
    (numbers and file names only) with ``ok``, ``mode``, ``attemptN`` (stats and check, or
    the error of a cut that failed), ``sha256`` and ``bytes``.
    """
    cfg = _load_cfg(cfg)
    t0 = time.time()
    r: Dict[str, Any] = dict(video=os.path.basename(str(src)), output=os.path.basename(str(dst)), ok=False)
    table = _load_table(table)
    plan = plan if plan is not None else _make_plan(table, cfg)
    r["rule"] = _rule_stats(plan)
    if not _has_boxes(plan):
        if os.path.exists(dst) and os.path.samefile(src, dst):
            r["sha256"], r["bytes"] = sha256_file(src), os.path.getsize(src)     # in place: nothing to write
        else:
            tmp_copy = str(dst) + ".openveil-tmp"
            try:
                r["sha256"], r["bytes"] = sha256_copy(src, tmp_copy)
                os.replace(tmp_copy, dst)
            finally:
                if os.path.exists(tmp_copy):
                    os.remove(tmp_copy)
        r.update(ok=True, unchanged=True, mode="copy", seconds=round(time.time() - t0, 1))
        return r
    tmp = str(dst) + ".openveil-tmp.mp4"
    attempts = [bool(full_reencode)]
    if not full_reencode and cfg.video.fallback_full_reencode:
        attempts.append(True)
    ver: Dict[str, Any] = {"ok": False}
    st: Dict[str, Any] = {}
    try:
        for i, full in enumerate(attempts, 1):
            try:
                st = smart_cut(src, table, tmp, cfg, full=full, plan=plan, log=log)
            except Exception as e:
                if i == len(attempts):
                    raise
                r[f"attempt{i}"] = dict(error=repr(e)[:300])     # e.g. a decoder error: try in full
                continue
            if cfg.video.verify == "fast":
                ver = verify_fast(src, tmp, st)
            elif cfg.video.verify == "full":
                ver = verify(src, tmp, st)
            else:
                ver = {"ok": True, "skipped": True}
            r[f"attempt{i}"] = dict(stats={k: v for k, v in st.items() if k not in _STAT_DETAIL_KEYS}, verify=ver)
            if ver["ok"] and not st.get("frames_mosaicked"):
                ver = dict(ver, ok=False, why="the plan has boxes but no frame was mosaicked: the table does not "
                                              "fit this video")
                r[f"attempt{i}"]["verify"] = ver
            if ver["ok"] or st["mode"] == "full":
                break
        r["ok"] = bool(ver["ok"])
        r["mode"] = st.get("mode")
        if r["ok"]:
            d = os.path.dirname(os.path.abspath(str(dst)))
            os.makedirs(d, exist_ok=True)
            os.replace(tmp, dst)
            r["sha256"], r["bytes"] = sha256_file(dst), os.path.getsize(dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    r["seconds"] = round(time.time() - t0, 1)
    return r


def mask_videos(jobs: Iterable[Dict[str, Any]], results_dir: str, cfg: Any = None, worker: int = 0,
                n_workers: int = 1, log: Optional[Callable[[str], None]] = print) -> List[Dict[str, Any]]:
    """Mask a batch of videos with resume.

    ``jobs``: dicts with ``name`` (unique, used for the result file), ``src``, ``table`` (path
    or table) and ``dst``, optionally ``cost`` (bigger jobs first) and ``full_reencode``.
    Worker ``worker`` of ``n_workers`` takes every ``n_workers``-th job of the cost-sorted list.
    The result of each job is written to ``<results_dir>/<name>.json``; a job whose result
    says ``ok`` is skipped when the batch is run again. A job that raises is recorded with
    ``ok: false`` and the error, and the batch goes on.
    """
    cfg = _load_cfg(cfg)
    jobs = sorted(jobs, key=lambda j: -float(j.get("cost", 0) or 0))
    mine = [j for i, j in enumerate(jobs) if i % n_workers == worker]
    os.makedirs(results_dir, exist_ok=True)
    out = []
    for i, j in enumerate(mine):
        rp = os.path.join(results_dir, f"{j['name']}.json")
        if result_ok(rp):
            continue
        try:
            r = mask_video(j["src"], j["table"], j["dst"], cfg, full_reencode=bool(j.get("full_reencode")))
        except Exception as e:
            r = dict(video=os.path.basename(str(j["src"])), ok=False, error=repr(e)[:300],
                     trace=traceback.format_exc()[-1500:])
        r["name"] = j["name"]
        write_json_atomic(rp, r)
        out.append(r)
        if log:
            a1 = (r.get("attempt1") or {}).get("stats", {})
            log(f"[w{worker}] {i + 1}/{len(mine)} {j['name']}: ok={r.get('ok')} mode={r.get('mode', '-')} "
                f"reencoded={a1.get('packets_reencoded', '-')} s={r.get('seconds', '-')} {r.get('error', '')}")
    return out
