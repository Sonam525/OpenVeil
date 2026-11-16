# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Video input: decode frames in a child process, and read basic stream facts.

Why a child process
-------------------
FFmpeg decoders can hang inside C code (for example when a frame-threaded HEVC decoder is
closed while its threads still run). A hang in C code cannot be interrupted from Python in
the same process, so a long batch job would block forever. :class:`FrameStream` therefore
decodes in a child Python process that sends the frames back through a pipe. The parent
kills the child when no frame arrives for ``timeout`` seconds (``first_timeout`` before the
first frame, which includes opening and seeking) and raises :class:`DecodeStall`; the caller
can then retry the window. The child never closes the container: it sends an end marker and
leaves with ``os._exit``, because closing the container is where decoders were seen to hang.
Decoding in a child costs almost nothing, as the frames stream back while the parent works on
the previous batch.

Usage::

    # detector frames (BGR, every 5th frame of a 60 s window)
    for idx, t_s, bgr in stream_frames(path, start_s=0, dur_s=60, stride=5):
        ...                      # bgr is None for the frames that were not selected

    # I420 frames with their source pts
    s = FrameStream(path, 0, 60, fmt="i420")
    for idx, t_s, buf, pts, full_range in s:
        ...                      # buf: (H * 3 / 2) x W uint8 I420 buffer
    s.meta                       # after the first frame: width, height, time_base, color, ...

``idx`` is the index of the frame inside the window, ``t_s`` the seconds since the first
frame of the window. Frames whose pts does not increase are dropped (up to 50 per window by
default):
some cameras write two frames with the same timestamp at the start of a file.

Only numpy and PyAV (``av``) are needed, and both are imported lazily.
"""
from __future__ import annotations

import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

__all__ = [
    "DecodeStall", "FrameStream", "stream_frames", "VideoInfo", "probe",
    "frame_to_i420", "DEFAULT_ERR_LOG",
]

# Message header between child and parent: idx (-1 end marker, -2 metadata, -3 error),
# time_s, payload bytes, height, width, flag (full range), pts.
_HDR = struct.Struct("<qdqiiiq")
_END, _META, _ERROR = -1, -2, -3
_MAX_DROPPED = 50

#: Where the child's stderr (FFmpeg messages, Python tracebacks) is appended by default.
DEFAULT_ERR_LOG = os.path.join(tempfile.gettempdir(), "openveil_decode_stderr.log")

_FORMATS = ("bgr24", "rgb24", "i420")


class DecodeStall(RuntimeError):
    """The decoder child sent nothing for longer than the allowed timeout."""


def _frac(x: Any) -> Fraction:
    """``x`` as an exact Fraction (strings and floats are read through their decimal text)."""
    return x if isinstance(x, Fraction) else Fraction(str(x))


def frame_to_i420(frame) -> Tuple["Any", bool]:
    """Planar I420 buffer of a decoded ``av.VideoFrame`` and whether it is full range.

    Returns ``(buf, full_range)`` where ``buf`` is a contiguous ``(H * 3 // 2) x W`` uint8
    array (Y plane, then U, then V). ``yuv420p`` and ``yuvj420p`` frames are returned as they
    are (no conversion); any other pixel format is converted to limited-range ``yuv420p``.
    """
    import numpy as np

    name = frame.format.name
    if name in ("yuv420p", "yuvj420p"):
        buf = frame.to_ndarray()
        full = name == "yuvj420p" or int(frame.color_range) == 2
    else:
        buf = frame.reformat(format="yuv420p").to_ndarray()
        full = False
    return np.ascontiguousarray(buf), full


# ---------------------------------------------------------------------------------------
# child process
# ---------------------------------------------------------------------------------------
def _child(spec: Dict[str, Any]) -> None:
    """Decode the window described by ``spec`` and write frames to stdout (child side)."""
    out = sys.stdout.buffer
    try:
        _child_decode(spec, out)
    except BaseException as e:  # report the error to the parent, then leave
        msg = repr(e)[:2000].encode("utf-8", "replace")
        try:
            out.write(_HDR.pack(_ERROR, 0.0, len(msg), 0, 0, 0, 0))
            out.write(msg)
            out.flush()
        except Exception:
            pass
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    out.write(_HDR.pack(_END, 0.0, 0, 0, 0, 0, 0))
    out.flush()
    os._exit(0)                     # no container.close(): that is where decoders were seen to hang


def _child_decode(spec: Dict[str, Any], out) -> None:
    import numpy as np
    import av

    av.logging.set_level(av.logging.PANIC)          # nothing from FFmpeg threads into Python
    if spec.get("delay_s"):                          # test hook: simulate a slow start
        time.sleep(float(spec["delay_s"]))
    fmt = spec.get("fmt", "bgr24")
    wanted = set(spec["wanted"]) if spec.get("wanted") is not None else None
    stride, stop_after = spec.get("stride"), spec.get("stop_after")
    keep_last = bool(spec.get("keep_last")) and wanted is None and bool(stride)
    held = None                                      # (idx, t_s, frame, pts) not yet written (keep_last)
    cont = av.open(spec["path"], mode="r")
    st = cont.streams.video[0]
    st.thread_type = "AUTO"
    if spec.get("threads"):
        st.codec_context.thread_count = int(spec["threads"])
    tb = st.time_base
    p0 = st.start_time if st.start_time is not None else 0
    start_s = _frac(spec["start_s"])
    start_pts = p0 + start_s / tb
    end_pts = None if spec.get("dur_s") is None else p0 + (start_s + _frac(spec["dur_s"])) / tb
    if start_s > 0:
        cont.seek(int(start_pts // 1), stream=st, backward=True, any_frame=False)
    last, t_first, idx, dropped = None, None, -1, 0
    for fr in cont.decode(st):
        pts = fr.pts
        if pts is None:
            raise RuntimeError("decoded frame without pts")
        if pts < start_pts:
            continue
        if end_pts is not None and pts >= end_pts:
            break
        if last is not None and pts <= last:
            dropped += 1                    # repeated or decreasing timestamp: keep the first frame
            if dropped > int(spec.get("max_bad_pts", _MAX_DROPPED)):
                raise RuntimeError(f"non-increasing pts {last} -> {pts} ({dropped} frames dropped)")
            continue
        last = pts
        idx += 1
        if t_first is None:
            t_first = pts
            meta = dict(width=fr.width, height=fr.height, time_base=[tb.numerator, tb.denominator],
                        color=dict(colorspace=int(fr.colorspace), color_primaries=int(fr.color_primaries),
                                   color_trc=int(fr.color_trc), color_range=int(fr.color_range)),
                        full_range=bool(fr.format.name == "yuvj420p" or int(fr.color_range) == 2),
                        first_pts=int(pts), pix_fmt=fr.format.name)
            mb = json.dumps(meta).encode()
            out.write(_HDR.pack(_META, 0.0, len(mb), 0, 0, 0, 0))
            out.write(mb)
        t_s = float((pts - t_first) * tb)
        take = ((wanted is not None and idx in wanted) or (stride and idx % stride == 0)
                or (wanted is None and not stride))
        if keep_last:
            # the frame is written once the next one is known, so the last frame of the window
            # can still be delivered with pixels
            if held is not None:
                _emit(out, held, False, fmt)
                held = None
            if take:
                _emit(out, (idx, t_s, fr, pts), True, fmt)
            else:
                held = (idx, t_s, fr, pts)
        else:
            _emit(out, (idx, t_s, fr, pts), take, fmt)
        if stop_after is not None and idx >= stop_after:
            break
    if held is not None:
        _emit(out, held, True, fmt)


def _emit(out, item: tuple, take: bool, fmt: str) -> None:
    import numpy as np

    idx, t_s, fr, pts = item
    if take:
        if fmt == "i420":
            a, full = frame_to_i420(fr)
        else:
            a, full = np.ascontiguousarray(fr.to_ndarray(format=fmt)), False
        out.write(_HDR.pack(idx, t_s, a.nbytes, a.shape[0], a.shape[1], int(full), int(pts)))
        out.write(a.data)
    else:
        out.write(_HDR.pack(idx, t_s, 0, 0, 0, 0, int(pts)))


# ---------------------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------------------
def _read_exact(f, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class FrameStream:
    """Iterator over the frames of a time window, decoded by a child process.

    Yields ``(idx, t_s, array_or_None, pts, full_range)`` per decoded frame.

    Parameters
    ----------
    path:
        Video file.
    start_s, dur_s:
        Window start and length in seconds from the start of the stream; ``dur_s=None``
        decodes to the end. Numbers are converted exactly (through their decimal text).
    stride:
        Deliver every ``stride``-th frame (``idx % stride == 0``).
    wanted:
        Deliver only these frame indices.
    stop_after:
        Stop after frame ``stop_after`` of the window.
    keep_last:
        With ``stride``: deliver the last frame of the window with pixels too, whatever its index
        (a sparse scan uses it as a checkpoint).
        Without ``stride`` and ``wanted`` every frame is delivered. Frames that are not
        delivered still yield ``(idx, t_s, None, pts, False)``.
    threads:
        Decoder threads (``None`` keeps the FFmpeg default).
    fmt:
        ``"bgr24"`` or ``"rgb24"`` (H x W x 3 arrays) or ``"i420"`` ((H * 3 / 2) x W planar).
    first_timeout, timeout:
        Seconds to wait for the first message (includes the seek) and between messages
        afterwards; :class:`DecodeStall` is raised when they run out.
    err_log:
        File the child's stderr is appended to.
    queue_size:
        Frames buffered between the reader thread and the consumer.
    max_bad_pts:
        Frames with a repeated or decreasing pts that are dropped before the window fails.

    After the first frame, ``meta`` holds ``width``, ``height``, ``time_base`` ``[num, den]``,
    ``color`` (colorspace, primaries, transfer, range as FFmpeg enum values),
    ``full_range``, ``first_pts`` and ``pix_fmt``.
    """

    def __init__(self, path: str, start_s: Any = 0, dur_s: Any = None, stride: Optional[int] = None,
                 wanted: Optional[Iterable[int]] = None, stop_after: Optional[int] = None,
                 threads: Optional[int] = 6, fmt: str = "bgr24", first_timeout: float = 300.0,
                 timeout: float = 120.0, err_log: Optional[str] = None, queue_size: int = 16,
                 max_bad_pts: int = _MAX_DROPPED, _delay_s: float = 0.0, keep_last: bool = False):
        if fmt not in _FORMATS:
            raise ValueError(f"fmt must be one of {_FORMATS}, not {fmt!r}")
        self.path, self.start_s, self.fmt = str(path), start_s, fmt
        self.first_timeout, self.timeout = float(first_timeout), float(timeout)
        self.meta: Optional[Dict[str, Any]] = None
        self.err_log = err_log or DEFAULT_ERR_LOG
        spec = dict(path=self.path, start_s=str(start_s), dur_s=None if dur_s is None else str(dur_s),
                    stride=stride, wanted=sorted(int(i) for i in wanted) if wanted is not None else None,
                    stop_after=stop_after, threads=threads, fmt=fmt, max_bad_pts=int(max_bad_pts),
                    delay_s=_delay_s, keep_last=bool(keep_last))
        d = os.path.dirname(os.path.abspath(self.err_log))
        os.makedirs(d, exist_ok=True)
        self._err = open(self.err_log, "ab")
        # The child runs this file as a script, so it needs neither the package nor its
        # dependencies beyond numpy and av.
        self._proc = subprocess.Popen([sys.executable, "-u", "-B", os.path.abspath(__file__), "--child",
                                       json.dumps(spec)],
                                      stdout=subprocess.PIPE, stderr=self._err, bufsize=0)
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._closed = False
        threading.Thread(target=self._reader, daemon=True).start()

    # reader thread -------------------------------------------------------------------
    def _put(self, item: tuple) -> None:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def _reader(self) -> None:
        try:
            while True:
                hdr = _read_exact(self._proc.stdout, _HDR.size)
                if hdr is None:
                    self._put(("eof",))
                    return
                idx, t_s, nb, h, w, flag, pts = _HDR.unpack(hdr)
                payload = _read_exact(self._proc.stdout, nb) if nb else None
                if nb and payload is None:
                    self._put(("eof",))
                    return
                self._put((idx, t_s, payload, h, w, flag, pts))
                if idx in (_END, _ERROR):
                    return
        except Exception as e:                     # pipe closed by the kill
            self._put(("err", repr(e)))

    # consumer --------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[int, float, Any, int, bool]]:
        import numpy as np

        try:
            first = True
            while True:
                wait = self.first_timeout if first else self.timeout
                try:
                    item = self._q.get(timeout=wait)
                except queue.Empty:
                    raise DecodeStall(f"no frame from the decoder for {wait:g} s: {self.path} @ {self.start_s}")
                first = False
                if item[0] == "eof":
                    raise RuntimeError(f"decoder process ended without the end marker "
                                       f"(return code {self._proc.poll()}), see {self.err_log}")
                if item[0] == "err":
                    raise RuntimeError(item[1])
                idx, t_s, payload, h, w, flag, pts = item
                if idx == _END:
                    return
                if idx == _ERROR:
                    raise RuntimeError(f"decoder failed on {self.path}: "
                                       f"{payload.decode('utf-8', 'replace') if payload else ''}")
                if idx == _META:
                    self.meta = json.loads(payload.decode())
                    continue
                if payload is None:
                    arr = None
                elif self.fmt == "i420":
                    arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w)
                else:
                    arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
                yield idx, t_s, arr, pts, bool(flag)
        finally:
            self.close()

    def close(self) -> None:
        """Kill the child (if still running) and release the log file."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._proc.kill()
            self._proc.wait(timeout=10)
        except Exception:
            pass
        try:
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        except Exception:
            pass
        try:
            self._err.close()
        except Exception:
            pass

    def __enter__(self) -> "FrameStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def stream_frames(path: str, start_s: Any = 0, dur_s: Any = None, stride: Optional[int] = None,
                  wanted: Optional[Iterable[int]] = None, stop_after: Optional[int] = None,
                  threads: Optional[int] = 6, fmt: str = "bgr24", first_timeout: float = 300.0,
                  timeout: float = 120.0, err_log: Optional[str] = None,
                  queue_size: int = 16, max_bad_pts: int = _MAX_DROPPED) -> Iterator[Tuple[int, float, Any]]:
    """Detector frames: yields ``(idx, t_s, array_or_None)``; see :class:`FrameStream`."""
    for idx, t_s, arr, _pts, _full in FrameStream(path, start_s, dur_s, stride=stride, wanted=wanted,
                                                  stop_after=stop_after, threads=threads, fmt=fmt,
                                                  first_timeout=first_timeout, timeout=timeout,
                                                  err_log=err_log, queue_size=queue_size,
                                                  max_bad_pts=max_bad_pts):
        yield idx, t_s, arr


# ---------------------------------------------------------------------------------------
# stream facts
# ---------------------------------------------------------------------------------------
@dataclass
class VideoInfo:
    """Facts about the first video stream of a file, read from the container headers.

    ``fps`` is the average frame rate (``None`` if the container does not state one),
    ``n_frames`` the frame count stated by the container (0 if unknown) and ``duration_s``
    the stated stream duration (``None`` if unknown). ``start_pts`` is the stream start time
    in ``time_base`` units. Nothing is decoded.
    """
    path: str
    container: str
    codec: str
    width: int
    height: int
    pix_fmt: Optional[str]
    time_base: Fraction
    average_rate: Optional[Fraction]
    fps: Optional[float]
    duration_s: Optional[float]
    n_frames: int
    start_pts: int
    bit_rate: Optional[int]
    profile: Optional[str]
    n_streams: int
    stream_types: Tuple[str, ...]
    extradata: bytes = field(default=b"", repr=False)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (fractions as ``[num, den]``, no extradata bytes)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extradata"}
        d["time_base"] = [self.time_base.numerator, self.time_base.denominator]
        d["average_rate"] = (None if self.average_rate is None
                             else [self.average_rate.numerator, self.average_rate.denominator])
        d["stream_types"] = list(self.stream_types)
        d["extradata_len"] = len(self.extradata)
        return d


def probe(path: str) -> VideoInfo:
    """Read :class:`VideoInfo` for the first video stream of ``path`` (headers only)."""
    import av

    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError(f"no video stream in {path}")
        st = c.streams.video[0]
        cc = st.codec_context
        rate = st.average_rate
        dur = float(st.duration * st.time_base) if st.duration else None
        if dur is None and c.duration:
            dur = c.duration / 1_000_000.0          # container duration in AV_TIME_BASE units
        return VideoInfo(
            path=str(path), container=c.format.name, codec=cc.name, width=int(cc.width),
            height=int(cc.height), pix_fmt=cc.format.name if cc.format else None,
            time_base=Fraction(st.time_base.numerator, st.time_base.denominator),
            average_rate=Fraction(rate.numerator, rate.denominator) if rate else None,
            fps=float(rate) if rate else None, duration_s=dur, n_frames=int(st.frames or 0),
            start_pts=int(st.start_time or 0), bit_rate=int(cc.bit_rate) if cc.bit_rate else None,
            profile=cc.profile, n_streams=len(c.streams),
            stream_types=tuple(s.type for s in c.streams), extradata=bytes(cc.extradata or b""))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]))
    else:
        # self-test:  python video.py <video> [start_s] [dur_s] [stride]
        p = sys.argv[1]
        s0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        d0 = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        t0, n, ns = time.time(), 0, 0
        for _i, _t, a in stream_frames(p, s0, d0, stride=k):
            n += 1
            ns += a is not None
        print(f"{n} frames, {ns} selected, {time.time() - t0:.1f} s")
