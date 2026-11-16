# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Detection over whole videos: one box table per video.

:func:`detect_video` processes one video:

1. the video is cut into chunks of ``schedule.chunk_s`` seconds; the day/night policy gives each
   chunk a scan mode (``full``, ``sparse`` or ``skip``; default always ``full``);
2. each chunk is decoded in a child process (:class:`openveil.io.video.FrameStream`, killed and
   retried with fewer threads when it stalls) while the previous chunk is being detected;
3. the scan schedule (:func:`openveil.schedule.scan_chunk`) decides which frames the detector
   runs on: a checkpoint every ``base_s``, dense every ``dense_s`` around a hit; at night only
   checkpoints every ``night_base_s``;
4. on each of those frames the proposer suggests boxes, :func:`openveil.rules.candidates` keeps
   the candidates, the verifier scores them, boxes with a score >= the verifier threshold are
   kept and :func:`openveil.rules.postprocess` removes duplicates and loose group boxes. Any
   kept box is a hit;
5. gaps between detections of the same person are interpolated, and the boxes of every frame
   are written to the box table (source pixels, rounded to 0.1 px; detections keep their
   verifier score).

In ``full`` chunks every decoded frame stays in memory until the chunk is done (the schedule
may need any of them), so memory grows with ``chunk_s * fps * width * height * 3`` bytes; lower
``schedule.chunk_s`` if that is too much.

:func:`run_many` processes a list of videos with several worker processes that share one GPU:

* videos are split over the workers by cost (frames), largest first;
* finished tables are skipped, so a run can be started again after any interruption;
* a failed video is recorded in a failure ledger (``failed/<name>.json``) and retried, once at the
  end of the worker's pass and again when the worker is restarted; after ``run.max_attempts``
  failures it is abandoned (reported, not retried);
* a worker that sends no progress message for ``run.stall_s`` seconds while it works on a video
  is killed; a worker that exits with videos left is restarted after ``run.restart_wait_s``
  seconds; a video that kills its worker in three launches in a row is quarantined (marked as
  failed ``max_attempts`` times). Only the video a worker was working on when it died is
  blamed: a worker that dies before it reaches a video (for example because the models cannot
  be loaded) is restarted, and after three such launches its slot is given up without touching
  the failure ledger;
* missing model packages and an incomplete verifier config are reported before any worker
  starts (with the default model factory); ``retry_failed=True`` clears the failure ledger of
  the listed videos first;
* GPU use is sampled with ``nvidia-smi`` when it is available;
* ``summary.json`` lists the videos that are missing, quarantined or abandoned.

Proposer and verifier are any objects with this interface (so tests and other models can be
plugged in):

* ``proposer(frames_bgr) -> [[[x1, y1, x2, y2, score], ...] per frame]``;
* ``verifier.threshold``, ``verifier.score_frames(frames_bgr, boxes_per_frame) -> [scores per frame]``.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = ["detect_video", "run_many", "default_factory", "make_detect", "video_name", "table_path", "is_done",
           "attempts", "note_failure", "partition", "Ledger", "config_summary", "preflight", "SetupError"]

COORD_DECIMALS = 1           # box coordinates are rounded to 0.1 px
SCORE_DECIMALS = 4


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def video_name(path: str) -> str:
    """Table name of a video: the file name without its extension."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def table_path(out_dir: str, name: str) -> str:
    """Where the box table of video ``name`` is written."""
    return os.path.join(out_dir, f"{name}.json.gz")


def _cfg(cfg: Any):
    from ..config import load_config  # noqa: PLC0415

    return load_config(cfg)


def default_factory(cfg: Any):
    """``(proposer, verifier)`` built from a config (Grounding DINO + crop verifier, or mode none)."""
    from .proposer import make_proposer  # noqa: PLC0415
    from .verifier import make_verifier  # noqa: PLC0415

    cfg = _cfg(cfg)
    device = cfg.run.device
    return make_proposer(cfg.proposer, device=device), make_verifier(cfg.verifier, device=device)


class SetupError(RuntimeError):
    """The models cannot be built or run here (a missing package, an incomplete config): not the
    fault of any video, so it never goes to the failure ledger."""


def preflight(cfg: Any) -> None:
    """Check, before any worker starts, what the default model factory needs.

    Raises :class:`SetupError` when the verifier is in ``crop`` mode without weights, or when
    torch / transformers (or peft for the crop verifier) are not installed.
    """
    import importlib.util  # noqa: PLC0415

    cfg = _cfg(cfg)
    v = cfg.verifier
    if str(v.mode).lower() == "crop" and not (v.adapter_dir or v.head_path):
        raise SetupError("verifier mode 'crop' needs verifier.adapter_dir (or verifier.head_path) in the config, for "
                         "example the folder of the released weights Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16 downloaded "
                         "from the Hugging Face Hub; set verifier.mode to 'none' to run the proposer alone")
    need = ["torch", "transformers"] + (["peft", "safetensors"] if str(v.mode).lower() == "crop" else [])
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        raise SetupError(f"missing package(s) {', '.join(missing)}: install them with  pip install \"openveil[detect]\"")


def _detector_params(cfg) -> dict:
    """The settings stored in a table's ``params`` (flat production keys + the full detector sections).

    No local paths are written: the verifier's weight folder is recorded by its folder name only.
    """
    from ..config import to_dict  # noqa: PLC0415

    s, dn = cfg.schedule, cfg.schedule.day_night
    v = to_dict(cfg.verifier)
    for k in ("adapter_dir", "head_path"):
        if v.get(k):
            v[k] = os.path.basename(os.path.normpath(str(v[k])))
    try:
        from .. import __version__ as version  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        version = None
    return dict(base_s=s.base_s, dense_s=s.dense_s, night_base_s=s.night_base_s, clear_gaps=s.clear_gaps,
                lookahead=s.lookahead, max_gap_s=s.max_gap_s, day_policy=dn.policy, day_start=dn.day_start,
                day_end=dn.day_end, night_mode=dn.night_mode, mask_thr=cfg.verifier.threshold,
                cand_thr=cfg.proposer.cand_thr, nms=cfg.postprocess.nms, container=cfg.postprocess.container,
                chunk_s=s.chunk_s, detector=dict(schedule=to_dict(s), proposer=to_dict(cfg.proposer), verifier=v,
                                                 postprocess=to_dict(cfg.postprocess)),
                openveil=version)


# ---------------------------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------------------------
def _decode_chunk(path: str, s0: float, d0: float | None, stride: int | None, dcfg) -> tuple:
    """``(frames {idx: bgr}, times [s], pts [int], meta, seconds, stalls)`` of one window.

    With ``stride`` only every ``stride``-th frame and the last frame of the window carry
    pixels. A stalled decoder is retried once with ``dcfg.retry_threads`` threads.
    """
    from ..io.video import DecodeStall, FrameStream  # noqa: PLC0415

    stalls = 0
    tries = (dcfg.threads, dcfg.retry_threads)
    for attempt, thr in enumerate(tries, 1):
        t0 = time.time()
        frames, times, pts = {}, [], []
        fs = FrameStream(path, s0, d0, stride=stride, threads=thr, fmt="bgr24", first_timeout=dcfg.first_frame_timeout_s,
                         timeout=dcfg.stall_timeout_s, max_bad_pts=dcfg.max_bad_pts, keep_last=bool(stride))
        try:
            for idx, t_s, arr, p, _full in fs:
                times.append(t_s)
                pts.append(int(p))
                if arr is not None:
                    frames[idx] = arr
        except DecodeStall:
            stalls += 1
            if attempt == len(tries):
                raise
            continue
        return frames, times, pts, fs.meta, time.time() - t0, stalls
    raise RuntimeError("unreachable")  # pragma: no cover


def _first_frame_meta(path: str, dcfg, start_s: float = 0) -> dict | None:
    """Stream metadata of the first frame at or after ``start_s`` (decodes one frame, no pixels);
    ``None`` when there is no such frame."""
    from ..io.video import FrameStream  # noqa: PLC0415

    fs = FrameStream(path, start_s, None, wanted=[], stop_after=0, threads=dcfg.retry_threads, fmt="bgr24",
                     first_timeout=dcfg.first_frame_timeout_s, timeout=dcfg.stall_timeout_s, max_bad_pts=dcfg.max_bad_pts)
    for _ in fs:
        pass
    return fs.meta


def _next_frame_s(path: str, start_s: float, dcfg, info) -> float | None:
    """Seconds (from the stream start) of the first frame at or after ``start_s``, or ``None``."""
    from fractions import Fraction  # noqa: PLC0415

    meta = _first_frame_meta(path, dcfg, start_s)
    if not meta or meta.get("first_pts") is None:
        return None
    return float((int(meta["first_pts"]) - int(info.start_pts or 0)) * Fraction(info.time_base))


# ---------------------------------------------------------------------------------------------
# one video
# ---------------------------------------------------------------------------------------------
def make_detect(frames: Mapping[int, Any], proposer, verifier, cfg, stats: collections.Counter) -> Callable:
    """The ``detect(frame indices)`` callback of the scan schedule for one decoded chunk.

    Returns ``{idx: [[x1, y1, x2, y2, score, 1.0], ...]}`` (the last field is the hit flag).
    """
    from ..rules import candidates, postprocess  # noqa: PLC0415

    pc, pp = cfg.proposer, cfg.postprocess
    thr = float(verifier.threshold)

    def detect(idxs: Sequence[int]) -> dict:
        imgs = [frames[i] for i in idxs]
        props = proposer(imgs)
        if len(props) != len(imgs):
            raise RuntimeError(f"the proposer returned {len(props)} results for {len(imgs)} frames")
        stats["detector_frames"] += len(idxs)
        cands = [candidates(p, pc.cand_thr, pc.merge_iou) for p in props]
        need = [k for k in range(len(idxs)) if cands[k]]
        scores = verifier.score_frames([imgs[k] for k in need], [cands[k] for k in need]) if need else []
        kept: list[list] = [[] for _ in idxs]
        n_crops = 0
        for k, sc in zip(need, scores):
            n_crops += len(cands[k])
            for c, p in zip(cands[k], sc):
                if p >= thr:
                    kept[k].append([c[0], c[1], c[2], c[3], float(p)])
        stats["candidates"] += n_crops
        out = {}
        for k, i in enumerate(idxs):
            boxes = postprocess(kept[k], pp.nms, pp.container, pp.container_min_inner, pp.container_ratio,
                                pp.container_inner_frac)
            out[i] = [[b[0], b[1], b[2], b[3], b[4], 1.0] for b in boxes]
        return out

    return detect


def _row_box(b: Sequence[Any]) -> list:
    row = [round(float(b[0]), COORD_DECIMALS), round(float(b[1]), COORD_DECIMALS), round(float(b[2]), COORD_DECIMALS),
           round(float(b[3]), COORD_DECIMALS), b[4]]
    if len(b) > 5 and b[5] is not None:
        row.append(round(min(1.0, max(0.0, float(b[5]))), SCORE_DECIMALS))
    return row


def detect_video(path: str, cfg: Any = None, proposer=None, verifier=None, *, name: str | None = None,
                 out_path: str | None = None, start_local: str | None = None, progress: Callable[[dict], None] | None = None,
                 stats: collections.Counter | None = None, fps: float | None = None):
    """Detect people in one video and return its :class:`openveil.table.BoxTable`.

    ``cfg``: an :class:`openveil.config.OpenVeilConfig`, a preset name, a file or ``None``
    (production preset). ``proposer`` / ``verifier`` default to the models of the config.
    ``name``: table name (default: the file name without extension). ``start_local``: camera
    time of the first frame (ISO); without it the ``clock`` day/night policy tries the file name
    (``day_night.clock_regex``) and otherwise scans every chunk in full. ``out_path``: write
    the table there as well. ``progress(info)`` is called after every chunk. ``fps``: the frame
    rate the scan strides are computed from (default: the container's average frame rate; when
    the container states none, the rate measured on the first decoded chunk).

    Coverage: the chunks follow the stated duration, and after them the video is probed once
    for frames beyond it (a duration that is too short). Without a stated duration an empty
    chunk is followed by a probe for the next frame (a recording pause), so no frame is left
    unscanned.
    """
    from ..io.video import probe  # noqa: PLC0415
    from ..schedule import chunk_mode, plan_chunks, scan_chunk, start_local_from_name  # noqa: PLC0415
    from ..table import BoxTable, save_table  # noqa: PLC0415

    cfg = _cfg(cfg)
    if proposer is None or verifier is None:
        p_def, v_def = default_factory(cfg)
        proposer = proposer if proposer is not None else p_def
        verifier = verifier if verifier is not None else v_def
    name = name or video_name(path)
    sc, dn, dcfg = cfg.schedule, cfg.schedule.day_night, cfg.decode
    if start_local is None and dn.policy == "clock":
        start_local = start_local_from_name(os.path.basename(path), dn.clock_regex)
    stats = stats if stats is not None else collections.Counter()
    t_start = time.time()
    info = probe(path)
    fps = float(fps) if fps else (float(info.fps) if info.fps else None)

    # chunk plan: the windows that cover the stated duration; after them (or without a stated
    # duration) the video goes on for as long as a probe finds another frame
    n_planned = len(plan_chunks(float(info.duration_s), sc.chunk_s)) if info.duration_s else 0
    pos = dict(k=0, open=not info.duration_s)

    def next_chunk():
        while True:
            k = pos["k"]
            if not pos["open"] and k >= n_planned:
                # the stated duration may be too short: look for a frame after the planned end
                t = _next_frame_s(path, n_planned * sc.chunk_s, dcfg, info)
                if t is None:
                    return None
                stats["chunks_beyond_duration"] += 1
                pos["open"] = True
                pos["k"] = max(k, int(t // sc.chunk_s))
                continue
            s0 = k * sc.chunk_s
            pos["k"] = k + 1
            mode = chunk_mode(start_local, s0, sc.chunk_s, dn)
            if mode == "skip":
                if pos["open"]:
                    # without a known end only a decoded chunk can show where the video ends, so a
                    # chunk that would be skipped is scanned sparsely instead
                    mode = "sparse"
                else:
                    stats["chunks_skipped"] += 1
                    continue
            return (s0, sc.chunk_s, mode)

    def stride_of(mode: str, rate: float | None) -> int | None:
        # with an unknown frame rate every frame is decoded with pixels, so the scan stride
        # (computed from the measured rate) always finds its frames
        if mode != "sparse" or not rate:
            return None
        return max(1, int(round(sc.night_base_s * rate)))

    rows: list[list] = []
    chunks: list[list] = []
    meta_out: dict | None = None
    first_meta: dict | None = None
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    pool = ThreadPoolExecutor(max_workers=1)

    def submit(c):
        return pool.submit(_decode_chunk, path, c[0], c[1], stride_of(c[2], fps), dcfg) if c else None

    try:
        cur = next_chunk()
        nxt_f = submit(cur)
        while cur is not None:
            s0, d0, mode = cur
            t_wait = time.time()
            frames, times, pts, meta, dec_s, stalls = nxt_f.result()
            stats["decode_s"] += dec_s
            stats["decode_wait_s"] += time.time() - t_wait
            stats["stalls"] += stalls
            if not times and pos["open"]:
                # an empty window of a video without a known end: a pause, or the end of the video
                t = _next_frame_s(path, s0 + d0, dcfg, info)
                if t is None:
                    break
                stats["empty_chunks"] += 1
                pos["k"] = max(pos["k"], int(t // sc.chunk_s))
                cur = next_chunk()
                nxt_f = submit(cur)
                continue
            if times and not fps:
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 25.0
            cur = next_chunk()
            nxt_f = submit(cur)
            if not times:
                continue
            if s0 == 0:
                first_meta = meta
            meta_out = meta_out or meta
            calls_before = stats["detector_frames"]
            detect = make_detect(frames, proposer, verifier, cfg, stats)
            fb, _ran = scan_chunk(len(times), times, detect, mode, fps, sc, keep_score=True)
            for idx, bxs in fb.items():
                rows.append([int(pts[idx]), [_row_box(b) for b in bxs]])
            stats["frames"] += len(times)
            stats["chunks"] += 1
            stats["frames_with_boxes"] += len(fb)
            stats["boxes"] += sum(len(b) for b in fb.values())
            chunks.append([round(s0, 1), mode, len(times), stats["detector_frames"] - calls_before])
            del frames
            if progress is not None:
                progress(dict(video=name, t=s0, mode=mode, chunks=len(chunks), detector_frames=stats["detector_frames"],
                              seconds=time.time() - t_start))
    finally:
        pool.shutdown(wait=False)
    if meta_out is None:
        meta_out = _first_frame_meta(path, dcfg) or {}
    elif first_meta is None:
        # the first chunk was skipped: times in the table still count from the first frame of the video
        first_meta = _first_frame_meta(path, dcfg) or meta_out
    first_meta = first_meta or meta_out
    rows.sort(key=lambda r: r[0])
    tb = meta_out.get("time_base") or [info.time_base.numerator, info.time_base.denominator]
    table = BoxTable(video=name, width=int(meta_out.get("width") or info.width), height=int(meta_out.get("height") or info.height),
                     fps=float(fps) if fps else None, kind="video", time_base=(int(tb[0]), int(tb[1])),
                     first_pts=int(first_meta.get("first_pts", info.start_pts)) if first_meta else int(info.start_pts),
                     start_local=start_local, params=_detector_params(cfg), chunks=chunks, frames=rows)
    table.validate()
    stats["video_s"] = time.time() - t_start
    if out_path:
        save_table(table, out_path)
    return table


# ---------------------------------------------------------------------------------------------
# ledger and resume
# ---------------------------------------------------------------------------------------------
class Ledger:
    """Done / failed bookkeeping of a run folder.

    ``<out>/<name>.json.gz`` is a finished table; ``<out>/failed/<name>.json`` is
    ``{video, attempts, last_error, at}`` for a video that failed.
    """

    def __init__(self, out_dir: str, max_attempts: int = 3):
        self.out_dir, self.max_attempts = out_dir, int(max_attempts)

    def failed_path(self, name: str) -> str:
        return os.path.join(self.out_dir, "failed", f"{name}.json")

    def attempts(self, name: str) -> int:
        try:
            with open(self.failed_path(name), encoding="utf-8") as f:
                return int(json.load(f)["attempts"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def note_failure(self, name: str, err: Any) -> int:
        """Record one more failed attempt; returns the new attempt count."""
        os.makedirs(os.path.join(self.out_dir, "failed"), exist_ok=True)
        n = self.attempts(name) + 1
        p = self.failed_path(name)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(dict(video=name, attempts=n, last_error=str(err)[:400], at=time.strftime("%Y-%m-%dT%H:%M:%S")), f)
        os.replace(p + ".tmp", p)
        return n

    def abandoned(self, name: str) -> bool:
        return self.attempts(name) >= self.max_attempts

    def is_done(self, name: str) -> bool:
        p = table_path(self.out_dir, name)
        if not os.path.exists(p):
            return False
        try:
            from ..table import read_table_dict  # noqa: PLC0415

            return read_table_dict(p).get("video") == name
        except Exception:
            return False


def is_done(out_dir: str, name: str) -> bool:
    """True when ``out_dir`` holds a readable table of video ``name``."""
    return Ledger(out_dir).is_done(name)


def attempts(out_dir: str, name: str) -> int:
    """Failed attempts recorded for video ``name``."""
    return Ledger(out_dir).attempts(name)


def note_failure(out_dir: str, name: str, err: Any) -> int:
    """Record a failed attempt of video ``name``; returns the attempt count."""
    return Ledger(out_dir).note_failure(name, err)


def partition(items: Sequence[Any], k: int, cost: Callable[[Any], float]) -> list[list]:
    """Greedy split into ``k`` lists of similar total cost (largest item first, to the lightest list)."""
    lists: list[list] = [[] for _ in range(k)]
    loads = [0.0] * k
    for it in sorted(items, key=lambda x: -cost(x)):
        j = loads.index(min(loads))
        lists[j].append(it)
        loads[j] += cost(it)
    return lists


# ---------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------
def _free_gpu_memory() -> None:
    import gc  # noqa: PLC0415

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _worker_loop(w: int, items: list[dict], out_dir: str, cfg, factory: Callable, send: Callable[[tuple], None]) -> int:
    """Process ``items`` in order, then retry the failures once. Returns 4 if any video failed, else 0."""
    led = Ledger(out_dir, cfg.run.max_attempts)
    proposer, verifier = factory(cfg)
    send(("ready", w, None))
    stats_all: dict[str, dict] = {}
    failed: list[dict] = []
    t0 = time.time()

    def one(v: dict, label: str) -> None:
        st: collections.Counter = collections.Counter()
        send(("start", w, v["name"]))
        try:
            detect_video(v["path"], cfg, proposer, verifier, name=v["name"], out_path=table_path(out_dir, v["name"]),
                         start_local=v.get("start_local"), stats=st, fps=v.get("fps"),
                         progress=lambda info: send(("beat", w, info)))
            stats_all[v["name"]] = dict(st)
            send(("log", w, f"{label} {v['name']}: {st['frames']} frames in {st['video_s']:.0f} s, detector calls "
                            f"{st['detector_frames']}, candidates {st['candidates']}, boxes in {st['frames_with_boxes']} frames"))
        except (ImportError, SetupError) as e:
            # a missing package is not the video's fault: stop the worker, leave the ledger alone
            send(("done", w, v["name"]))
            raise SetupError(f"{e!r}") from e
        except Exception as e:  # noqa: BLE001 - every other failure goes to the ledger
            led.note_failure(v["name"], repr(e))
            failed.append(v)
            send(("log", w, f"FAILED {v['name']} ({label}): {e!r}"))
            _free_gpu_memory()
        send(("done", w, v["name"]))

    for n, v in enumerate(items):
        if led.is_done(v["name"]) or led.abandoned(v["name"]):
            continue
        one(v, f"video {n + 1}/{len(items)}")
    retry, failed[:] = list(failed), []
    for v in retry:                      # second try of the videos that failed in this pass
        if led.is_done(v["name"]) or led.abandoned(v["name"]):
            continue
        one(v, "retry")
    os.makedirs(os.path.join(out_dir, "workers"), exist_ok=True)
    summ = dict(worker=w, seconds=time.time() - t0, stats=stats_all, failed=[v["name"] for v in failed])
    p = os.path.join(out_dir, "workers", f"w{w}_{int(time.time() * 1000)}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summ, f)
    return 4 if failed else 0


def _worker_main(w: int, items: list[dict], out_dir: str, cfg, factory: Callable, conn) -> None:
    """Entry point of a worker process; progress and log messages go through ``conn``."""
    def send(msg: tuple) -> None:
        try:
            conn.send(msg)
        except Exception:
            pass

    try:
        rc = _worker_loop(w, items, out_dir, cfg, factory, send)
    except BaseException as e:  # noqa: BLE001 - e.g. a model that cannot be loaded
        send(("log", w, f"worker crashed: {e!r}"))
        rc = 3
    send(("exit", w, rc))
    try:
        conn.close()
    except Exception:
        pass
    sys.exit(rc)


# ---------------------------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------------------------
class _GpuSampler:
    """Samples GPU utilisation and memory every ``every_s`` seconds with nvidia-smi, if present."""

    def __init__(self, every_s: float = 15.0):
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self.exe = shutil.which("nvidia-smi")
        self.every_s = every_s
        if self.exe:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                o = subprocess.run([self.exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0].split(",")
                self.samples.append((float(o[0]), float(o[1])))
            except Exception:
                pass
            self._stop.wait(self.every_s)

    def stop(self) -> dict:
        self._stop.set()
        if not self.samples:
            return dict(gpu_util_mean=None, gpu_mem_max_mb=None)
        return dict(gpu_util_mean=round(sum(s[0] for s in self.samples) / len(self.samples), 1),
                    gpu_mem_max_mb=max(s[1] for s in self.samples))


def _jobs(videos: Iterable[Any]) -> list[dict]:
    """Normalise the video list: paths or dicts ``{path, name?, start_local?, fps?}``; adds ``cost``.

    Table names must be unique ignoring letter case (they are file names)."""
    from ..io.video import probe  # noqa: PLC0415

    jobs, seen = [], {}
    for v in videos:
        d = dict(v) if isinstance(v, Mapping) else dict(path=os.fspath(v))
        d["path"] = os.fspath(d["path"])
        d["name"] = d.get("name") or video_name(d["path"])
        key = d["name"].casefold()            # table files of 'Cam1' and 'cam1' collide on Windows and macOS
        if key in seen:
            raise ValueError(f"two videos have the table name {d['name']!r} (letter case ignored): {seen[key]} and "
                             f"{d['path']} (give one of them a different name)")
        seen[key] = d["path"]
        if "cost" not in d:
            try:
                info = probe(d["path"])
                d["cost"] = float(info.duration_s or 0) * float(info.fps or 25.0) or float(info.n_frames or 1)
            except Exception:
                d["cost"] = os.path.getsize(d["path"]) / 1e6 if os.path.exists(d["path"]) else 1.0
        jobs.append(d)
    return jobs


def run_many(videos: Iterable[Any], out_dir: str, cfg: Any = None, workers: int | None = None, *,
             factory: Callable | None = None, in_process: bool = False, log: Callable[[str], None] = print,
             poll_s: float = 2.0, retry_failed: bool = False) -> dict:
    """Detect people in many videos; one table per video in ``out_dir``.

    ``videos``: paths, or dicts ``{path, name?, start_local?, fps?, cost?}`` (``fps`` overrides
    the container's frame rate for the scan strides). ``workers`` (default
    ``cfg.run.workers``) worker processes share the device. ``factory(cfg) -> (proposer,
    verifier)`` builds the models inside each worker (default :func:`default_factory`); it must
    be picklable (a module-level function). ``in_process=True`` runs a single worker loop in this
    process (no restarts or stall watchdog; for debugging). ``retry_failed=True`` deletes the
    failure records (``failed/<name>.json``) of the listed videos first, so abandoned videos are
    tried again. With the default factory, :func:`preflight` runs first.

    Returns the summary (also written to ``<out_dir>/summary.json``): ``videos``, ``done``,
    ``todo_at_start``, ``seconds``, ``workers``, ``restarts``, ``missing`` (no table and not
    abandoned), ``quarantined``, ``abandoned``, ``gpu_util_mean``, ``gpu_mem_max_mb``.
    """
    cfg = _cfg(cfg)
    os.makedirs(out_dir, exist_ok=True)
    led = Ledger(out_dir, cfg.run.max_attempts)
    jobs = _jobs(videos)
    if retry_failed:
        for v in jobs:
            try:
                os.remove(led.failed_path(v["name"]))
            except FileNotFoundError:
                pass
    todo = [v for v in jobs if not led.is_done(v["name"]) and not led.abandoned(v["name"])]
    if factory is None:
        if todo:
            preflight(cfg)
        factory = default_factory
    n_workers = max(1, int(workers if workers is not None else cfg.run.workers))
    W = min(n_workers, len(todo)) if todo else 0
    log(f"{len(jobs)} videos, {len(todo)} to do, {sum(v['cost'] for v in todo) / 1e6:.2f} M frames, "
        f"{1 if (in_process and W) else W} worker(s)")
    t_all = time.time()
    gpu = _GpuSampler()
    restarts: collections.Counter = collections.Counter()
    quarantined: list[str] = []
    try:
        if in_process and todo:
            def send(msg: tuple) -> None:
                if msg[0] == "log":
                    log(f"[w0] {msg[2]}")
            W = 1
            _worker_loop(0, todo, out_dir, cfg, factory, send)
        elif W:
            _orchestrate(todo, W, out_dir, cfg, factory, led, log, poll_s, restarts, quarantined)
    finally:
        gstats = gpu.stop()
    done = [v["name"] for v in jobs if led.is_done(v["name"])]
    abandoned = [v["name"] for v in jobs if not led.is_done(v["name"]) and led.abandoned(v["name"])]
    missing = [v["name"] for v in jobs if not led.is_done(v["name"]) and not led.abandoned(v["name"])]
    summ = dict(videos=len(jobs), done=len(done), todo_at_start=len(todo), seconds=round(time.time() - t_all, 1),
                workers=W, restarts=dict(restarts), missing=missing, quarantined=quarantined, abandoned=abandoned, **gstats)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    if abandoned:
        log(f"{len(abandoned)} videos failed {cfg.run.max_attempts} times and were left out: {abandoned[:20]} "
            f"(the reasons are in {os.path.join(out_dir, 'failed')}; run again with --retry-failed, or delete "
            "failed/<name>.json, to try them again)")
    if missing:
        log(f"{len(missing)} videos have no box table: {missing[:20]}")
    return summ


def _orchestrate(todo: list[dict], W: int, out_dir: str, cfg, factory: Callable, led: Ledger, log: Callable[[str], None],
                 poll_s: float, restarts: collections.Counter, quarantined: list[str]) -> None:
    """Start ``W`` worker processes, watch them, restart them, quarantine videos that kill them.

    Each worker has its own pipe for progress messages, so killing a stalled worker cannot
    corrupt the channel of the others. Workers report ``ready`` (models built), ``start`` and
    ``done`` of every video; only the video in progress when a worker dies can be quarantined.
    Until a worker is ready the stall limit is ``max(stall_s, 300 s)`` (loading or downloading
    the models can take a while).
    """
    import multiprocessing as mp  # noqa: PLC0415
    from multiprocessing.connection import wait as mp_wait  # noqa: PLC0415

    ctx = mp.get_context("spawn")             # CUDA cannot be used in a forked child
    procs: dict[int, tuple[Any, list[dict], Any]] = {}
    last_beat: dict[int, float] = {}
    ready: dict[int, bool] = {}
    current: dict[int, str | None] = {}                    # the video a worker is working on
    last_log: dict[int, str] = {}
    stuck: dict[int, tuple[str, int]] = {}                 # slot -> (video, launches in a row that died on it)
    setup_fail: collections.Counter = collections.Counter()  # launches in a row that died before any video
    pending: list[tuple[float, int, list[dict]]] = []      # (start time, slot, items) of delayed restarts
    stall_s, wait_s = float(cfg.run.stall_s), float(cfg.run.restart_wait_s)
    startup_s = max(stall_s, 300.0)

    def start(w: int, items: list[dict]) -> None:
        r, s_ = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_worker_main, args=(w, items, out_dir, cfg, factory, s_), daemon=False)
        p.start()
        s_.close()                            # the parent keeps only the reading end
        procs[w] = (p, items, r)
        last_beat[w] = time.time()
        ready[w], current[w] = False, None

    def drain(timeout: float) -> None:
        conns = {procs[w][2]: w for w in procs if procs[w][2] is not None}
        if not conns:
            time.sleep(timeout)
            return
        for c in mp_wait(list(conns), timeout):
            w = conns[c]
            while True:
                try:
                    if not c.poll():
                        break
                    msg = c.recv()
                except (EOFError, OSError):
                    p, items, _r = procs[w]
                    procs[w] = (p, items, None)  # the worker closed its end (it exited or was killed)
                    break
                last_beat[w] = time.time()
                if msg[0] == "log":
                    last_log[w] = str(msg[2])
                    log(f"[w{w}] {msg[2]}")
                elif msg[0] == "ready":
                    ready[w] = True
                elif msg[0] == "start":
                    current[w] = msg[2]
                elif msg[0] == "done":
                    current[w] = None

    for w, items in enumerate(partition(todo, W, lambda v: v["cost"])):
        start(w, items)
    try:
        while procs or pending:
            drain(poll_s)
            now = time.time()
            for k in [k for k, (t, _w, _i) in enumerate(pending) if t <= now][::-1]:
                _t, w, items = pending.pop(k)
                start(w, items)
            for w in list(procs):
                p, items, r = procs[w]
                rc = p.exitcode
                limit = stall_s if ready.get(w) else startup_s
                if rc is None and time.time() - last_beat[w] > limit:
                    log(f"worker {w} sent nothing for {limit / 60:.1f} min: killing it")
                    p.kill()
                    p.join(30)
                    rc = p.exitcode if p.exitcode is not None else -9
                if rc is None:
                    continue
                p.join(5)
                if r is not None:
                    drain(0.0)                  # last messages of the worker
                    try:
                        r.close()
                    except Exception:
                        pass
                del procs[w]
                left = [v for v in items if not led.is_done(v["name"]) and not led.abandoned(v["name"])]
                if not left:
                    continue
                dying_on = current.get(w)
                if len(left) < len(items):     # progress: start counting again
                    stuck.pop(w, None)
                    setup_fail[w] = 0
                if dying_on is None and len(left) == len(items) and rc != 4:
                    # died before (or between) videos without finishing any: not a video's fault
                    setup_fail[w] += 1
                    if setup_fail[w] >= 3:
                        log(f"worker {w} could not get to work in 3 launches in a row (last message: "
                            f"{last_log.get(w, 'none')}); giving up its {len(left)} videos, the failure ledger "
                            "is not touched")
                        continue
                elif dying_on is not None:
                    setup_fail[w] = 0
                    prev, n = stuck.get(w, (None, 0))
                    n = n + 1 if prev == dying_on else 1
                    stuck[w] = (dying_on, n)
                    if n >= 3:                 # this video killed the worker in three launches in a row
                        quarantined.append(dying_on)
                        for _ in range(led.max_attempts):
                            led.note_failure(dying_on, "the worker died in three launches in a row on this video")
                        log(f"worker {w}: video {dying_on} quarantined after 3 launches that died on it")
                        left = [v for v in left if v["name"] != dying_on]
                        stuck.pop(w, None)
                if left:
                    restarts[w] += 1
                    log(f"worker {w} exited with code {rc}; restarting it in {wait_s:g} s with {len(left)} videos "
                        f"(restart {restarts[w]})")
                    pending.append((time.time() + wait_s, w, left))
    finally:
        for p, _items, r in procs.values():
            if p.exitcode is None:
                p.kill()
                p.join(10)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass


def config_summary(cfg: Any) -> dict:
    """The detector settings of a config, as stored in a table's ``params``."""
    return _detector_params(_cfg(cfg))
