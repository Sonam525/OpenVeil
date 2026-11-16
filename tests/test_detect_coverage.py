# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Detection must scan every frame of a video, and setup problems must not be blamed on videos.

Synthetic videos only (flat grey frames with a bright rectangle standing in for a person):

* a video without a stated duration and a recording pause longer than a chunk;
* a video whose header states a duration shorter than its content;
* night (sparse) chunks: a person in view all the time keeps a box in every frame, also at chunk
  boundaries and at the end of the video;
* a sparse chunk of a video whose container states no frame rate;
* a worker whose models cannot be built: no video is quarantined;
* table names that differ only in letter case; ``retry_failed``; the preflight check;
* a crafted legacy ``head.pt`` is refused without running its code;
* the built-in model-free factory; a day window that crosses midnight.
"""
from __future__ import annotations

import os
import shutil
import struct
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np
from test_detect_fakes import HAVE_AV, FakeProposer, FakeVerifier, H, W, make_video

from openveil.config import DayNightPolicy, load_config, validate_config

try:
    import av
    HAVE_X264 = HAVE_AV and "libx264" in av.codecs_available
except ImportError:  # pragma: no cover
    HAVE_X264 = False


def _cfg(**over):
    cfg = load_config({"verifier": {"mode": "none"}})
    cfg.run.device = "cpu"
    cfg.run.restart_wait_s = 0.0
    for k, v in over.items():
        setattr(cfg.schedule, k, v)
    return cfg


def _write(path, times_s, person_from_s=None, fmt=None, options=None, tb=Fraction(1, 1000), fps=10):
    """H.264 video with frames at ``times_s``; from ``person_from_s`` on a bright rectangle is in view."""
    import av

    with av.open(path, "w", format=fmt, options=options or {}) as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"bf": "0", "crf": "4", "preset": "ultrafast"}
        s.codec_context.time_base = tb
        for t in times_s:
            img = np.full((H, W, 3), 60, np.uint8)
            if person_from_s is not None and t >= person_from_s:
                img[30:90, 40:60] = 235
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts = int(round(t / tb))
            fr.time_base = tb
            for pk in s.encode(fr):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path


@unittest.skipUnless(HAVE_X264, "PyAV with libx264 is needed")
class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_cov_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _detect(self, path, cfg):
        from openveil.detect.run import detect_video
        return detect_video(path, cfg, FakeProposer(), FakeVerifier())

    def test_pause_in_a_video_without_duration(self):
        from openveil.apply.plan import MaskPlan
        from openveil.io.video import probe

        times = [i / 10 for i in range(100)] + [150 + i / 10 for i in range(100)]      # 0-10 s, pause, 150-160 s
        p = _write(os.path.join(self.tmp, "live.mkv"), times, person_from_s=150, fmt="matroska", options={"live": "1"})
        self.assertIsNone(probe(p).duration_s)
        cfg = _cfg(chunk_s=20.0)
        tab = self._detect(p, cfg)
        self.assertEqual([c[0] for c in tab.chunks], [0.0, 140.0])                 # the empty windows are skipped
        self.assertEqual(sum(c[2] for c in tab.chunks), 200)                       # every frame was decoded
        self.assertTrue(MaskPlan(tab, cfg).boxes_at(155.0))

    def test_content_past_an_understated_duration(self):
        from openveil.apply.plan import MaskPlan
        from openveil.io.video import probe

        p = _write(os.path.join(self.tmp, "rec.mkv"), [i / 10 for i in range(400)], person_from_s=30, fmt="matroska")
        with open(p, "rb") as f:
            b = bytearray(f.read())
        i = b.find(b"\x44\x89\x88")                                                  # the Matroska Duration element
        self.assertGreater(i, 0)
        b[i + 3:i + 11] = struct.pack(">d", 10000.0)                                 # 40 s -> "10 s"
        j = b.find(b"00:00:40")
        if j >= 0:
            b[j:j + 8] = b"00:00:10"
        with open(p, "wb") as f:
            f.write(bytes(b))
        self.assertLess(probe(p).duration_s, 11)
        cfg = _cfg(chunk_s=10.0)
        tab = self._detect(p, cfg)
        self.assertEqual(sum(c[2] for c in tab.chunks), 400)
        self.assertTrue(MaskPlan(tab, cfg).boxes_at(35.0))

    def test_night_person_masked_in_every_frame(self):
        from openveil.apply.plan import MaskPlan

        p = _write(os.path.join(self.tmp, "cam1_20240305-020000.mp4"), [i / 10 for i in range(260)], person_from_s=0,
                   fmt="mp4", tb=Fraction(1, 10))
        cfg = _cfg(chunk_s=14.0)          # checkpoints at 0, 5, 10 s: without the last frame 12-14 s would be bare
        cfg.schedule.day_night.policy = "clock"
        tab = self._detect(p, cfg)
        self.assertEqual({c[1] for c in tab.chunks}, {"sparse"})
        plan = MaskPlan(tab, cfg)
        bare = [i for i in range(260) if not plan.boxes_at(i / 10)]
        self.assertEqual(bare, [])

    def test_sparse_chunk_without_a_stated_frame_rate(self):
        import openveil.io.video as iov
        from openveil.apply.plan import MaskPlan

        p = _write(os.path.join(self.tmp, "cam1_20240305-020000.mp4"), [i / 10 for i in range(300)], person_from_s=0,
                   fmt="mp4", tb=Fraction(1, 10))
        real = iov.probe

        def no_rate(path):
            info = real(path)
            info.fps, info.average_rate = None, None
            return info

        cfg = _cfg(chunk_s=12.0)
        cfg.schedule.day_night.policy = "clock"
        with mock.patch.object(iov, "probe", no_rate):
            tab = self._detect(p, cfg)
        self.assertAlmostEqual(tab.fps, 10.0, places=3)
        self.assertTrue(MaskPlan(tab, cfg).boxes_at(20.0))

    def test_fps_override(self):
        p = make_video(os.path.join(self.tmp, "v.mp4"), seconds=3)
        from openveil.detect.run import detect_video
        tab = detect_video(p, _cfg(), FakeProposer(), FakeVerifier(), fps=12.5)
        self.assertEqual(tab.fps, 12.5)


@unittest.skipUnless(HAVE_AV, "PyAV is not installed")
class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_run_")
        self.logs = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_worker_that_cannot_build_its_models_blames_no_video(self):
        from openveil.detect.run import run_many
        from test_detect_fakes import broken_factory

        vids = [make_video(os.path.join(self.tmp, f"v{i}.mp4"), seconds=2) for i in range(2)]
        out = os.path.join(self.tmp, "tables")
        s = run_many(vids, out, _cfg(), workers=1, factory=broken_factory, log=self.logs.append, poll_s=0.2)
        self.assertEqual(s["quarantined"], [], self.logs)
        self.assertEqual(s["abandoned"], [])
        self.assertEqual(sorted(s["missing"]), ["v0", "v1"])
        self.assertFalse(os.path.exists(os.path.join(out, "failed")))
        self.assertTrue(any("could not get to work" in m for m in self.logs), self.logs)

    def test_names_differing_only_in_case_are_refused(self):
        from openveil.detect.run import run_many

        os.makedirs(os.path.join(self.tmp, "A"))
        os.makedirs(os.path.join(self.tmp, "B"))
        a = make_video(os.path.join(self.tmp, "A", "Cam1.mp4"), seconds=1)
        b = make_video(os.path.join(self.tmp, "B", "cam1.mp4"), seconds=1)
        with self.assertRaises(ValueError):
            run_many([a, b], os.path.join(self.tmp, "t"), _cfg(), in_process=True, log=self.logs.append)

    def test_retry_failed_clears_the_ledger(self):
        from openveil.detect.run import Ledger, run_many
        from test_detect_fakes import fake_factory

        v = make_video(os.path.join(self.tmp, "v.mp4"), seconds=2)
        out = os.path.join(self.tmp, "t")
        led = Ledger(out, 3)
        for _ in range(3):
            led.note_failure("v", "an earlier failure")
        s = run_many([v], out, _cfg(), factory=fake_factory, in_process=True, log=self.logs.append)
        self.assertEqual(s["abandoned"], ["v"])
        self.assertTrue(any("--retry-failed" in m for m in self.logs), self.logs)
        s = run_many([v], out, _cfg(), factory=fake_factory, in_process=True, log=self.logs.append, retry_failed=True)
        self.assertEqual((s["done"], s["abandoned"]), (1, []))

    def test_preflight(self):
        from openveil.detect.run import SetupError, preflight

        cfg = load_config("production")
        cfg.verifier.mode, cfg.verifier.adapter_dir, cfg.verifier.head_path = "crop", None, None
        with self.assertRaises(SetupError):
            preflight(cfg)
        cfg.verifier.mode = "none"
        import importlib.util
        real = importlib.util.find_spec
        with mock.patch.object(importlib.util, "find_spec", lambda m, *a: None if m == "transformers" else real(m, *a)):
            with self.assertRaises(SetupError) as cm:
                preflight(cfg)
        self.assertIn("openveil[detect]", str(cm.exception))

    def test_builtin_fake_factory(self):
        from openveil.detect.fake import factory
        from openveil.detect.run import detect_video

        p = make_video(os.path.join(self.tmp, "v.mp4"), seconds=8)
        prop, ver = factory(None)
        tab = detect_video(p, _cfg(), prop, ver)
        self.assertTrue(tab.frames)                                # the bright walker is boxed


class TestCrossMidnightDay(unittest.TestCase):
    def test_day_window_crossing_midnight(self):
        from openveil.schedule import chunk_mode

        p = DayNightPolicy(policy="clock", day_start=18, day_end=6, night_mode="skip")
        modes = [chunk_mode(f"2024-03-05T{t}", 0, 60, p) for t in ("20:00:00", "23:30:00", "02:00:00", "12:00:00")]
        self.assertEqual(modes, ["full", "full", "full", "skip"])
        cfg = load_config("production")
        cfg.schedule.day_night.day_start = cfg.schedule.day_night.day_end = 6
        with self.assertRaises(ValueError):
            validate_config(cfg)


class TestLegacyHead(unittest.TestCase):
    def test_crafted_head_pt_is_refused_without_running_code(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        import zipfile
        from openveil.detect.verifier import load_head

        d = tempfile.mkdtemp(prefix="ov_head_")
        try:
            marker = os.path.join(d, "ran.txt")
            base = os.path.join(d, "base.pt")
            torch.save({}, base)
            code = f"open({marker!r}, 'w').write('x')"
            # OrderedDict() ; BUILD (None, {}) -> TypeError in the weights-only unpickler ; then exec(code)
            pkl = (b"\x80\x02" + b"ccollections\nOrderedDict\n" + b")R" + b"N}\x86b0"
                   + b"cbuiltins\nexec\n(X" + len(code).to_bytes(4, "little") + code.encode() + b"tR0" + b"}.")
            evil = os.path.join(d, "head.pt")
            with zipfile.ZipFile(base) as zin, zipfile.ZipFile(evil, "w", zipfile.ZIP_STORED) as zout:
                for it in zin.infolist():
                    zout.writestr(it, pkl if it.filename.endswith("data.pkl") else zin.read(it.filename))
            with self.assertRaises(ValueError):
                load_head(evil)
            self.assertFalse(os.path.exists(marker))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
