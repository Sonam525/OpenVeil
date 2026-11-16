# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.io.video: child-process decoding (FrameStream) and probe()."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from test_video_synth import FPS, H, HAVE_AV, W, decoded_frames, has_encoder, make_video


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestFrameStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ov_stream_")
        cls.path = make_video(os.path.join(cls.tmp, "clip.mp4"), seconds=3.0)
        cls.log = os.path.join(cls.tmp, "decode.log")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def stream(self, *a, **kw):
        from openveil.io.video import FrameStream
        kw.setdefault("err_log", self.log)
        return FrameStream(self.path, *a, **kw)

    def test_all_frames_bgr(self):
        s = self.stream(0, None)
        out = list(s)
        self.assertEqual(len(out), 3 * FPS)
        self.assertEqual([o[0] for o in out], list(range(3 * FPS)))
        self.assertAlmostEqual(out[5][1], 0.5, places=6)
        self.assertEqual(out[0][2].shape, (H, W, 3))
        self.assertEqual(s.meta["width"], W)
        self.assertEqual(s.meta["height"], H)
        # the same pixels as an in-process decode
        ref = decoded_frames(self.path, fmt="bgr24")
        self.assertTrue(np.array_equal(out[7][2], ref[7][1]))
        self.assertEqual([o[3] for o in out], [r[0] for r in ref])

    def test_stride_window_and_rgb(self):
        out = list(self.stream(1.0, 1.0, stride=3, fmt="rgb24"))
        self.assertEqual(len(out), FPS)                      # every frame of the window is reported
        sel = [o for o in out if o[2] is not None]
        self.assertEqual([o[0] for o in sel], [0, 3, 6, 9])
        self.assertAlmostEqual(out[0][1], 0.0)               # time from the window's first frame
        ref = decoded_frames(self.path, fmt="rgb24")
        self.assertTrue(np.array_equal(sel[1][2], ref[FPS + 3][1]))

    def test_wanted_stop_after_i420(self):
        s = self.stream(0, None, wanted={2, 4}, stop_after=5, fmt="i420")
        out = list(s)
        self.assertEqual(len(out), 6)
        sel = [o for o in out if o[2] is not None]
        self.assertEqual([o[0] for o in sel], [2, 4])
        self.assertEqual(sel[0][2].shape, (H * 3 // 2, W))
        self.assertFalse(sel[0][4])                          # limited range
        self.assertEqual(s.meta["first_pts"], out[0][3])

    def test_stream_frames(self):
        from openveil.io.video import stream_frames
        out = list(stream_frames(self.path, 0, 2.0, stride=5, err_log=self.log))
        self.assertEqual(len(out), 2 * FPS)
        self.assertEqual(sum(a is not None for _, _, a in out), 4)

    def test_missing_file_raises(self):
        from openveil.io.video import FrameStream
        with self.assertRaises(RuntimeError):
            list(FrameStream(os.path.join(self.tmp, "nope.mp4"), err_log=self.log))

    def test_stall_raises(self):
        from openveil.io.video import DecodeStall
        s = self.stream(0, None, first_timeout=0.5, _delay_s=10)
        with self.assertRaises(DecodeStall):
            list(s)
        self.assertIsNotNone(s._proc.poll())                 # the child was killed

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            self.stream(fmt="yuv444p")


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestProbe(unittest.TestCase):
    def test_probe(self):
        from openveil.io.video import probe
        with tempfile.TemporaryDirectory() as d:
            p = make_video(os.path.join(d, "a.mp4"), seconds=2.0)
            info = probe(p)
            self.assertEqual((info.codec, info.width, info.height), ("h264", W, H))
            self.assertAlmostEqual(info.fps, FPS)
            self.assertEqual(info.n_frames, 2 * FPS)
            self.assertAlmostEqual(info.duration_s, 2.0, places=2)
            self.assertEqual(info.stream_types, ("video",))
            d_ = info.as_dict()
            self.assertIsInstance(d_["time_base"], list)
            self.assertNotIn("extradata", d_)


class TestHelpers(unittest.TestCase):
    def test_import_without_av(self):
        import openveil.io.video as v           # importing never needs av
        self.assertTrue(callable(v.probe))
        self.assertEqual(v._frac("0.1") * 10, 1)


if __name__ == "__main__":
    unittest.main()
