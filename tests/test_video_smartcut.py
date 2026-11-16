# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.apply.video: smart cut, eligibility probe, checks and fallbacks, on
synthetic H.264 / HEVC / MPEG-4 Part 2 videos (moving rectangles, no people)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_video_synth import FPS, GOP, H, HAVE_AV, W, decoded_frames, has_encoder, make_video, table_dict

# Boxes at 0.5 s and 7.5 s of an 8 s video (GOPs of 1 s): with the +-2 s hold the GOPs of
# seconds 0-2 and 5-7 hold masked frames, the GOPs of seconds 3 and 4 are copied.
ROWS = [(0.5, [[40, 40, 140, 140]]), (7.5, [[150, 60, 260, 200]])]
SECONDS = 8.0
EXPECTED_RUNS = [[0, 3 * GOP], [5 * GOP, 8 * GOP]]


def _md5s(path):
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        return [(p.pts, hashlib.md5(bytes(p)).hexdigest()) for p in c.demux(st) if p.size > 0]


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _table(path, rows=ROWS):
    from openveil.table import BoxTable
    return BoxTable.from_dict(table_dict(path, rows))


class _CutCase:
    """Shared checks of a smart cut for one encoder (mixed into a TestCase)."""
    encoder = "libx264"
    codec = "h264"

    @classmethod
    def setUpClass(cls):
        from openveil.apply import video as V
        cls.tmp = tempfile.mkdtemp(prefix=f"ov_cut_{cls.codec}_")
        cls.src = make_video(os.path.join(cls.tmp, "src.mp4"), codec=cls.encoder, seconds=SECONDS)
        cls.table = _table(cls.src)
        cls.dst = os.path.join(cls.tmp, "out.mp4")
        cls.stats = V.smart_cut(cls.src, cls.table, cls.dst)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_eligible_and_smart(self):
        from openveil.apply import video as V
        e = V.probe_eligibility(self.src)
        self.assertTrue(e.smart_cut, e.reasons)
        self.assertEqual(self.stats["mode"], "smart")
        self.assertEqual(self.stats["codec"], self.codec)
        self.assertEqual(self.stats["run_ranges"], EXPECTED_RUNS)
        self.assertEqual(self.stats["packets_reencoded"], 6 * GOP)
        self.assertEqual(self.stats["block"], 16)

    def test_verify_fast_and_full(self):
        from openveil.apply import video as V
        v = V.verify_fast(self.src, self.dst, self.stats)
        self.assertTrue(v["ok"], v)
        self.assertEqual(v["windows"], 2)
        full = V.verify(self.src, self.dst, self.stats)
        self.assertTrue(full["ok"], full)
        pair = V.verify_pair(self.src, self.dst)
        self.assertTrue(pair["ok"], pair)
        self.assertEqual((pair["mode"], pair["changed_runs"]), ("smart", EXPECTED_RUNS))
        self.assertTrue(V.verify_pair(self.src, self.src)["ok"])
        other = make_video(os.path.join(self.tmp, "other.mp4"), codec=self.encoder, seconds=SECONDS - 1)
        self.assertFalse(V.verify_pair(self.src, other)["ok"])

    def test_untouched_packets_identical_and_pts(self):
        a, b = _md5s(self.src), _md5s(self.dst)
        self.assertEqual(len(a), len(b))
        self.assertEqual([x[0] for x in a], [x[0] for x in b])          # same pts, packet by packet
        inrun = np.zeros(len(a), bool)
        for s, e in EXPECTED_RUNS:
            inrun[s:e] = True
        for i, (x, y) in enumerate(zip(a, b)):
            if inrun[i]:
                self.assertNotEqual(x[1], y[1], f"packet {i} should be re-encoded")
            else:
                self.assertEqual(x[1], y[1], f"packet {i} should be copied")

    def test_frames_and_mosaic(self):
        from openveil.apply.mosaic import mosaic_array
        from openveil.apply.plan import MaskPlan
        from openveil.apply.quality import cell_mask
        plan = MaskPlan(self.table)
        fs, fd = decoded_frames(self.src), decoded_frames(self.dst)
        self.assertEqual(len(fs), int(SECONDS * FPS))
        self.assertEqual([p for p, _ in fs], [p for p, _ in fd])
        first = fs[0][0]
        tb = self.table.time_base
        n_masked = 0
        for i, ((pts, a), (_, b)) in enumerate(zip(fs, fd)):
            t = (pts - first) * tb[0] / tb[1]
            boxes = plan.boxes_at(t)
            in_run = any(s <= i < e for s, e in EXPECTED_RUNS)
            if not in_run:
                self.assertTrue(np.array_equal(a, b), f"copied frame {i} changed")
                self.assertFalse(boxes)
                continue
            a16, b16 = a.astype(np.int16), b.astype(np.int16)
            if boxes:
                n_masked += 1
                m = cell_mask(boxes, W, H, 16)
                expect = mosaic_array(a.copy(), boxes, 16).astype(np.int16)
                self.assertLess(np.abs(b16 - expect)[m].mean(), 6.0, f"frame {i}: mosaic missing")
                self.assertGreater(np.abs(b16 - a16)[m].mean(), 15.0, f"frame {i}: cells unchanged")
                self.assertLess(np.abs(b16 - a16)[~m].mean(), 6.0, f"frame {i}: outside changed")
            else:
                # re-encoded but not masked (end of a GOP after the hold)
                self.assertLess(np.abs(b16 - a16).mean(), 6.0, f"frame {i}: unmasked frame changed")
        self.assertEqual(n_masked, self.stats["frames_mosaicked"])
        self.assertEqual(n_masked, 26 + 25)                    # 0.0-2.5 s and 5.5-7.9 s


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestSmartCutH264(_CutCase, unittest.TestCase):
    encoder, codec = "libx264", "h264"


@unittest.skipUnless(HAVE_AV and has_encoder("libx265"), "PyAV with libx265 needed")
class TestSmartCutHEVC(_CutCase, unittest.TestCase):
    encoder, codec = "libx265", "hevc"


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestFallbacks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_fb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check_full(self, src, reason_part):
        from openveil.apply import video as V
        e = V.probe_eligibility(src)
        self.assertFalse(e.smart_cut)
        self.assertTrue(any(reason_part in r for r in e.reasons), e.reasons)
        dst = os.path.join(self.tmp, "out.mp4")
        r = V.mask_video(src, _table(src), dst)
        self.assertTrue(r["ok"], json.dumps(r))
        self.assertEqual(r["mode"], "full")
        self.assertNotIn("attempt2", r)                        # an ineligible source is not tried twice
        st = r["attempt1"]["stats"]
        self.assertEqual(st["packets_reencoded"], st["packets"])
        self.assertIn(reason_part, st["full_reason"])
        fs, fd = decoded_frames(src), decoded_frames(dst)
        self.assertEqual(sorted(p for p, _ in fs), [p for p, _ in fd])
        self.assertEqual(st["frames_mosaicked"], 51)
        self.assertEqual(r["sha256"], hashlib.sha256(_bytes(dst)).hexdigest())
        pair = V.verify_pair(src, dst)
        self.assertTrue(pair["ok"], pair)
        self.assertEqual(pair["mode"], "full")
        return r

    def test_bframes_full(self):
        src = make_video(os.path.join(self.tmp, "b.mp4"), bframes=2, seconds=SECONDS)
        self._check_full(src, "B-frames")

    def test_out_of_band_parameter_sets_full(self):
        src = make_video(os.path.join(self.tmp, "oob.mp4"), repeat_headers=False, seconds=SECONDS)
        self._check_full(src, "parameter sets")

    @unittest.skipUnless(has_encoder("mpeg4"), "mpeg4 encoder needed")
    def test_mpeg4_full(self):
        src = make_video(os.path.join(self.tmp, "m4.mp4"), codec="mpeg4", seconds=SECONDS)
        r = self._check_full(src, "mpeg4")
        self.assertEqual(r["attempt1"]["stats"]["encoder"], "mpeg4")

    def test_requested_full_reencode(self):
        from openveil.apply import video as V
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=4.0)
        dst = os.path.join(self.tmp, "out.mp4")
        r = V.mask_video(src, _table(src, [(0.5, [[40, 40, 140, 140]])]), dst, full_reencode=True)
        self.assertTrue(r["ok"])
        self.assertEqual(r["mode"], "full")
        self.assertEqual(r["attempt1"]["stats"]["full_reason"], "requested")

    def test_failed_check_falls_back_to_full(self):
        from openveil.apply import video as V
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=4.0)
        dst = os.path.join(self.tmp, "out.mp4")
        real = V.verify_fast
        calls = []

        def flaky(s, d, st):
            calls.append(st["mode"])
            if len(calls) == 1:
                return {"ok": False, "simulated": True}
            return real(s, d, st)

        with mock.patch.object(V, "verify_fast", side_effect=flaky):
            r = V.mask_video(src, _table(src, [(0.5, [[40, 40, 140, 140]])]), dst)
        self.assertEqual(calls, ["smart", "full"])
        self.assertTrue(r["ok"])
        self.assertEqual(r["mode"], "full")
        self.assertFalse(r["attempt1"]["verify"]["ok"])
        self.assertTrue(r["attempt2"]["verify"]["ok"])

    def test_failed_cut_falls_back_to_full(self):
        from openveil.apply import video as V
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=3.0)
        dst = os.path.join(self.tmp, "out.mp4")
        real = V.smart_cut

        def broken(*a, **kw):
            if not kw.get("full"):
                raise RuntimeError("simulated decoder error")
            return real(*a, **kw)

        with mock.patch.object(V, "smart_cut", side_effect=broken):
            r = V.mask_video(src, _table(src, [(0.5, [[40, 40, 140, 140]])]), dst)
        self.assertIn("simulated", r["attempt1"]["error"])
        self.assertTrue(r["ok"])
        self.assertEqual(r["mode"], "full")

    def test_failed_check_without_fallback_writes_nothing(self):
        from openveil.apply import video as V
        from openveil.config import load_config
        cfg = load_config({"video": {"fallback_full_reencode": False}})
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=3.0)
        dst = os.path.join(self.tmp, "out.mp4")
        with mock.patch.object(V, "verify_fast", return_value={"ok": False}):
            r = V.mask_video(src, _table(src, [(0.5, [[40, 40, 140, 140]])]), dst, cfg)
        self.assertFalse(r["ok"])
        self.assertFalse(os.path.exists(dst))
        self.assertEqual(os.listdir(self.tmp), ["a.mp4"])     # no temporary files left behind

    def test_table_of_other_frame_size_is_refused(self):
        from openveil.apply import video as V
        from openveil.table import BoxTable
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=2.0)
        t = table_dict(src, [(0.5, [[40, 40, 140, 140]])])
        t["width"], t["height"] = 2 * W, 2 * H
        with self.assertRaises(ValueError):
            V.smart_cut(src, BoxTable.from_dict(t), os.path.join(self.tmp, "out.mp4"))
        self.assertEqual(os.listdir(self.tmp), ["a.mp4"])

    def test_no_boxes_copies_source(self):
        from openveil.apply import video as V
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=2.0)
        dst = os.path.join(self.tmp, "sub", "out.mp4")
        r = V.mask_video(src, _table(src, []), dst)
        self.assertTrue(r["ok"] and r["unchanged"])
        self.assertEqual(_bytes(src), _bytes(dst))

    def test_giant_rule_box_not_mosaicked(self):
        """A giant box with two smaller boxes inside is dropped by the plan, so the mosaic
        covers the small boxes only."""
        from openveil.apply import video as V
        src = make_video(os.path.join(self.tmp, "a.mp4"), seconds=3.0)
        dst = os.path.join(self.tmp, "out.mp4")
        rows = [(0.5, [[0, 0, 300, 230], [20, 20, 60, 80], [200, 120, 250, 200]])]
        st = V.smart_cut(src, _table(src, rows), dst)
        self.assertEqual(st["rule"]["boxes_dropped"], 1)
        fs, fd = decoded_frames(src), decoded_frames(dst)
        a, b = fs[5][1].astype(int), fd[5][1].astype(int)
        self.assertLess(np.abs(a - b)[100:110, 120:180].mean(), 6.0)   # inside the giant box only


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestStreamQuirks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_quirk_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repeated_timestamp(self):
        """A packet that repeats the previous pts (seen in some camera files) is moved by one
        tick in the output, recorded in ts_fixes, and the checks account for it."""
        import av
        from openveil.apply import video as V
        base = make_video(os.path.join(self.tmp, "base.mp4"), seconds=4.0)
        src = os.path.join(self.tmp, "rep.mp4")
        with av.open(base) as ic, av.open(src, "w", format="mp4") as oc:
            ist = ic.streams.video[0]
            ost = oc.add_stream_from_template(ist)
            ost.time_base = ist.time_base
            step = None
            for i, p in enumerate(q for q in ic.demux(ist) if q.size > 0):
                step = step or (ist.time_base.denominator // FPS)
                if i == 15:
                    p.pts = 14 * step                 # same pts as packet 14
                p.dts = p.dts - step                  # dts one frame early keeps dts increasing
                p.stream = ost
                oc.mux(p)
        self.assertTrue(V.probe_eligibility(src).smart_cut)
        dst = os.path.join(self.tmp, "out.mp4")
        r = V.mask_video(src, _table(src, [(3.5, [[40, 40, 140, 140]])]), dst)
        self.assertTrue(r["ok"], json.dumps(r))
        st = r["attempt1"]["stats"]
        self.assertEqual(st["mode"], "smart")
        self.assertEqual(st["packets_reencoded"], 3 * GOP)
        self.assertIn([15, 14 * step + 1], st["ts_fixes"])
        a, b = _md5s(src), _md5s(dst)
        self.assertEqual(a[:GOP], b[:GOP])                     # first GOP copied as it was
        self.assertEqual(b[15][0], 14 * step + 1)
        self.assertTrue(V.verify_pair(src, dst)["ok"])

    def test_full_range_source(self):
        """Full-range (yuvj420p) frames keep their sample values and their range flag."""
        import av
        from fractions import Fraction
        from openveil.apply import video as V
        from test_video_synth import synth_frame
        src = os.path.join(self.tmp, "full.mp4")
        with av.open(src, "w", format="mp4") as oc:
            st = oc.add_stream("libx264", rate=FPS)
            st.width, st.height, st.pix_fmt = W, H, "yuvj420p"
            st.codec_context.options = {"preset": "veryfast",
                                        "x264-params": f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes=0:repeat-headers=1"}
            for i in range(4 * FPS):
                f = av.VideoFrame.from_ndarray(synth_frame(i), format="rgb24").reformat(format="yuvj420p")
                f.pts, f.time_base = i, Fraction(1, FPS)
                for p in st.encode(f):
                    oc.mux(p)
            for p in st.encode(None):
                oc.mux(p)
        dst = os.path.join(self.tmp, "out.mp4")
        r = V.mask_video(src, _table(src, [(0.2, [[40, 40, 140, 140]])]), dst)
        self.assertTrue(r["ok"])
        self.assertEqual(r["mode"], "smart")
        with av.open(dst) as c:
            f0 = next(c.decode(video=0))
            self.assertEqual(int(f0.color_range), 2)
        fs, fd = decoded_frames(src), decoded_frames(dst)
        # frame 25: re-encoded (GOP 2) but outside the hold; no range conversion
        self.assertLess(np.abs(fs[25][1].astype(int) - fd[25][1].astype(int)).mean(), 3.0)
        self.assertTrue(np.array_equal(fs[35][1], fd[35][1]))


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestBatchResume(unittest.TestCase):
    def test_mask_videos_resume(self):
        from openveil.apply import video as V
        from openveil.table import save_table
        with tempfile.TemporaryDirectory() as d:
            src = make_video(os.path.join(d, "a.mp4"), seconds=3.0)
            tp = os.path.join(d, "a.json.gz")
            save_table(table_dict(src, [(0.5, [[40, 40, 140, 140]])]), tp)
            jobs = [dict(name="a", src=src, table=tp, dst=os.path.join(d, "out", "a.mp4"), cost=3),
                    dict(name="bad", src=os.path.join(d, "missing.mp4"), table=tp, dst=os.path.join(d, "out", "b.mp4"))]
            res = os.path.join(d, "results")
            r1 = V.mask_videos(jobs, res, log=None)
            self.assertEqual([r["name"] for r in r1], ["a", "bad"])
            self.assertTrue(r1[0]["ok"])
            self.assertFalse(r1[1]["ok"])
            self.assertIn("error", r1[1])
            self.assertTrue(V.result_ok(os.path.join(res, "a.json")))
            r2 = V.mask_videos(jobs, res, log=None)                     # resume: only the failed job again
            self.assertEqual([r["name"] for r in r2], ["bad"])
            r3 = V.mask_videos(jobs, res, worker=1, n_workers=2, log=None)
            self.assertEqual([r["name"] for r in r3], ["bad"])
            with open(os.path.join(res, "a.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["sha256"], V.sha256_file(jobs[0]["dst"]))
            text = json.dumps(saved)
            self.assertNotIn(d.replace("\\", "/"), text.replace("\\\\", "/").replace("\\", "/"))  # no paths in results


class TestBitstreamHelpers(unittest.TestCase):
    def test_annexb_to_lp(self):
        from openveil.apply.video import annexb_to_lp, nal_types
        nal1, nal2 = b"\x67\x42\x00\x1e\xaa", b"\x65\x88\x84\x10"
        # a 4-byte start code, trailing zero bytes after the first NAL unit, a 3-byte start code
        annexb = b"\x00\x00\x00\x01" + nal1 + b"\x00\x00" + b"\x00\x00\x01" + nal2
        lp = annexb_to_lp(annexb)
        self.assertEqual(lp, len(nal1).to_bytes(4, "big") + nal1 + len(nal2).to_bytes(4, "big") + nal2)
        self.assertEqual(nal_types(lp, "h264", 4), [7, 5])
        self.assertEqual(nal_types(annexb, "h264", None), [7, 5])
        self.assertEqual(annexb_to_lp(b"\x12\x34"), b"\x12\x34")
        lp2 = annexb_to_lp(b"\x00\x00\x01" + nal1, nal_len=2)
        self.assertEqual(lp2, len(nal1).to_bytes(2, "big") + nal1)
        self.assertEqual(nal_types(b"\x00\x00\x00\x09\x67", "h264", 4), [-1])
        hevc = (3).to_bytes(4, "big") + bytes([32 << 1, 1, 0]) + (2).to_bytes(4, "big") + bytes([19 << 1, 1])
        self.assertEqual(nal_types(hevc, "hevc", 4), [32, 19])

    def test_nal_length_size(self):
        from openveil.apply.video import nal_length_size
        self.assertEqual(nal_length_size("h264", bytes([1, 100, 0, 30, 0xFD])), 2)
        self.assertEqual(nal_length_size("h264", b"\x00\x00\x00\x01\x67"), None)
        self.assertEqual(nal_length_size("hevc", bytes([1] + [0] * 20 + [0x0F])), 4)

    def test_gops_and_runs(self):
        from openveil.apply.video import gops_of, plan_runs
        pk = [(i, i, i % 4 == 0, 10) for i in range(12)]
        gops = gops_of(pk)
        self.assertEqual(gops, [(0, 4), (4, 8), (8, 12)])
        self.assertEqual(gops_of([(0, 0, False, 1), (1, 1, True, 1)]), [(0, 1), (1, 2)])

        class Plan:
            def intervals(self):
                return [(5.0, 5.0)]

            def boxes_at(self, t):
                return [[0, 0, 1, 1]] if abs(t - 9.0) < 1e-12 else []

        runs, times = plan_runs(pk, gops, Plan(), lambda p: float(p))
        self.assertEqual(runs, [[4, 12]])          # 5.0 by interval, 9.0 only by boxes_at
        self.assertEqual(times[3], 3.0)

    def test_mux_repeated_timestamps(self):
        from openveil.apply.video import _Mux

        class P:
            def __init__(self, pts):
                self.pts = self.dts = pts

        class Sink:
            def __init__(self):
                self.got = []

            def mux(self, p):
                self.got.append((p.pts, p.dts))

        s = Sink()
        m = _Mux(s)
        for pts in (0, 100, 100, 200, 150):
            m.mux(P(pts))
        self.assertEqual(s.got, [(0, 0), (100, 100), (101, 101), (200, 200), (201, 201)])
        self.assertEqual(m.fixes, [[2, 101], [4, 201]])

    def test_encoder_settings(self):
        from openveil.apply.video import encoder_settings
        name, opts = encoder_settings("h264", None, 25)
        self.assertEqual(name, "libx264")
        self.assertEqual(opts, {"preset": "medium", "crf": "20", "bf": "0",
                                "x264-params": "repeat-headers=1:bframes=0:scenecut=0:keyint=25:min-keyint=25"})
        name, opts = encoder_settings("hevc", None, 4)
        self.assertEqual(name, "libx265")
        self.assertEqual(opts["x265-params"],
                         "crf=24:bframes=0:repeat-headers=1:scenecut=0:info=0:log-level=error:keyint=4:min-keyint=4")
        with self.assertRaises(ValueError):
            encoder_settings("vp9")

    def test_eligibility_rules(self):
        from openveil.apply.video import eligibility
        if not HAVE_AV:
            self.skipTest("PyAV not installed")
        pk = [(0, 0, True, 9), (1, 1, False, 9), (2, 2, True, 9)]
        facts = dict(codec="h264", nal_len=4, key_nals=[[7, 8, 5], [7, 8, 5]], pix_fmt="yuv420p")
        self.assertTrue(eligibility(pk, facts).smart_cut or not has_encoder("libx264"))
        e = eligibility(pk, dict(facts, key_nals=[[7, 8, 5], [7, 8, 1]]))
        self.assertEqual(e.keyframes_not_random_access, 1)
        e = eligibility(pk, dict(facts, nal_len=None))
        self.assertFalse(e.smart_cut)
        e = eligibility([(0, 0, True, 9), (2, 1, False, 9), (1, 2, False, 9)], facts)
        self.assertEqual(e.reordered_packets, 1)
        e = eligibility(pk, dict(facts, codec="vp9"))
        self.assertFalse(e.smart_cut)
        # another pixel format than 8-bit 4:2:0 is re-encoded in full (the runs would change it mid-stream)
        for pix in ("yuv422p", "yuv444p", "yuv420p10le", None):
            e = eligibility(pk, dict(facts, pix_fmt=pix))
            self.assertFalse(e.smart_cut)
            self.assertTrue(any("pixel format" in r for r in e.reasons), e.reasons)
        self.assertTrue(eligibility(pk, dict(facts, pix_fmt="yuvj420p")).smart_cut or not has_encoder("libx264"))
        hevc = dict(codec="hevc", nal_len=4, key_nals=[[32, 33, 34, 19], [32, 33, 34, 21]], pix_fmt="yuv420p")
        self.assertTrue(eligibility(pk, hevc).smart_cut or not has_encoder("libx265"))


if __name__ == "__main__":
    unittest.main()
