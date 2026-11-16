# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.schedule on synthetic detections."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil import schedule as S  # noqa: E402
from openveil.config import DayNightPolicy, ScheduleConfig  # noqa: E402

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class Scene:
    """A synthetic video: people present in frame ranges, moving linearly; a detect() that boxes them
    and counts its calls."""

    def __init__(self, people, miss=()):
        self.people = people          # list of (first_frame, last_frame, box_at_first, box_at_last)
        self.miss = set(miss)         # frames where the detector misses everybody
        self.calls = []

    def box(self, p, f):
        f0, f1, b0, b1 = p
        w = (f - f0) / (f1 - f0) if f1 > f0 else 0.0
        return [(1 - w) * b0[k] + w * b1[k] for k in range(4)]

    def truth(self, f):
        return [self.box(p, f) for p in self.people if p[0] <= f <= p[1]]

    def detect(self, idxs):
        self.calls.append(list(idxs))
        out = {}
        for f in idxs:
            out[f] = [] if f in self.miss else [b + [0.9, 1.0] for b in self.truth(f)]
        return out

    @property
    def ran(self):
        return sorted(f for c in self.calls for f in c)


class TestChunksAndDayNight(unittest.TestCase):
    def test_plan_chunks(self):
        self.assertEqual(S.plan_chunks(125.0, 60.0), [(0.0, 60.0), (60.0, 60.0), (120.0, 60.0)])
        self.assertEqual(S.plan_chunks(120.0, 60.0), [(0.0, 60.0), (60.0, 60.0)])
        self.assertEqual(S.plan_chunks(0.0, 60.0), [(0.0, 60.0)])

    def test_always_full_is_the_default(self):
        self.assertEqual(S.mode_at("2024-03-05T02:00:00", 0.0), "full")
        self.assertEqual(S.chunk_mode("2024-03-05T02:00:00", 0.0, 60.0), "full")
        self.assertEqual(S.chunk_mode(None, 0.0, 60.0, DayNightPolicy(policy="clock")), "full")       # no clock: safe default

    def test_clock_policy(self):
        p = DayNightPolicy(policy="clock")
        self.assertEqual(S.clock_hour("2024-03-05T12:34:56", 3600 + 1504), 14.0)
        self.assertEqual(S.mode_at("2024-03-05T05:59:00", 0.0, p), "sparse")
        self.assertEqual(S.mode_at("2024-03-05T05:59:00", 60.0, p), "full")
        self.assertEqual(S.mode_at("2024-03-05T22:59:59", 1.0, p), "sparse")
        # a chunk is day when its end plus the 300 s margin reaches 06:00
        self.assertEqual(S.chunk_mode("2024-03-05T05:50:00", 0.0, 60.0, p), "sparse")       # end 05:51 + 5 min = 05:56: night
        self.assertEqual(S.chunk_mode("2024-03-05T05:54:00", 0.0, 60.0, p), "full")         # end 05:55 + 5 min = 06:00
        self.assertEqual(S.chunk_mode("2024-03-05T22:59:00", 0.0, 60.0, p), "full")         # starts at day
        self.assertEqual(S.chunk_mode("2024-03-05T23:00:00", 0.0, 60.0, p), "sparse")
        self.assertEqual(S.chunk_mode("2024-03-05T23:00:00", 0.0, 60.0, DayNightPolicy(policy="clock", night_mode="skip")), "skip")
        with self.assertRaises(ValueError):
            S.mode_at("2024-03-05T23:00:00", 0.0, DayNightPolicy(policy="moon"))

    def test_start_local_from_name(self):
        self.assertEqual(S.start_local_from_name("cam3_20240305-123456.mp4"), "2024-03-05T12:34:56")
        self.assertEqual(S.start_local_from_name("cam3_20240305_123456_extra.mp4"), "2024-03-05T12:34:56")
        self.assertIsNone(S.start_local_from_name("clip.mp4"))
        self.assertIsNone(S.start_local_from_name("x_20241399-999999.mp4"))


class TestSparseScan(unittest.TestCase):
    def test_no_person_runs_checkpoints_only(self):
        sc = Scene([])
        res, ran = S.sparse_scan(100, sc.detect, stride=10, clear_gaps=2, lookahead=4)
        self.assertEqual(ran, list(range(0, 100, 10)))
        self.assertEqual(sorted(res), ran)
        self.assertEqual([len(c) for c in sc.calls], [4, 4, 2])                    # batches of lookahead checkpoints
        sc = Scene([])
        _, ran = S.sparse_scan(95, sc.detect, stride=10, last_checkpoint=True)
        self.assertEqual(ran, list(range(0, 95, 10)) + [94])

    def test_hit_backfills_and_goes_dense(self):
        sc = Scene([(33, 57, [10, 10, 30, 50], [40, 10, 60, 50])])
        res, ran = S.sparse_scan(200, sc.detect, stride=10, clear_gaps=2, lookahead=4)
        # every frame from the previous clear checkpoint (30) to two clear strides after the last hit is run
        for f in range(31, 58):
            self.assertIn(f, res)
            self.assertEqual(bool(res[f]), f >= 33, f)
        self.assertEqual(ran, sorted(set(ran)))
        self.assertTrue(set(range(31, 80)) <= set(ran))
        self.assertNotIn(85, ran)                                                    # back to sparse
        self.assertIn(90, ran)

    def test_dense_stride(self):
        sc = Scene([(33, 57, [10, 10, 30, 50], [40, 10, 60, 50])])
        res, ran = S.sparse_scan(200, sc.detect, stride=25, clear_gaps=2, lookahead=4, dense_stride=5)
        dense = [f for f in ran if 25 < f < 100]
        self.assertTrue(all(f % 5 == 0 or f in (49, 74, 99) for f in dense), dense)     # every 5th frame plus the last of each group
        self.assertIn(35, dense)
        self.assertIn(55, dense)

    def test_is_hit_filters_and_never_hit_stays_sparse(self):
        sc = Scene([(33, 57, [10, 10, 30, 50], [40, 10, 60, 50])])
        res, ran = S.sparse_scan(200, sc.detect, stride=10, is_hit=lambda bx: False)
        self.assertEqual(ran, list(range(0, 200, 10)))
        self.assertTrue(res[40] and res[50])                                         # boxes are still returned


class TestInterpolation(unittest.TestCase):
    def test_linear_fill(self):
        times = [i * 0.04 for i in range(100)]
        res = {0: [[0, 0, 10, 20, 0.9]], 10: [[10, 0, 20, 20, 0.8]]}
        out = S.interpolate_gaps(res, times, max_gap_s=3.0)
        self.assertEqual(out[0], [[0, 0, 10, 20, "det"]])
        for g in range(1, 10):
            self.assertEqual(len(out[g]), 1)
            b = out[g][0]
            self.assertEqual(b[4], "interp")
            self.assertAlmostEqual(b[0], g, places=6)
            self.assertAlmostEqual(b[2], 10 + g, places=6)
        self.assertEqual(sorted(out), list(range(11)))
        scored = S.interpolate_gaps(res, times, max_gap_s=3.0, keep_score=True)
        self.assertEqual(scored[0], [[0, 0, 10, 20, "det", 0.9]])
        self.assertEqual(len(scored[5][0]), 5)                                      # interpolated boxes carry no score

    def test_variable_frame_rate(self):
        times = [0.0, 0.1, 0.5, 0.6, 1.0]
        out = S.interpolate_gaps({0: [[0, 0, 10, 10]], 4: [[100, 0, 110, 10]]}, times, max_gap_s=3.0, gate_base=100)
        self.assertAlmostEqual(out[2][0][0], 50.0)                                   # half way in time, not in frames
        self.assertAlmostEqual(out[1][0][0], 10.0)

    def test_gaps_too_long_or_too_far_are_not_bridged(self):
        times = [i * 0.04 for i in range(200)]
        res = {0: [[0, 0, 10, 20]], 100: [[1, 0, 11, 20]]}                          # 4 s apart
        self.assertEqual(sorted(S.interpolate_gaps(res, times, max_gap_s=3.0)), [0, 100])
        self.assertEqual(len(S.interpolate_gaps(res, times, max_gap_s=5.0)), 101)
        far = {0: [[0, 0, 10, 20]], 10: [[500, 0, 510, 20]]}                        # teleport
        self.assertEqual(sorted(S.interpolate_gaps(far, times, max_gap_s=3.0)), [0, 10])

    def test_two_people_are_matched_correctly(self):
        times = [i * 0.04 for i in range(50)]
        res = {0: [[0, 0, 10, 20], [200, 0, 210, 20]], 10: [[205, 0, 215, 20], [5, 0, 15, 20]]}
        for use_scipy in ((True, False) if HAVE_SCIPY else (False,)):
            out = S.interpolate_gaps(res, times, max_gap_s=3.0, use_scipy=use_scipy)
            xs = sorted(b[0] for b in out[5])
            self.assertEqual(len(xs), 2)
            self.assertAlmostEqual(xs[0], 2.5)
            self.assertAlmostEqual(xs[1], 202.5)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_greedy_fallback_agrees_on_well_separated_tracks(self):
        rnd = random.Random(3)
        for _ in range(50):
            times = [i * 0.04 for i in range(60)]
            starts = [rnd.uniform(0, 100) + 400 * k for k in range(3)]
            res = {}
            for f in (0, 7, 15, 30):
                res[f] = [[x + f, 50, x + f + 40, 150, 0.9] for x in starts]
                rnd.shuffle(res[f])
            a = S.interpolate_gaps(res, times, 3.0, use_scipy=True)
            b = S.interpolate_gaps(res, times, 3.0, use_scipy=False)
            self.assertEqual({f: sorted(v) for f, v in a.items()}, {f: sorted(v) for f, v in b.items()})


class TestScanChunk(unittest.TestCase):
    P = ScheduleConfig()

    def test_day_chunk_covers_the_person(self):
        fps, n = 25.0, 500
        times = [i / fps for i in range(n)]
        sc = Scene([(130, 330, [100, 100, 150, 250], [400, 120, 450, 270])])
        fb, ran = S.scan_chunk(n, times, sc.detect, "full", fps, self.P)
        for f in range(130, 331):                                                    # every frame with the person carries a box
            self.assertTrue(fb.get(f), f)
        self.assertLess(len(ran), n / 2)                                             # far fewer calls than frames
        self.assertIn(n - 1, ran)                                                    # last checkpoint
        kinds = {b[4] for f in fb for b in fb[f]}
        self.assertEqual(kinds, {"det", "interp"})
        # a dict of parameters works as well
        sc2 = Scene(sc.people)
        fb2, ran2 = S.scan_chunk(n, times, sc2.detect, "full", fps, dict(base_s=1.0, dense_s=0.2, night_base_s=5.0, clear_gaps=2,
                                                                          lookahead=4, max_gap_s=3.0))
        self.assertEqual((fb2, ran2), (fb, ran))

    def test_missed_detections_are_bridged(self):
        fps, n = 25.0, 500
        times = [i / fps for i in range(n)]
        sc = Scene([(130, 330, [100, 100, 150, 250], [160, 100, 210, 250])], miss=range(200, 260))
        fb, _ = S.scan_chunk(n, times, sc.detect, "full", fps, self.P)
        self.assertTrue(all(fb.get(f) for f in range(130, 331)))
        self.assertTrue(all(b[4] == "interp" for f in range(205, 255) for b in fb[f]))

    def test_night_chunk_is_sparse_with_a_longer_bridge(self):
        fps, n = 25.0, 1500
        times = [i / fps for i in range(n)]
        sc = Scene([(100, 1400, [100, 100, 150, 250], [130, 100, 180, 250])], miss={500})
        fb, ran = S.scan_chunk(n, times, sc.detect, "sparse", fps, self.P)
        self.assertEqual(ran, list(range(0, n, 125)) + [n - 1])                      # every 5 s + the last frame, no dense mode
        # checkpoints 375 and 625 are 10 s apart (one missed): bridged, since 10 s <= 2.2 x 5 s
        self.assertTrue(all(fb.get(f) for f in range(125, 1376)))
        # a person in view to the end of the chunk keeps a box up to its last frame (no gap at the boundary)
        sc = Scene([(0, n - 1, [100, 100, 150, 250], [100, 100, 150, 250])])
        fb, ran = S.scan_chunk(n, times, sc.detect, "sparse", fps, self.P)
        self.assertEqual(ran[-1], n - 1)
        self.assertTrue(all(fb.get(f) for f in range(n)))
        self.assertEqual(S.scan_chunk(n, times, sc.detect, "skip", fps, self.P), ({}, []))
        self.assertEqual(S.scan_chunk(0, [], sc.detect, "full", fps, self.P), ({}, []))
        with self.assertRaises(ValueError):
            S.scan_chunk(n, times, sc.detect, "dusk", fps, self.P)

    def test_hit_flag_controls_the_dense_mode(self):
        fps, n = 25.0, 500
        times = [i / fps for i in range(n)]

        def detect(idxs):                                                            # boxes below the hit threshold (flag 0)
            return {f: [[100, 100, 150, 250, 0.2, 0.0]] if 130 <= f <= 330 else [] for f in idxs}
        fb, ran = S.scan_chunk(n, times, detect, "full", fps, self.P)
        self.assertEqual(ran, list(range(0, n, 25)) + [n - 1])                       # never dense
        self.assertTrue(fb[150])                                                     # but the boxes are kept and interpolated
        self.assertTrue(fb[160])


if __name__ == "__main__":
    unittest.main(verbosity=2)
