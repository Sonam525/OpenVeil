# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""detect_video end to end on synthetic videos with a fake proposer and verifier, and the
multi-video runner (resume, failure ledger, worker restart, stall watchdog, quarantine)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from test_detect_fakes import (DECOY_BOX, FPS, HANG_BACKGROUND, HAVE_AV, WALK_T0, WALK_T1, FakeProposer, FakeVerifier,
                               make_video, walker_box)

from openveil.config import load_config
from openveil.table import load_table


def small_cfg(**run):
    cfg = load_config("production")
    cfg.schedule.chunk_s = 5.0
    cfg.run.restart_wait_s = 0.0
    for k, v in run.items():
        setattr(cfg.run, k, v)
    return cfg


def _boxes_by_time(table):
    return {round(table.seconds(r[0]), 3): r[1] for r in table.frames}


@unittest.skipUnless(HAVE_AV, "PyAV is not installed")
class TestDetectVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ov_detect_")
        cls.video = make_video(os.path.join(cls.tmp, "cam1_20240305-120000.mp4"), seconds=12.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_detect(self, cfg, **kw):
        from openveil.detect.run import detect_video

        prop, ver = FakeProposer(), FakeVerifier(cfg.verifier.threshold)
        out = os.path.join(self.tmp, "out", f"t{len(os.listdir(self.tmp))}.json.gz")
        table = detect_video(self.video, cfg, prop, ver, out_path=out, **kw)
        return table, prop, ver, out

    def test_day_schedule_dense_on_hits_and_interpolation(self):
        cfg = small_cfg()
        table, prop, ver, out = self.run_detect(cfg)
        # table written, valid, named after the file, no source path
        saved = load_table(out, validate=True)
        self.assertEqual(saved.video, "cam1_20240305-120000")
        self.assertIsNone(saved.source)
        self.assertEqual((saved.width, saved.height, saved.fps), (160, 120, 10.0))
        self.assertEqual(saved.chunks, table.chunks)
        self.assertEqual([c[1] for c in table.chunks], ["full", "full", "full"])
        self.assertEqual([c[2] for c in table.chunks], [50, 50, 20])
        self.assertEqual(sum(c[3] for c in table.chunks), sum(prop.calls))
        self.assertEqual(table.params["base_s"], 1.0)
        self.assertEqual(table.params["mask_thr"], cfg.verifier.threshold)

        boxes = _boxes_by_time(table)
        n = int(12.0 * FPS)
        # far fewer detector calls than frames, but every frame with the walker that lies between two detections has a box
        self.assertLess(sum(prop.calls), n / 2)
        walk = [round(i / FPS, 3) for i in range(n) if WALK_T0 <= i / FPS < WALK_T1]
        covered = [t for t in walk if t in boxes]
        self.assertEqual(covered, walk[:len(covered)])
        self.assertGreaterEqual(len(covered), len(walk) - 1)          # at most the last frame before it leaves
        for t in boxes:
            self.assertTrue(WALK_T0 <= t < WALK_T1, t)               # nothing outside the walker's window
        kinds = {b[4] for bx in boxes.values() for b in bx}
        self.assertEqual(kinds, {"det", "interp"})
        for t, bx in boxes.items():
            self.assertEqual(len(bx), 1, (t, bx))                     # the decoy is rejected, the duplicate merged
            b = bx[0]
            true = walker_box(t)
            for k in range(4):
                self.assertAlmostEqual(b[k], true[k], delta=1.6)
            if b[4] == "det":
                self.assertEqual(len(b), 6)
                self.assertAlmostEqual(b[5], 0.95, places=3)
            else:
                self.assertEqual(len(b), 5)
            self.assertLess(b[2], DECOY_BOX[0])

    def test_sparse_schedule_never_goes_dense(self):
        cfg = small_cfg()
        dn = cfg.schedule.day_night
        dn.policy, dn.day_start, dn.day_end, dn.margin_s = "clock", 6.0, 7.0, 0.0      # 12:00 is "night"
        cfg.schedule.night_base_s = 1.0
        table, prop, _ver, _out = self.run_detect(cfg)
        self.assertEqual(table.start_local, "2024-03-05T12:00:00")     # parsed from the file name
        self.assertEqual([c[1] for c in table.chunks], ["sparse"] * 3)
        # one call per checkpoint batch: checkpoints every 10 frames and the last frame of each chunk only,
        # no back-fill or dense calls
        self.assertEqual(sum(c[3] for c in table.chunks), 6 + 6 + 3)
        dets = sorted(t for t, bx in _boxes_by_time(table).items() if bx[0][4] == "det")
        self.assertEqual(dets, [3.0, 4.0, 4.9, 5.0, 6.0])                # 4.9 s: the last frame of the first chunk
        interp = [t for t, bx in _boxes_by_time(table).items() if bx[0][4] == "interp"]
        self.assertTrue(interp)

    def test_skip_mode_and_first_pts_from_first_frame(self):
        cfg = small_cfg()
        dn = cfg.schedule.day_night
        dn.policy, dn.day_start, dn.day_end, dn.margin_s, dn.night_mode = "clock", 6.0, 7.0, 0.0, "skip"
        table, prop, _ver, _out = self.run_detect(cfg, start_local="2024-03-05T06:59:56")
        # day ends at 07:00: only the first chunk (from 06:59:56) is scanned, the rest is skipped
        self.assertEqual([c[0] for c in table.chunks], [0.0])
        self.assertEqual(table.first_pts, 0)
        self.assertTrue(table.frames)
        self.assertLess(max(_boxes_by_time(table)), 5.0)
        dn.day_start, dn.day_end = 7.0, 8.0
        table2, _p, _v, _o = self.run_detect(cfg, start_local="2024-03-05T06:59:52")
        # day starts at 07:00 = 8 s into the video: the first chunk is skipped
        self.assertEqual([c[0] for c in table2.chunks], [5.0, 10.0])
        # times still count from the first frame of the video, not from the first scanned chunk
        self.assertEqual(table2.first_pts, 0)
        times = sorted(_boxes_by_time(table2))
        self.assertEqual(times[0], 5.0)
        self.assertLess(times[-1], WALK_T1)

    def test_verifier_none_mode_keeps_by_proposer_score(self):
        from openveil.detect.verifier import make_verifier
        from openveil.detect.run import detect_video

        cfg = small_cfg()
        cfg.verifier.mode = "none"
        cfg.verifier.threshold = 0.2           # keeps walker (0.9) and decoy (0.3) proposals
        ver = make_verifier(cfg.verifier)
        table = detect_video(self.video, cfg, FakeProposer(), ver)
        boxes = _boxes_by_time(table)
        self.assertIn(0.0, boxes)                                   # the decoy is now masked
        self.assertTrue(any(len(bx) == 2 for bx in boxes.values()))


@unittest.skipUnless(HAVE_AV, "PyAV is not installed")
class TestRunMany(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_runmany_")
        self.good = [make_video(os.path.join(self.tmp, f"good{i}.mp4"), seconds=6.0 + i) for i in range(2)]
        self.bad = os.path.join(self.tmp, "broken.mp4")
        with open(self.bad, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42 this is not a video" * 10)
        self.out = os.path.join(self.tmp, "tables")
        self.logs = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_in_process_resume_and_failure_ledger(self):
        from openveil.detect.run import Ledger, run_many
        from test_detect_fakes import counting_factory

        cfg = small_cfg()
        count = os.path.join(self.tmp, "count.txt")
        os.environ["OV_TEST_COUNT_FILE"] = count
        try:
            s1 = run_many(self.good + [self.bad], self.out, cfg, factory=counting_factory, in_process=True, log=self.logs.append)
            self.assertEqual(s1["done"], 2)
            self.assertEqual(s1["missing"], ["broken"])                 # failed twice (pass + retry), not yet abandoned
            led = Ledger(self.out, cfg.run.max_attempts)
            self.assertEqual(led.attempts("broken"), 2)
            with open(led.failed_path("broken"), encoding="utf-8") as f:
                rec = json.load(f)
            self.assertEqual(rec["video"], "broken")
            self.assertTrue(rec["last_error"])
            for g in ("good0", "good1"):
                load_table(os.path.join(self.out, f"{g}.json.gz"), validate=True)
            mtimes = {g: os.path.getmtime(os.path.join(self.out, f"{g}.json.gz")) for g in ("good0", "good1")}
            # second run: finished tables are skipped, the broken video fails a third time and is abandoned
            s2 = run_many(self.good + [self.bad], self.out, cfg, factory=counting_factory, in_process=True, log=self.logs.append)
            self.assertEqual(s2["todo_at_start"], 1)
            self.assertEqual(s2["abandoned"], ["broken"])
            self.assertEqual(s2["missing"], [])
            self.assertEqual(led.attempts("broken"), 3)
            for g, m in mtimes.items():
                self.assertEqual(os.path.getmtime(os.path.join(self.out, f"{g}.json.gz")), m)
            # third run: nothing to do, the models are not even loaded
            s3 = run_many(self.good + [self.bad], self.out, cfg, factory=counting_factory, in_process=True, log=self.logs.append)
            self.assertEqual(s3["todo_at_start"], 0)
            with open(count, encoding="utf-8") as f:
                self.assertEqual(f.read().count("build"), 2)
            with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["abandoned"], ["broken"])
        finally:
            os.environ.pop("OV_TEST_COUNT_FILE", None)

    def test_duplicate_names_refused(self):
        from openveil.detect.run import run_many

        sub = os.path.join(self.tmp, "sub")
        os.makedirs(sub)
        dup = shutil.copy(self.good[0], os.path.join(sub, "good0.mp4"))
        with self.assertRaises(ValueError):
            run_many([self.good[0], dup], self.out, small_cfg(), in_process=True, log=self.logs.append)

    def test_partition_balances_cost(self):
        from openveil.detect.run import partition

        lists = partition([dict(n=i, cost=c) for i, c in enumerate([1, 9, 10, 1, 8, 1])], 2, lambda v: v["cost"])
        # largest first, each to the lightest list (ties: the first list)
        self.assertEqual([[v["cost"] for v in lst] for lst in lists], [[10, 1, 1, 1], [9, 8]])
        self.assertEqual(partition([], 3, lambda v: 1.0), [[], [], []])

    def test_worker_processes_restart_and_abandon(self):
        from openveil.detect.run import run_many
        from test_detect_fakes import fake_factory

        cfg = small_cfg(stall_s=120.0)
        s = run_many(self.good + [self.bad], self.out, cfg, workers=2, factory=fake_factory, log=self.logs.append, poll_s=0.2)
        self.assertEqual(s["done"], 2, self.logs)
        self.assertEqual(s["abandoned"], ["broken"], self.logs)
        self.assertEqual(s["missing"], [])
        self.assertEqual(sum(s["restarts"].values()), 1, self.logs)       # the worker with the broken video ran twice
        self.assertTrue(os.listdir(os.path.join(self.out, "workers")))

    def test_stall_watchdog_quarantines_a_hanging_video(self):
        from openveil.detect.run import run_many
        from test_detect_fakes import fake_factory

        # the hanging video is the longest, so the single worker starts with it and never gets further
        hang = make_video(os.path.join(self.tmp, "hang.mp4"), seconds=8.0, background=HANG_BACKGROUND)
        cfg = small_cfg(stall_s=10.0)            # the stall clock starts once the models are built
        s = run_many([hang, self.good[0]], self.out, cfg, workers=1, factory=fake_factory, log=self.logs.append, poll_s=0.2)
        self.assertEqual(s["quarantined"], ["hang"], self.logs)
        self.assertEqual(s["abandoned"], ["hang"])
        self.assertEqual(s["done"], 1)                                  # the run went on with the other video
        self.assertEqual(sum("killing it" in m for m in self.logs), 3, self.logs)
        self.assertEqual(s["restarts"], {0: 3})


if __name__ == "__main__":
    unittest.main()
