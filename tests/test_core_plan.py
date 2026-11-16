# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.apply.plan.MaskPlan (giant rule, enlargement, hold, intervals)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil.apply.plan import MaskPlan, block_size  # noqa: E402
from openveil.config import config_from_dict  # noqa: E402
from openveil.table import BoxTable, save_table  # noqa: E402

W, H = 1000, 500
GIANT = [0, 0, 1000, 300, "interp"]                  # 60% of the image
LEFT = [100, 100, 200, 300, "det"]
RIGHT = [700, 100, 800, 300, "det"]


def table(rows):
    """rows: [(t_s, boxes)]; time base 1/1000, first pts 5000"""
    return BoxTable(video="v", width=W, height=H, fps=25.0, time_base=(1, 1000), first_pts=5000,
                    frames=[[5000 + int(round(t * 1000)), boxes] for t, boxes in rows])



class TestMaskPlan(unittest.TestCase):
    def test_giant_rule_enlargement_and_hold(self):
        plan = MaskPlan(table([(10.0, [GIANT, LEFT, RIGHT])]))
        self.assertEqual(plan.rule_stats["boxes_dropped"], 1)
        got = plan.boxes_at(10.0)
        # 20% of 100 x 200 on each side: 20 px left/right, 40 px top/bottom
        self.assertEqual(got, [[80.0, 60.0, 220.0, 340.0], [680.0, 60.0, 820.0, 340.0]])
        self.assertEqual(plan.boxes_at(12.0), got)                         # hold +-2 s, inclusive
        self.assertEqual(plan.boxes_at(8.0), got)
        self.assertEqual(plan.boxes_at(12.01), [])
        self.assertEqual(plan.boxes_at(7.99), [])
        self.assertTrue(plan)
        self.assertFalse(MaskPlan(table([])))

    def test_giant_threshold_is_0_55(self):
        g50 = [0, 50, 1000, 300, "det"]                                   # 50%: not a giant at the production value
        plan = MaskPlan(table([(0.0, [g50, LEFT, RIGHT])]))
        self.assertEqual(len(plan.boxes_at(0.0)), 3)
        plan = MaskPlan(table([(0.0, [g50, LEFT, RIGHT])]), min_share=0.5)
        self.assertEqual(len(plan.boxes_at(0.0)), 2)
        cfg = config_from_dict({"apply": {"min_share": 0.45}})
        self.assertEqual(len(MaskPlan(table([(0.0, [g50, LEFT, RIGHT])]), cfg).boxes_at(0.0)), 2)
        # a reviewer's manual giant is never dropped
        manual = [0, 0, 1000, 300, "manual"]
        self.assertEqual(len(MaskPlan(table([(0.0, [manual, LEFT, RIGHT])])).boxes_at(0.0)), 3)

    def test_clipping_at_the_image_border(self):
        plan = MaskPlan(table([(0.0, [[0, 450, 100, 500, "det"]])]))
        self.assertEqual(plan.boxes_at(0.0), [[0.0, 440.0, 120.0, 500]])

    def test_hold_merges_rows_and_removes_duplicates(self):
        a, b = [10, 10, 60, 110, "det"], [500, 10, 550, 110, "interp"]
        plan = MaskPlan(table([(1.0, [a]), (2.0, [a, b]), (5.0, [b])]))
        self.assertEqual(len(plan.boxes_at(1.0)), 2)                       # a (from 1 s and 2 s, once) + b (from 2 s)
        self.assertEqual(len(plan.boxes_at(3.5)), 2)                       # a and b from 2 s; b from 5 s is the same box
        self.assertEqual(plan.boxes_at(6.5), [plan.boxes[2][0]])
        self.assertEqual(plan.times, [1.0, 2.0, 5.0])
        no_hold = MaskPlan(table([(1.0, [a]), (2.0, [a, b])]), hold_s=0.0)
        self.assertEqual(no_hold.boxes_at(1.5), [])
        self.assertEqual(len(no_hold.boxes_at(2.0)), 2)

    def test_intervals(self):
        plan = MaskPlan(table([(1.0, [LEFT]), (4.0, [LEFT]), (10.0, [LEFT])]))
        self.assertEqual(plan.intervals(), [(0.0, 6.0), (8.0, 12.0)])
        self.assertEqual(plan.intervals(duration=11.0), [(0.0, 6.0), (8.0, 11.0)])
        self.assertEqual(plan.masked_seconds(), 10.0)
        self.assertEqual(MaskPlan(table([])).intervals(), [])

    def test_inputs_dict_path_and_object_agree(self):
        t = table([(1.0, [GIANT, LEFT, RIGHT]), (1.5, [LEFT])])
        ref = MaskPlan(t)
        self.assertEqual(MaskPlan(t.to_dict()).boxes, ref.boxes)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.json.gz")
            save_table(t, p)
            self.assertEqual(MaskPlan(p).boxes, ref.boxes)
        self.assertEqual((ref.W, ref.H, ref.width, ref.height), (W, H, W, H))

    def test_block_size(self):
        self.assertEqual([block_size(h) for h in (360, 480, 720, 1080, 1296, 2160)], [16, 16, 20, 30, 36, 60])
        self.assertEqual(MaskPlan(table([])).block(), 16)                  # 500 / 36 = 13.9 -> at least 16


if __name__ == "__main__":
    unittest.main(verbosity=2)
