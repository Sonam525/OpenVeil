# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.geometry."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil import geometry as g  # noqa: E402


class TestGeometry(unittest.TestCase):
    def test_area_and_inter(self):
        self.assertEqual(g.area([0, 0, 10, 20]), 200)
        self.assertEqual(g.area([10, 0, 0, 20, "det"]), 0.0)                       # inverted: zero, extra field ignored
        self.assertEqual(g.inter([0, 0, 10, 10], [5, 5, 20, 20]), 25)
        self.assertEqual(g.inter([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)            # touching edges
        self.assertEqual(g.inter([0, 0, 10, 10], [30, 30, 40, 40]), 0.0)

    def test_iou(self):
        self.assertEqual(g.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(g.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)
        self.assertEqual(g.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(g.iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)                   # empty union
        rnd = random.Random(1)
        for _ in range(500):
            a = [rnd.uniform(0, 50), rnd.uniform(0, 50)]
            a += [a[0] + rnd.uniform(0.1, 50), a[1] + rnd.uniform(0.1, 50)]
            b = [rnd.uniform(0, 50), rnd.uniform(0, 50)]
            b += [b[0] + rnd.uniform(0.1, 50), b[1] + rnd.uniform(0.1, 50)]
            v = g.iou(a, b)
            self.assertTrue(0.0 <= v <= 1.0)
            self.assertEqual(v, g.iou(b, a))
            self.assertAlmostEqual(v, g.inter(a, b) / (g.area(a) + g.area(b) - g.inter(a, b)))

    def test_clip_and_share(self):
        self.assertEqual(g.clip_box([-10, -5, 50, 200, "det"], 100, 100), [0.0, 0.0, 50, 100])
        self.assertEqual(g.clip_box([120, 10, 160, 50], 100, 100), [100, 10, 100, 50])   # outside: zero area
        self.assertEqual(g.share([-50, -50, 150, 60], 100, 100), 0.6)
        self.assertEqual(g.share([0, 0, 1152, 648], 2304, 1296), 0.25)

    def test_contains_frac(self):
        self.assertEqual(g.contains_frac([10, 10, 20, 20], [0, 0, 100, 100]), 1.0)
        self.assertEqual(g.contains_frac([90, 0, 110, 10], [0, 0, 100, 100]), 0.5)
        self.assertEqual(g.contains_frac([5, 5, 5, 9], [0, 0, 100, 100]), 0.0)    # zero area never counts as inside

    def test_union_box(self):
        self.assertEqual(g.union_box([[10, 20, 30, 40, "det"], [5, 25, 15, 60]]), [5, 20, 30, 60])
        with self.assertRaises(ValueError):
            g.union_box([])

    def test_enlarge(self):
        self.assertEqual(g.enlarge([100, 100, 200, 300], 0.2, 1000, 1000), [80.0, 60.0, 220.0, 340.0])
        self.assertEqual(g.enlarge([0, 10, 100, 60], 0.2, 110, 65), [0.0, 0.0, 110, 65])   # clipped to the image
        self.assertEqual(g.enlarge([10, 10, 20, 20], 0.0, 100, 100), [10.0, 10.0, 20, 20])

    def test_scale_boxes(self):
        self.assertEqual(g.scale_boxes([[10, 20, 30, 40, "det"]], 0.5, 0.25), [[5.0, 5.0, 15.0, 10.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
