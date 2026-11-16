# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.apply.quality on a synthetic smart cut."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from test_video_synth import GOP, HAVE_AV, has_encoder, make_video, table_dict


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openveil.apply.video import smart_cut
        from openveil.table import BoxTable
        cls.tmp = tempfile.mkdtemp(prefix="ov_quality_")
        cls.src = make_video(os.path.join(cls.tmp, "src.mp4"), seconds=6.0)
        cls.table = BoxTable.from_dict(table_dict(cls.src, [(0.5, [[40, 40, 140, 140]])]))
        cls.dst = os.path.join(cls.tmp, "out.mp4")
        cls.stats = smart_cut(cls.src, cls.table, cls.dst)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_report(self):
        from openveil.apply.quality import quality_report
        r = quality_report(self.src, self.dst, self.table, every=2)
        self.assertEqual(r["frames"], 60)
        self.assertEqual((r["frames_src"], r["frames_dst"]), (60, 60))
        copied = self.stats["packets"] - self.stats["packets_reencoded"]
        self.assertEqual(copied, 3 * GOP)
        self.assertGreaterEqual(r["frames_identical"], copied)
        self.assertGreater(r["sampled_reencoded"], 5)
        self.assertGreater(r["psnr_outside_mosaic_median"], 30.0)
        self.assertGreater(r["mosaic_pixels_changed_median"], 0.5)

    def test_identical_files(self):
        from openveil.apply.quality import quality_report
        r = quality_report(self.src, self.src, self.table)
        self.assertEqual(r["frames_identical"], 60)
        self.assertIsNone(r["psnr_outside_mosaic_median"])

    def test_cell_mask(self):
        from openveil.apply.quality import cell_mask
        m = cell_mask([[10, 10, 20, 20]], 64, 48, 16)
        self.assertEqual(m.shape, (48, 64))
        self.assertEqual(int(m.sum()), 32 * 32)             # cells (0..1, 0..1)
        self.assertTrue(m[0, 0] and m[31, 31] and not m[32, 32])
        self.assertFalse(cell_mask([], 64, 48, 16).any())
        self.assertIsInstance(m, np.ndarray)


if __name__ == "__main__":
    unittest.main()
