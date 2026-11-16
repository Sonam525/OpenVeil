# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.apply.mosaic against a slow per-cell reference."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil.apply.mosaic import block_size, cells, mosaic_array, mosaic_yuv420  # noqa: E402

try:
    import av  # noqa: F401
    HAVE_AV = True
except ImportError:
    HAVE_AV = False


def reference(img, boxes, B):
    """the mosaic written out cell by cell: every cell touching a box gets the mean of its pixels"""
    H, W = img.shape[:2]
    for r in range(0, H, B):
        for c in range(0, W, B):
            touched = False
            for b in boxes:
                x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                if x1 < c + B and x2 > c and y1 < r + B and y2 > r:      # open overlap with the cell
                    touched = True
                    break
            if touched:
                cell = img[r:r + B, c:c + B]
                cell[...] = np.round(cell.reshape(-1, *img.shape[2:]).astype(np.float64).mean(axis=0)).astype(img.dtype)
    return img


def random_boxes(rng, W, H, n):
    boxes = []
    for _ in range(n):
        x, y = rng.uniform(-20, W), rng.uniform(-20, H)
        boxes.append([x, y, x + rng.uniform(1, 150), y + rng.uniform(1, 150)])
    return boxes


class TestMosaic(unittest.TestCase):
    def test_against_reference(self):
        rng = np.random.default_rng(0)
        worst = 0
        for k in range(300):
            H, W = int(rng.integers(20, 300)), int(rng.integers(20, 300))
            B = int(rng.integers(2, 40))
            img = rng.integers(0, 256, (H, W) if k % 2 else (H, W, 3), dtype=np.uint8)
            boxes = random_boxes(rng, W, H, int(rng.integers(1, 5)))
            a = mosaic_array(img.copy(), boxes, B)
            b = reference(img.copy(), boxes, B)
            diff = int(np.abs(a.astype(int) - b.astype(int)).max())
            worst = max(worst, diff)
            # float32 cell means may round the other way at exactly .5: never more than one level
            self.assertLessEqual(diff, 1, (k, H, W, B))
            changed_a, changed_b = (a != img), (b != img)
            self.assertTrue(np.all(changed_a <= (changed_b | (np.abs(a.astype(int) - img.astype(int)) <= 1))))
        self.assertLessEqual(worst, 1)

    def test_cells_on_grid_lines(self):
        g = cells([[20, 20, 40, 40]], 100, 100, 20)                     # exactly one cell
        self.assertEqual(int(g.sum()), 1)
        self.assertTrue(g[1, 1])
        g = cells([[19.5, 20, 40, 40.5]], 100, 100, 20)                 # touches its neighbours
        self.assertEqual(int(g.sum()), 4)
        self.assertEqual(int(cells([[120, 10, 160, 50]], 100, 100, 20).sum()), 0)     # outside
        self.assertEqual(int(cells([[30, 30, 30, 50]], 100, 100, 20).sum()), 0)       # zero width

    def test_untouched_pixels_and_order(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 50], [40, 30, 90, 100], [150, 0, 210, 20]]
        a = mosaic_array(img.copy(), boxes, 16)
        b = mosaic_array(img.copy(), list(reversed(boxes)), 16)
        self.assertTrue(np.array_equal(a, b))                            # order of the boxes does not matter
        g = cells(boxes, 200, 120, 16)
        mask = np.repeat(np.repeat(g, 16, 0), 16, 1)[:120, :200]
        self.assertTrue(np.array_equal(a[~mask], img[~mask]))            # pixels of untouched cells are unchanged
        cell = a[16:32, 16:32].reshape(-1, 3)
        self.assertTrue(np.all(cell == cell[0]))                         # a touched cell is flat
        same = img.copy()
        self.assertIs(mosaic_array(same, [], 16), same)
        self.assertTrue(np.array_equal(mosaic_array(same, [[300, 300, 400, 400]], 16), img))

    def test_default_block_and_float_images(self):
        img = np.zeros((720, 40), np.uint8)
        img[:, :20] = 200
        out = mosaic_array(img.copy(), [[0, 0, 40, 20]])                 # block 20 at 720p: two cells per row of cells
        self.assertTrue(np.all(out[:20, :20] == 200) and np.all(out[:20, 20:] == 0))
        self.assertEqual(block_size(720), 20)
        f = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        m = mosaic_array(f.copy(), [[0, 0, 4, 4]], 4)
        self.assertAlmostEqual(float(m[0, 0]), float(f[:4, :4].mean()), places=6)

    def test_yuv420_planes_and_packed_buffer(self):
        rng = np.random.default_rng(2)
        H, W, B = 96, 128, 16
        y = rng.integers(0, 256, (H, W), dtype=np.uint8)
        u = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
        v = rng.integers(0, 256, (H // 2, W // 2), dtype=np.uint8)
        boxes = [[10, 20, 60, 70], [90, 5, 127, 40]]
        planes = (y.copy(), u.copy(), v.copy())
        self.assertIs(mosaic_yuv420(planes, boxes, B), planes)
        self.assertTrue(np.array_equal(planes[0], mosaic_array(y.copy(), boxes, B)))
        half = [[b[0] / 2, b[1] / 2, b[2] / 2, b[3] / 2] for b in boxes]
        self.assertTrue(np.array_equal(planes[1], mosaic_array(u.copy(), half, B // 2)))
        self.assertTrue(np.array_equal(planes[2], mosaic_array(v.copy(), half, B // 2)))
        packed = np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(H * 3 // 2, W)
        out = mosaic_yuv420(packed, boxes, B)                            # in place on the packed buffer
        self.assertIs(out, packed)
        flat = packed.reshape(-1)
        self.assertTrue(np.array_equal(flat[:H * W].reshape(H, W), planes[0]))
        self.assertTrue(np.array_equal(flat[H * W:H * W + H * W // 4].reshape(H // 2, W // 2), planes[1]))
        self.assertTrue(np.array_equal(flat[H * W + H * W // 4:].reshape(H // 2, W // 2), planes[2]))
        with self.assertRaises(ValueError):
            mosaic_yuv420(np.zeros((10, 10, 3), np.uint8), boxes, B)
        with self.assertRaises(ValueError):
            mosaic_yuv420(np.asfortranarray(packed[:, :-1]), boxes, B)

    @unittest.skipUnless(HAVE_AV, "PyAV not installed")
    def test_pyav_frame_layout(self):
        import av
        rng = np.random.default_rng(3)
        H, W = 72, 96
        a = rng.integers(0, 256, (H * 3 // 2, W), dtype=np.uint8)
        fr = av.VideoFrame.from_ndarray(a, format="yuv420p")
        buf = fr.to_ndarray(format="yuv420p")
        self.assertTrue(np.array_equal(buf, a))
        mosaic_yuv420(buf, [[0, 0, 48, 36]], 20)
        y = buf[:H]
        self.assertTrue(np.all(y[:20, :20] == y[0, 0]))
        out = av.VideoFrame.from_ndarray(buf, format="yuv420p")
        self.assertEqual((out.width, out.height), (W, H))


if __name__ == "__main__":
    unittest.main(verbosity=2)
