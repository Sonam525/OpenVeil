# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.apply.images: tar and folder rewriting of synthetic JPEG/PNG images,
boxes JSON, image -> video time mapping, and the Pillow fallback."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:  # pragma: no cover
    HAVE_CV2 = False
try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False

from test_video_synth import HAVE_AV, has_encoder, make_video, table_dict

HAVE_CODEC = HAVE_CV2 or HAVE_PIL


def texture(h=120, w=160, channels=3, seed=0):
    """A textured synthetic image (stripes and a gradient; no people)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx // 8 + yy // 8 + seed) % 2) * 140 + 50          # 8 px checker: JPEG-friendly
    chans = [base, np.clip(base + xx // 8, 0, 255), np.clip(base + yy // 8, 0, 255), np.full_like(base, 200)]
    return np.stack(chans[:channels], axis=-1).astype(np.uint8)


def encode(img, ext, quality=95):
    if HAVE_CV2:
        import cv2
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
        ok, b = cv2.imencode(ext, img, params)
        assert ok
        return b.tobytes()
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG" if ext == ".jpg" else "PNG", quality=quality)
    return buf.getvalue()


def decode(data, unchanged=False):
    if HAVE_CV2:
        import cv2
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR)
    from PIL import Image
    return np.array(Image.open(io.BytesIO(data)))


def add_member(tf, name, data, mtime=1700000000, mode=0o640):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode, ti.uname, ti.gname = len(data), mtime, mode, "lab", "lab"
    tf.addfile(ti, io.BytesIO(data))


BOX = [40.0, 30.0, 90.0, 80.0]


@unittest.skipUnless(HAVE_CODEC, "OpenCV or Pillow needed")
class TestTar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_img_")
        self.jpg_a = encode(texture(seed=1), ".jpg")
        self.jpg_b = encode(texture(seed=2)[:, ::-1].copy(), ".jpg")
        self.png = encode(texture(channels=4), ".png")
        self.src = os.path.join(self.tmp, "shard.tar")
        with tarfile.open(self.src, "w", format=tarfile.USTAR_FORMAT) as tf:
            d = tarfile.TarInfo("frames")
            d.type, d.mode, d.mtime = tarfile.DIRTYPE, 0o755, 1700000000
            tf.addfile(d)
            add_member(tf, "frames/0000001.jpg", self.jpg_a)
            add_member(tf, "frames/0000002.jpg", self.jpg_b, mtime=1700000100)
            add_member(tf, "frames/0000003.png", self.png)
            add_member(tf, "frames/notes.txt", b"synthetic notes\n")
            ln = tarfile.TarInfo("frames/latest.jpg")
            ln.type, ln.linkname, ln.mtime = tarfile.SYMTYPE, "0000002.jpg", 1700000000
            tf.addfile(ln)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def members(self, path):
        out = {}
        with tarfile.open(path) as t:
            for m in t.getmembers():
                out[m.name] = (m, t.extractfile(m).read() if m.isfile() else None)
        return out

    def test_rewrite_and_check(self):
        from openveil.apply.images import image_rules, mask_images
        from openveil.apply.mosaic import block_size, cells, mosaic_array
        dst = os.path.join(self.tmp, "out.tar")
        boxes = {"frames/0000001.jpg": [BOX], "0000003.png": [BOX + [0.9]], "frames/missing.jpg": [BOX],
                 "frames/0000002.jpg": []}
        # boxes for a name that matches no member: refused, nothing written
        r = mask_images(self.src, boxes, dst)
        self.assertFalse(r["ok"])
        self.assertEqual(r["planned_absent"], ["frames/missing.jpg"])
        self.assertFalse(os.path.exists(dst))
        r = mask_images(self.src, boxes, dst, allow_absent=True)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["kind"], "tar")
        self.assertEqual(r["masked_planned"], 2)
        self.assertEqual(r["planned_absent"], ["frames/missing.jpg"])
        self.assertEqual(r["stats"]["masked"], 2)
        with open(dst, "rb") as f:
            self.assertEqual(r["sha256"], hashlib.sha256(f.read()).hexdigest())
        a, b = self.members(self.src), self.members(dst)
        self.assertEqual(list(a), list(b))                                  # same order
        for name in ("frames/0000002.jpg", "frames/notes.txt"):
            self.assertEqual(a[name][1], b[name][1], name)                 # byte-identical
        for name, (m, _) in a.items():
            n = b[name][0]
            self.assertEqual((m.mtime, m.mode, m.type, m.uname, m.linkname), (n.mtime, n.mode, n.type, n.uname, n.linkname))
        self.assertEqual(b["frames/latest.jpg"][0].linkname, "0000002.jpg")
        # the mosaic is there, and only there
        orig = decode(self.jpg_a).astype(int)
        new = decode(b["frames/0000001.jpg"][1]).astype(int)
        h, w = orig.shape[:2]
        final = image_rules([BOX], w, h)
        self.assertEqual(final, [[30.0, 20.0, 100.0, 90.0]])               # +20% per side
        B = block_size(h)
        expect = mosaic_array(decode(self.jpg_a).copy(), final, B).astype(int)
        g = cells(final, w, h, B)
        m = np.kron(g, np.ones((B, B), bool))[:h, :w]
        self.assertLess(np.abs(new - expect)[m].mean(), 4.0)
        self.assertGreater(np.abs(new - orig)[m].mean(), 20.0)
        self.assertLess(np.abs(new - orig)[~m].mean(), 4.0)
        # PNG: lossless, alpha kept
        po, pn = decode(self.png, True), decode(b["frames/0000003.png"][1], True)
        self.assertEqual(po.shape, pn.shape)
        self.assertEqual(pn.shape[2], 4)
        mp = np.kron(cells(final, w, h, B), np.ones((B, B), bool))[:h, :w]
        self.assertTrue(np.array_equal(po[~mp], pn[~mp]))
        self.assertTrue(np.array_equal(pn[..., :3], mosaic_array(po.copy(), final, B)[..., :3]))

    def test_check_detects_changes(self):
        from openveil.apply.images import check_tar, rewrite_tar
        dst = os.path.join(self.tmp, "out.tar")
        plan = {"frames/0000001.jpg": [BOX]}
        st, names = rewrite_tar(self.src, dst, plan)
        self.assertEqual(st["members"], 6)
        self.assertEqual(check_tar(self.src, dst, plan), (True, ""))
        ok, why = check_tar(self.src, dst, {})                              # masked but not planned
        self.assertFalse(ok)
        self.assertIn("0000001.jpg", why)

    def test_no_boxes_gives_identical_members(self):
        from openveil.apply.images import mask_images
        dst = os.path.join(self.tmp, "out.tar")
        r = mask_images(self.src, {}, dst)
        self.assertTrue(r["ok"])
        a, b = self.members(self.src), self.members(dst)
        self.assertEqual({k: v[1] for k, v in a.items()}, {k: v[1] for k, v in b.items()})

    def test_giant_rule_on_image_boxes(self):
        from openveil.apply.images import image_rules
        raw = [[0, 0, 150, 110], [10, 10, 40, 50], [90, 40, 130, 100]]
        out = image_rules(raw, 160, 120)
        self.assertEqual(len(out), 2)                                       # the giant box is dropped
        self.assertEqual(image_rules([], 160, 120), [])


@unittest.skipUnless(HAVE_CODEC, "OpenCV or Pillow needed")
class TestFolder(unittest.TestCase):
    def test_folder(self):
        from openveil.apply.images import load_boxes_json, mask_images
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in")
            os.makedirs(os.path.join(src, "cam1"))
            files = {"cam1/a.jpg": encode(texture(), ".jpg"), "cam1/b.jpg": encode(texture()[::-1].copy(), ".jpg"),
                     "c.png": encode(texture(), ".png"), "readme.txt": b"x"}
            for rel, data in files.items():
                with open(os.path.join(src, rel), "wb") as f:
                    f.write(data)
            bj = os.path.join(d, "boxes.json")
            with open(bj, "w") as f:
                json.dump({"cam1/a.jpg": [BOX], "c.png": [[0, 0, 20, 20, "det", 0.8]]}, f)
            boxes = load_boxes_json(bj)
            out = os.path.join(d, "out")
            r = mask_images(src, boxes, out)
            self.assertTrue(r["ok"], r)
            self.assertEqual((r["kind"], r["stats"]["files"], r["stats"]["masked"]), ("folder", 4, 2))
            for rel in ("cam1/b.jpg", "readme.txt"):
                with open(os.path.join(out, rel), "rb") as f:
                    self.assertEqual(f.read(), files[rel])
            with open(os.path.join(out, "cam1/a.jpg"), "rb") as f:
                self.assertNotEqual(f.read(), files["cam1/a.jpg"])
            with self.assertRaises(ValueError):
                mask_images(src, boxes, out)                                # output not empty

    def test_boxes_json_validation(self):
        from openveil.apply.images import load_boxes_json
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            for bad in ([1, 2], {"a.jpg": [[1, 2, 3]]}, {"a.jpg": [[0, 0, "x", 1]]}, {"a.jpg": "x"}):
                with open(p, "w") as f:
                    json.dump(bad, f)
                with self.assertRaises(ValueError):
                    load_boxes_json(p)


@unittest.skipUnless(HAVE_PIL, "Pillow needed")
class TestPillowFallback(unittest.TestCase):
    def test_without_cv2(self):
        from openveil.apply import images
        data = encode(texture(), ".jpg")
        with mock.patch.object(images, "_have_cv2", return_value=False):
            out = images.mask_image_bytes(data, [BOX], "x.jpg")
            png = images.mask_image_bytes(encode(texture(channels=4), ".png"), [BOX], "x.png")
        self.assertTrue(out.startswith(b"\xff\xd8"))
        self.assertTrue(png.startswith(b"\x89PNG"))
        with self.assertRaises(ValueError):
            images.mask_image_bytes(data, [BOX], "x.gif")


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestImageMap(unittest.TestCase):
    def test_csv_map(self):
        from openveil.apply.images import boxes_from_map, load_image_map
        from openveil.apply.plan import MaskPlan
        from openveil.table import BoxTable, save_table
        with tempfile.TemporaryDirectory() as d:
            v = make_video(os.path.join(d, "clip01.mp4"), seconds=8.0)
            t = table_dict(v, [(1.0, [[40, 40, 140, 140]])])
            save_table(t, os.path.join(d, "clip01.json.gz"))
            p = os.path.join(d, "map.csv")
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image", "video", "t_s", "pts"])
                w.writerow(["f1.jpg", "clip01.mp4", "1.5", ""])
                w.writerow(["f2.jpg", "clip01", "", str(t["first_pts"] + 2 * 1024)])
                w.writerow(["f3.jpg", "clip01", "6.0", ""])                  # outside the hold
            m = load_image_map(p)
            self.assertEqual(m["f2.jpg"]["pts"], t["first_pts"] + 2048)
            got = boxes_from_map(m, d)
            plan = MaskPlan(BoxTable.from_dict(t))
            self.assertEqual(sorted(got), ["f1.jpg", "f2.jpg"])
            self.assertEqual(got["f1.jpg"], plan.boxes_at(1.5))
            got2 = boxes_from_map(m, {"clip01": BoxTable.from_dict(t)})
            self.assertEqual(got, got2)
            with open(p, "w", newline="") as f:
                f.write("image,video\nx.jpg,clip01\n")
            with self.assertRaises(ValueError):
                load_image_map(p)


if __name__ == "__main__":
    unittest.main()
