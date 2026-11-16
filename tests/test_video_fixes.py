# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Apply-step cases that must not leave a person unmasked or lose data (synthetic data only):

* video heights that are 2 mod 4 can be masked;
* sources that are not 8-bit 4:2:0 are re-encoded in full, not spliced;
* a table whose clock does not match the video is refused;
* masking a video without boxes in place keeps the file;
* image names: ``./`` prefixes, letter case and leading folders are matched; boxes for one image
  under two names are merged; names that match nothing make the run fail;
* JPEGs with an EXIF orientation: boxes are in upright pixels, boxes outside the image are refused;
* a box marked ``manual`` (confirmed by a reviewer) survives the giant rule of ``apply images``;
* the boxes file of ``review build`` (scores, layouts) is read the same way by ``apply images``.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest

import numpy as np
from test_video_synth import HAVE_AV, has_encoder, make_video, table_dict

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False


@unittest.skipUnless(HAVE_AV and has_encoder("libx264"), "PyAV with libx264 needed")
class TestVideoFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_vfix_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_height_two_mod_four(self):
        from openveil.apply.video import mask_video
        p = make_video(os.path.join(self.tmp, "v.mp4"), seconds=2.0, w=320, h=270)
        t = table_dict(p, [(0.5, [[10, 10, 80, 80]])])
        t["width"], t["height"] = 320, 270
        r = mask_video(p, t, os.path.join(self.tmp, "o.mp4"))
        self.assertTrue(r["ok"], r)
        self.assertGreater(r["attempt1"]["stats"]["frames_mosaicked"], 0)

    def test_pixel_format_other_than_420_is_reencoded_in_full(self):
        import av
        from fractions import Fraction
        from openveil.apply.video import mask_video, probe_eligibility
        p = os.path.join(self.tmp, "v422.mp4")
        with av.open(p, "w", format="mp4") as oc:
            st = oc.add_stream("libx264", rate=10)
            st.width, st.height, st.pix_fmt = 320, 240, "yuv422p"
            st.codec_context.options = {"x264-params": "keyint=10:min-keyint=10:scenecut=0:bframes=0:repeat-headers=1"}
            for i in range(30):
                fr = av.VideoFrame.from_ndarray(np.full((240, 320, 3), (i * 7) % 255, np.uint8), format="rgb24")
                fr.pts, fr.time_base = i, Fraction(1, 10)
                for k in st.encode(fr):
                    oc.mux(k)
            for k in st.encode(None):
                oc.mux(k)
        e = probe_eligibility(p)
        self.assertFalse(e.smart_cut)
        self.assertTrue(any("pixel format" in x for x in e.reasons), e.reasons)
        r = mask_video(p, table_dict(p, [(1.0, [[10, 10, 80, 80]])]), os.path.join(self.tmp, "o.mp4"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["mode"], "full")
        with av.open(os.path.join(self.tmp, "o.mp4")) as c:
            self.assertEqual({f.format.name for f in c.decode(video=0)}, {"yuv420p"})

    def test_table_from_another_container_is_refused(self):
        import av
        from openveil.apply.video import mask_video
        src = make_video(os.path.join(self.tmp, "v.mp4"), seconds=4.0)
        t = table_dict(src, [(2.0, [[10, 10, 120, 120]])])
        remux = os.path.join(self.tmp, "v.mkv")
        with av.open(src) as ic, av.open(remux, "w") as oc:
            ist = ic.streams.video[0]
            ost = oc.add_stream_from_template(ist)
            for k in ic.demux(ist):
                if k.dts is None:
                    continue
                k.stream = ost
                oc.mux(k)
        with self.assertRaises(ValueError):
            mask_video(remux, t, os.path.join(self.tmp, "o.mp4"))
        t2 = dict(t, first_pts=t["first_pts"] + 1)                   # a pts that is no frame of the video
        with self.assertRaises(ValueError):
            mask_video(src, t2, os.path.join(self.tmp, "o2.mp4"))
        self.assertTrue(mask_video(src, t, os.path.join(self.tmp, "o3.mp4"))["ok"])

    def test_in_place_without_boxes_keeps_the_file(self):
        from openveil.apply.video import mask_video
        p = make_video(os.path.join(self.tmp, "v.mp4"), seconds=2.0)
        size = os.path.getsize(p)
        r = mask_video(p, table_dict(p, []), p)
        self.assertTrue(r["ok"])
        self.assertEqual((os.path.getsize(p), r["bytes"]), (size, size))


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _noise(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


@unittest.skipUnless(HAVE_PIL, "Pillow needed")
class TestImageFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_ifix_")
        self.src = os.path.join(self.tmp, "src")
        os.makedirs(os.path.join(self.src, "cam1"))
        self.img = _noise()
        with open(os.path.join(self.src, "cam1", "f001.png"), "wb") as f:
            f.write(_png(self.img))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _out(self, path):
        return np.asarray(Image.open(path).convert("RGB"))

    def test_dot_slash_tar_members_and_case(self):
        from openveil.apply.images import mask_images
        tar = os.path.join(self.tmp, "x.tar")
        with tarfile.open(tar, "w") as t:
            t.add(self.src, arcname=".")
        for key in ("cam1/f001.png", "./cam1/f001.png", "CAM1/F001.PNG", "f001.png", "frames/cam1/f001.png"):
            out = os.path.join(self.tmp, f"o_{abs(hash(key))}.tar")
            r = mask_images(tar, {key: [[10, 10, 60, 60]]}, out, rules=False)
            self.assertTrue(r["ok"], (key, r))
            self.assertEqual((r["masked_planned"], r["planned_absent"]), (1, []), key)

    def test_unmatched_names_fail_and_write_nothing(self):
        from openveil.cli import main
        bj = os.path.join(self.tmp, "b.json")
        with open(bj, "w") as f:
            json.dump({"other/f001.png": [[10, 10, 60, 60]]}, f)
        out = os.path.join(self.tmp, "out")
        self.assertEqual(main(["apply", "images", self.src, "--boxes", bj, "--out", out]), 1)
        self.assertFalse(os.path.exists(os.path.join(out, "cam1", "f001.png")))
        self.assertEqual(main(["apply", "images", self.src, "--boxes", bj, "--out", out, "--allow-missing"]), 0)

    def test_boxes_under_two_names_are_merged(self):
        from openveil.apply.images import mask_images
        out = os.path.join(self.tmp, "out")
        r = mask_images(self.src, {"cam1/f001.png": [[0, 0, 40, 40]], "f001.png": [[100, 60, 150, 110]]}, out, rules=False)
        self.assertTrue(r["ok"], r)
        o = self._out(os.path.join(out, "cam1", "f001.png"))
        self.assertFalse(np.array_equal(o[0:40, 0:40], self.img[0:40, 0:40]))
        self.assertFalse(np.array_equal(o[60:110, 100:150], self.img[60:110, 100:150]))

    def test_manual_box_survives_the_giant_rule(self):
        from openveil.apply.images import load_boxes_json, mask_images
        big, a, b = [0, 0, 160, 80], [10, 10, 40, 70], [100, 10, 130, 70]
        for kind, changed in (("det", False), ("manual", True)):
            bj = os.path.join(self.tmp, f"{kind}.json")
            with open(bj, "w") as f:
                json.dump({"cam1/f001.png": [big + [kind], a + [kind], b + [kind]]}, f)
            out = os.path.join(self.tmp, f"out_{kind}")
            r = mask_images(self.src, load_boxes_json(bj), out)
            self.assertTrue(r["ok"], r)
            o = self._out(os.path.join(out, "cam1", "f001.png"))
            # a pixel inside the big box only (outside the enlarged inner boxes)
            self.assertEqual(not np.array_equal(o[5, 75], self.img[5, 75]), changed, kind)

    def test_scored_boxes_follow_the_review_threshold(self):
        from openveil.apply.images import load_boxes_json
        bj = os.path.join(self.tmp, "s.json")
        with open(bj, "w") as f:
            json.dump({"images": [{"image": "a.jpg", "boxes": [[1, 1, 9, 9, 0.07], {"box": [2, 2, 8, 8], "score": 0.9},
                                                               [3, 3, 7, 7]]}]}, f)
        self.assertEqual(load_boxes_json(bj), {"a.jpg": [[2.0, 2.0, 8.0, 8.0], [3.0, 3.0, 7.0, 7.0]]})

    def test_exif_orientation(self):
        from openveil.apply.images import mask_image_bytes
        from openveil.review.build import image_size
        im = Image.fromarray(_noise(120, 200))                          # stored 200 x 120
        exif = Image.Exif()
        exif[0x0112] = 6                                                  # rotate 90: upright 120 x 200
        p = os.path.join(self.tmp, "r.jpg")
        im.save(p, format="JPEG", exif=exif)
        self.assertEqual(image_size(p), (120, 200))                       # the review frame is upright too
        with open(p, "rb") as f:
            data = f.read()
        with self.assertRaises(ValueError):                               # stored-pixel box beyond x = 120
            mask_image_bytes(data, [[150, 10, 195, 60]], "r.jpg", rules=False)
        self.assertTrue(mask_image_bytes(data, [[10, 150, 100, 195]], "r.jpg", rules=False))


if __name__ == "__main__":
    unittest.main()
