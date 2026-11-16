# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Synthetic test videos (moving rectangles on a textured background, no people) and the
helpers the other ``test_video_*`` modules share. Also checks the generator itself."""
from __future__ import annotations

import gzip
import json
import os
import unittest
from fractions import Fraction

import numpy as np

try:
    import av  # noqa: F401
    HAVE_AV = True
except ImportError:  # pragma: no cover
    HAVE_AV = False

W, H, FPS, GOP = 320, 240, 10, 10


def has_encoder(name: str) -> bool:
    """Whether PyAV's FFmpeg build has encoder ``name``."""
    if not HAVE_AV:
        return False
    import av
    return name in av.codecs_available


def synth_frame(i: int, w: int = W, h: int = H) -> np.ndarray:
    """RGB frame ``i``: a fine checkerboard texture (so a mosaic is easy to detect) with two
    moving coloured rectangles."""
    yy, xx = np.mgrid[0:h, 0:w]
    tex = (((xx // 2) + (yy // 2)) % 2) * 120 + 60
    img = np.stack([tex, (tex + 40) % 256, 255 - tex], axis=-1).astype(np.uint8)
    x = (10 + 7 * i) % (w - 60)
    img[40:100, x:x + 50] = (220, 30, 30)
    y = (20 + 5 * i) % (h - 50)
    img[y:y + 40, 200:260] = (30, 200, 60)
    return img


def make_video(path: str, codec: str = "libx264", seconds: float = 8.0, fps: int = FPS, gop: int = GOP,
               bframes: int = 0, repeat_headers: bool = True, w: int = W, h: int = H) -> str:
    """Write an MP4 test video with a fixed GOP length.

    ``repeat_headers`` puts the parameter sets in every keyframe (like most cameras);
    ``bframes`` > 0 gives a stream with frame reordering.
    """
    import av

    n = int(round(seconds * fps))
    with av.open(path, "w", format="mp4") as oc:
        st = oc.add_stream(codec, rate=fps)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.codec_context.gop_size = gop
        st.codec_context.max_b_frames = bframes
        if codec == "libx264":
            params = f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes={bframes}"
            if repeat_headers:
                params += ":repeat-headers=1"
            st.codec_context.options = {"preset": "veryfast", "crf": "23", "x264-params": params}
        elif codec == "libx265":
            params = f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes={bframes}:info=0:log-level=error"
            if repeat_headers:
                params += ":repeat-headers=1"
            st.codec_context.options = {"preset": "ultrafast", "x265-params": params}
        else:
            st.bit_rate = 400_000
        for i in range(n):
            fr = av.VideoFrame.from_ndarray(synth_frame(i, w, h), format="rgb24")
            fr.pts = i
            fr.time_base = Fraction(1, fps)
            for p in st.encode(fr):
                oc.mux(p)
        for p in st.encode(None):
            oc.mux(p)
    return path


def stream_facts(path: str):
    """``(time_base, first_pts, [packet pts in decode order])`` of the video stream."""
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        pts = [p.pts for p in c.demux(st) if p.size > 0]
        tb = st.time_base
    return tb, min(pts), pts


def table_dict(path: str, rows, kind_tag: str = "det"):
    """A schema-v1 box table dict for test video ``path``.

    ``rows``: ``[(t_seconds, [[x1, y1, x2, y2], ...]), ...]``; each time is snapped to the
    nearest frame pts of the video.
    """
    tb, first, pts = stream_facts(path)
    sp = sorted(pts)
    frames = []
    for t, boxes in rows:
        target = first + t / float(tb)
        p = min(sp, key=lambda q: abs(q - target))
        frames.append([int(p), [[float(v) for v in b[:4]] + [kind_tag] for b in boxes]])
    frames.sort(key=lambda r: r[0])
    return {"schema": "openveil.boxes/1", "video": os.path.basename(path), "kind": "video",
            "fps": float(FPS), "width": W, "height": H, "time_base": [tb.numerator, tb.denominator],
            "first_pts": int(first), "start_local": None, "params": {},
            "chunks": [[0.0, "full", len(pts), 0]], "frames": frames}


def write_table(table: dict, path: str) -> str:
    """Write a table dict as ``.json.gz``."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(table, f)
    return path


def decoded_frames(path: str, fmt: str = "gray"):
    """``[(pts, array)]`` of every decoded frame of ``path``."""
    import av
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            out.append((f.pts, f.to_ndarray(format=fmt)))
    return out


@unittest.skipUnless(HAVE_AV, "PyAV not installed")
class TestSynth(unittest.TestCase):
    def test_generator(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = make_video(os.path.join(d, "a.mp4"), seconds=2.0)
            tb, first, pts = stream_facts(p)
            self.assertEqual(len(pts), 20)
            self.assertEqual(pts, sorted(pts))
            t = table_dict(p, [(0.5, [[10, 10, 50, 50]])])
            self.assertEqual(len(t["frames"]), 1)
            self.assertAlmostEqual(float((t["frames"][0][0] - first) * tb), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
