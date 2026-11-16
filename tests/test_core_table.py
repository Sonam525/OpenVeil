# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.table: schema, legacy tables, round trip, validation, time, overrides."""
import copy
import gzip
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil import table as tb  # noqa: E402
from openveil.table import BoxTable, TableError  # noqa: E402


def legacy_dict():
    """a table in the format written before the schema was versioned (no 'schema', no scores)"""
    return dict(video="clip01", kind="video", visit="site_x", source="/data/videos/clip01.mp4", fps=25.0, width=640, height=360,
                time_base=[1, 12800], first_pts=1024, start_local="2024-03-05T12:34:56",
                params={"base_s": 1.0, "mask_thr": 0.1}, chunks=[[0.0, "full", 1500, 70]],
                frames=[[1024, [[10.0, 20.0, 50.5, 120.0, "det"]]],
                        [1536, [[11.0, 20.0, 51.5, 121.0, "interp"], [300, 40, 340, 140, "det"]]],
                        [2048, [[12.0, 21.0, 52.5, 122.0, "det"]]]])


def new_table():
    return BoxTable(video="clip02", width=1280, height=720, fps=30.0, time_base=(1, 15360), first_pts=0,
                    params={"base_s": 1.0}, chunks=[[0.0, "full", 1800, 90]],
                    frames=[[512, [[100.0, 100.0, 200.0, 300.0, "det", 0.93]]],
                            [1024, [[101.0, 100.0, 201.0, 300.0, "interp"]]],
                            [1536, [[102.0, 100.0, 202.0, 300.0, "det", 0.88], [5.0, 5.0, 25.0, 45.0, "manual"]]]])


class TestRoundTrip(unittest.TestCase):
    def test_new_table_round_trip(self):
        t = new_table()
        t.validate()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clip02.json.gz")
            tb.save_table(t, p)
            back = tb.load_table(p, validate=True)
            self.assertEqual(back, t)
            raw = tb.read_table_dict(p)
            self.assertEqual(raw["schema"], tb.SCHEMA)
            self.assertEqual(list(raw)[:3], ["schema", "video", "kind"])
            self.assertEqual(raw["time_base"], [1, 15360])
            with open(p, "rb") as f:
                first = f.read()
            tb.save_table(back, p)                                             # reproducible bytes (no gzip timestamp)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), first)
            self.assertFalse(os.path.exists(p + ".tmp"))
            self.assertEqual(tb.list_tables(d), [p])
            plain = os.path.join(d, "clip02.json")                              # plain JSON is accepted too
            tb.save_table(t, plain)
            self.assertEqual(tb.load_table(plain), t)

    def test_legacy_table_without_schema(self):
        d0 = legacy_dict()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "legacy.json.gz")
            with gzip.open(p, "wt", encoding="utf-8") as f:
                json.dump(d0, f)
            t = tb.load_table(p, validate=True)
            self.assertEqual(t.schema, tb.SCHEMA)
            self.assertEqual((t.video, t.width, t.height, t.time_base, t.first_pts), ("clip01", 640, 360, (1, 12800), 1024))
            self.assertEqual(t.frames, d0["frames"])
            self.assertEqual(t.visit, "site_x")
            self.assertEqual(t.source, d0["source"])
            out = os.path.join(d, "out.json.gz")
            tb.save_table(t, out)
            raw = tb.read_table_dict(out)
            self.assertNotIn("source", raw)                                     # a path is never written by default
            self.assertEqual(raw["visit"], "site_x")
            self.assertEqual(raw["schema"], tb.SCHEMA)
            tb.save_table(t, out, include_source=True)
            self.assertEqual(tb.read_table_dict(out)["source"], d0["source"])
            # the loaded legacy table converts back to the same content plus the schema
            again = t.to_dict(include_source=True)
            self.assertEqual({k: v for k, v in again.items() if k != "schema"}, d0)

    def test_unknown_keys_are_kept_and_copy_is_deep(self):
        d0 = dict(legacy_dict(), candidates=[[1024, [[0, 0, 5, 5, 0.02]]]])
        t = BoxTable.from_dict(d0)
        self.assertEqual(t.extra, {"candidates": [[1024, [[0, 0, 5, 5, 0.02]]]]})
        self.assertEqual(t.to_dict()["candidates"], d0["candidates"])
        c = t.copy()
        c.frames[0][1][0][0] = -1
        self.assertEqual(t.frames[0][1][0][0], 10.0)

    def test_schema_checks(self):
        with self.assertRaises(TableError):
            BoxTable.from_dict(dict(legacy_dict(), schema="openveil.boxes/2"))
        with self.assertRaises(TableError):
            BoxTable.from_dict(dict(legacy_dict(), schema="something.else/1"))
        with self.assertRaises(TableError):
            BoxTable.from_dict({k: v for k, v in legacy_dict().items() if k != "frames"})
        with self.assertRaises(TableError):
            BoxTable.from_dict([1, 2, 3])


class TestValidate(unittest.TestCase):
    def test_problems(self):
        good = new_table()
        self.assertEqual(good.problems(), [])
        cases = [
            dict(frames=[[10, [[0, 0, 5, 5, "det"]]], [10, [[0, 0, 5, 5, "det"]]]]),       # pts not increasing
            dict(frames=[[10, [[0, 0, 5, 5, "person"]]]]),                                 # unknown kind
            dict(frames=[[10, [[0, 0, 5, 5, "det", 1.5]]]]),                               # score out of range
            dict(frames=[[10, [[5, 0, 0, 5, "det"]]]]),                                    # inverted box
            dict(frames=[[10, [[0, 0, float("nan"), 5, "det"]]]]),                         # not finite
            dict(frames=[[10.5, [[0, 0, 5, 5, "det"]]]]),                                  # pts not an integer
            dict(frames=[[10, [[0, 0, 5, "det"]]]]),                                       # too short
            dict(width=0), dict(kind="audio"), dict(time_base=(0, 1)), dict(time_base=None, fps=0),
        ]
        for kw in cases:
            t = copy.deepcopy(good)
            for k, v in kw.items():
                setattr(t, k, v)
            self.assertTrue(t.problems(), kw)
            with self.assertRaises(TableError):
                t.validate()


class TestTime(unittest.TestCase):
    def test_seconds_and_pts_at(self):
        t = BoxTable.from_dict(legacy_dict())
        self.assertEqual(t.seconds(1024), 0.0)
        self.assertEqual(t.seconds(1024 + 12800), 1.0)
        self.assertEqual(tb.seconds(legacy_dict(), 1024 + 6400), 0.5)              # the dict form gives the same
        for pts in (1024, 1537, 99999, 123457):
            self.assertEqual(t.pts_at(t.seconds(pts)), pts)
        f = BoxTable(video="frames", width=10, height=10, fps=5.0, kind="frames", time_base=None, first_pts=100)
        self.assertEqual(f.seconds(105), 1.0)
        self.assertEqual(f.pts_at(1.0), 105)
        # exact for large pts: equal to the correctly rounded value of the fraction
        from fractions import Fraction
        big = BoxTable(video="v", width=10, height=10, fps=25.0, time_base=(1, 90000), first_pts=7)
        for pts in (7, 8, 90007, 10 ** 12 + 3):
            self.assertEqual(big.seconds(pts), float(Fraction(pts - 7, 90000)))


class TestOverrides(unittest.TestCase):
    def table(self):
        # 10 fps, time base 1/10: pts = tenths of a second; boxes every second from 0 to 10 s
        frames = [[10 * s, [[100, 100, 150, 200, "det"], [400, 50, 450, 120, "interp"]]] for s in range(11)]
        return BoxTable(video="cam", width=640, height=360, fps=10.0, time_base=(1, 10), first_pts=0, frames=frames)

    def test_delete_window_and_iou(self):
        t = self.table()
        before = t.copy()
        ov = {"cam": {"delete": [[5.0, [101, 99, 151, 201]]]}}
        new, st = tb.apply_overrides(t, ov, delete_window_s=2.0)
        self.assertEqual(t, before)                                                # input untouched
        self.assertEqual(st["deleted_boxes"], 5)                                    # rows at 3, 4, 5, 6, 7 s
        for pts, boxes in new.frames:
            s = pts / 10
            has = any(b[:4] == [100, 100, 150, 200] for b in boxes)
            self.assertEqual(has, not 3 <= s <= 7, s)
        self.assertEqual(new.params["overrides"]["n_delete"], 1)
        # the default window is the temporal hold of the config (2 s in production)
        self.assertEqual(tb.apply_overrides(t, ov)[1]["deleted_boxes"], 5)
        from openveil.config import config_from_dict
        self.assertEqual(tb.apply_overrides(t, ov, config_from_dict({"apply": {"hold_s": 1.0}}))[1]["deleted_boxes"], 3)
        self.assertEqual(tb.apply_overrides(t, ov, config_from_dict({"apply": {"override_delete_window_s": 0.0}}))[1]["deleted_boxes"], 1)
        # a box that does not match (IoU < 0.5) deletes nothing
        _, st = tb.apply_overrides(t, {"cam": {"delete": [[5.0, [100, 100, 130, 140]]]}})
        self.assertEqual(st["deleted_boxes"], 0)
        # a range deletion, window 0
        new, st = tb.apply_overrides(t, {"cam": {"delete": [[2.0, 4.0, [400, 50, 450, 120]]]}}, delete_window_s=0.0)
        self.assertEqual(st["deleted_boxes"], 3)
        # deleting every box of a row removes the row
        new, st = tb.apply_overrides(t, {"delete": [[0.0, [100, 100, 150, 200]], [0.0, [400, 50, 450, 120]]]}, delete_window_s=0.0)
        self.assertEqual(st["removed_rows"], 1)
        self.assertEqual(new.frames[0][0], 10)

    def test_add_manual_boxes(self):
        t = self.table()
        ov = {"cam": {"add": [[2.25, 3.0, [10, 10, 60, 90]]]}}
        new, st = tb.apply_overrides(t, ov, step_s=0.5)
        rows = {pts: boxes for pts, boxes in new.frames}
        # rows at 2.25, 2.75 (new: pts 22 / 28 after rounding to the time base) and 3.0 (existing)
        added = sorted(pts for pts, boxes in rows.items() if any(b[4] == "manual" for b in boxes))
        self.assertEqual(added, [22, 28, 30])
        self.assertEqual(st["added_boxes"], 3)
        self.assertEqual(st["added_rows"], 2)
        self.assertEqual(rows[30][-1], [10.0, 10.0, 60.0, 90.0, "manual"])
        self.assertEqual([p for p, _ in new.frames], sorted(p for p, _ in new.frames))
        new.validate()
        again, st2 = tb.apply_overrides(new, ov, step_s=0.5)                         # applying twice changes nothing
        self.assertEqual(again.frames, new.frames)
        self.assertEqual(st2["added_boxes"], 0)
        # a single time
        new, st = tb.apply_overrides(t, {"cam": {"add": [[12.0, [1, 1, 5, 5]]]}})
        self.assertEqual(new.frames[-1], [120, [[1.0, 1.0, 5.0, 5.0, "manual"]]])
        # other videos are not touched; a dict table comes back as a dict
        d = t.to_dict()
        out, st = tb.apply_overrides(d, {"other": {"add": [[0, 1, [1, 1, 5, 5]]]}})
        self.assertIsInstance(out, dict)
        self.assertEqual(out["frames"], d["frames"])
        self.assertEqual(st["added_boxes"], 0)

    def test_deletions_run_before_additions(self):
        t = self.table()
        box = [100, 100, 150, 200]
        new, st = tb.apply_overrides(t, {"cam": {"delete": [[5.0, box]], "add": [[5.0, 5.0, box]]}})
        rows = dict((p, b) for p, b in new.frames)
        self.assertIn([100.0, 100.0, 150.0, 200.0, "manual"], rows[50])

    def test_overrides_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "overrides.json")
            ov = {"cam": {"add": [[1.0, 2.0, [0, 0, 10, 10]]], "delete": [[3.0, [5, 5, 9, 9]]]}}
            tb.save_overrides(ov, p)
            self.assertEqual(tb.load_overrides(p), ov)
            for bad in ({"cam": {"add": [[2.0, 1.0, [0, 0, 10, 10]]]}},             # t1 before t0
                        {"cam": {"add": [[1.0, 2.0, [0, 0, 0, 10]]]}},              # empty box
                        {"cam": {"delete": [[1.0, [0, 0, 10]]]}},                    # not four numbers
                        {"cam": {"move": []}},                                        # unknown key
                        {"cam": []}):
                with open(p, "w") as f:
                    json.dump(bad, f)
                with self.assertRaises(TableError, msg=str(bad)):
                    tb.load_overrides(p)
        self.assertEqual(tb.overrides_for(None, "x"), {"add": [], "delete": [], "keep": []})
        with self.assertRaises(ValueError):
            tb.apply_overrides(self.table(), {}, step_s=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
