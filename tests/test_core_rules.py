# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.rules (giant rule, table rules, episodes, detection post-processing).

Most tests use a 100 x 100 image (area 10000) so that the shares can be read off the numbers.
"""
import copy
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil import rules as br  # noqa: E402
from openveil.geometry import area, clip_box, inter  # noqa: E402
from openveil.table import BoxTable, load_table, read_table_dict, save_table  # noqa: E402

W = H = 100


def rule(boxes, **kw):
    return br.giant_rule(boxes, W, H, **kw)


def record(min_share, min_inner=2, inner_frac=0.8, keep_kinds=("manual",)):
    return dict(rule="giant", min_share=min_share, min_inner=min_inner, inner_frac=inner_frac, keep_kinds=list(keep_kinds))


class TestGiantRule(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((br.DEFAULT_MIN_SHARE, br.DEFAULT_MIN_INNER, br.DEFAULT_INNER_FRAC), (0.55, 2, 0.8))

    def test_two_people_apart_only_the_big_box_is_dropped(self):
        big, a, b = [0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"]
        kept, dropped = rule([a, big, b])
        self.assertEqual(kept, [a, b])                                   # the people stay, in their order
        self.assertEqual(dropped, [big])

    def test_lone_big_box_is_kept(self):
        for k in (1, 2, 3):
            for big in ([0, 0, 100, 100, "det"], [0, 0, 100, 41, "det"], [10, 10, 100, 100, "interp"]):
                kept, dropped = rule([big], min_inner=k)
                self.assertEqual((kept, dropped), ([big], []))
        far = [90, 90, 99, 99, "det"]                                    # a small box outside the big one does not make it a container
        kept, dropped = rule([[0, 0, 80, 60, "det"], far])
        self.assertEqual((len(kept), dropped), (2, []))

    def test_one_inner_box_kept_at_k2_dropped_at_k1(self):
        big, a = [0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"]
        self.assertEqual(rule([big, a]), ([big, a], []))
        self.assertEqual(rule([big, a], min_inner=2), ([big, a], []))
        self.assertEqual(rule([big, a], min_inner=1), ([a], [big]))
        self.assertEqual(rule([big, a], min_inner=3), ([big, a], []))
        two = [70, 10, 90, 50, "det"]
        self.assertEqual(rule([big, a, two], min_inner=3), ([big, a, two], []))      # two inner boxes are not enough for k = 3
        self.assertEqual(rule([big, a, two, [40, 10, 55, 50, "det"]], min_inner=3), ([a, two, [40, 10, 55, 50, "det"]], [big]))

    def test_inner_boxes_that_stick_out_do_not_count(self):
        big, a = [0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"]
        c = [70, 10, 90, 90, "det"]                                      # 20 x 50 of 20 x 80 inside = 62.5%
        self.assertEqual(rule([big, a, c]), ([big, a, c], []))
        self.assertEqual(rule([big, a, c], min_inner=1)[1], [big])       # a alone is enough for k = 1
        self.assertEqual(rule([big, c], min_inner=1), ([big, c], []))    # c alone is not
        # the 80% line: exactly 80% inside counts, a little less does not
        self.assertEqual(rule([big, a, [70, 20, 90, 70, "det"]])[1], [big])           # 800 of 1000 = 80.0%
        self.assertEqual(rule([big, a, [70, 20, 90, 70.1, "det"]])[1], [])            # 800 of 1002 = 79.8%
        self.assertEqual(rule([big, a, [70, 20, 90, 70.1, "det"]], inner_frac=0.79)[1], [big])
        # a box of the same area is not "smaller": an identical duplicate is not an inner box
        self.assertEqual(rule([big, a, [0, 0, 100, 60, "det"]]), ([big, a, [0, 0, 100, 60, "det"]], []))
        # a larger box is not an inner box of the big one, but it is a giant itself with `big` and `a` inside: it goes, `big` stays
        larger = [0, 0, 100, 70, "det"]
        self.assertEqual(rule([big, a, larger]), ([big, a], [larger]))

    def test_nested_giants_are_all_dropped(self):
        outer, mid = [0, 0, 100, 90, "det"], [2, 2, 98, 80, "det"]       # both >= 55% of the image
        a, b = [5, 10, 25, 50, "det"], [70, 10, 90, 50, "det"]
        kept, dropped = rule([outer, mid, a, b])
        self.assertEqual(kept, [a, b])
        self.assertEqual(dropped, [outer, mid])
        # each is judged against the original list: the outer box counts the middle one although the middle one is dropped
        kept, dropped = rule([outer, mid, a], min_inner=2)                # outer: mid + a; mid: a only -> only outer is dropped
        self.assertEqual((kept, dropped), ([mid, a], [outer]))
        self.assertEqual(rule([mid, outer, a], min_inner=2), ([mid, a], [outer]))   # the order of the input does not matter
        # a middle box below the threshold is not a giant, but it counts as an inner box of the outer one
        small_mid = [2, 2, 60, 60, "det"]                                 # 3364 -> 33.6%
        self.assertEqual(rule([outer, small_mid, [5, 5, 20, 20, "det"]]), ([small_mid, [5, 5, 20, 20, "det"]], [outer]))

    def test_box_exactly_at_the_threshold(self):
        inner = [[5, 5, 20, 20, "det"], [60, 5, 80, 20, "det"]]
        self.assertEqual(rule([[0, 0, 100, 50, "det"]] + inner, min_share=0.5)[1], [[0, 0, 100, 50, "det"]])       # 5000 >= 0.5 * 10000
        self.assertEqual(rule([[0, 0, 100, 49.9, "det"]] + inner, min_share=0.5)[1], [])                          # 4990 < 5000
        self.assertEqual(rule([[0, 0, 100, 50.1, "det"]] + inner, min_share=0.5)[1], [[0, 0, 100, 50.1, "det"]])
        # a box of exactly a quarter of a 2304 x 1296 image
        quarter = [0, 0, 1152, 648, "det"]
        kids = [[10, 10, 300, 300, "det"], [600, 10, 900, 300, "det"]]
        self.assertEqual(br.giant_rule([quarter] + kids, 2304, 1296, min_share=0.25)[1], [quarter])
        self.assertEqual(br.giant_rule([quarter] + kids, 2304, 1296, min_share=0.26)[1], [])
        self.assertEqual(br.share(quarter, 2304, 1296), 0.25)

    def test_extra_fields_are_preserved(self):
        big = [0, 0, 100, 60, "interp"]
        a, b = [5, 10, 25, 50, "det", 0.9, "x"], [70, 10, 90, 50, "interp"]
        kept, dropped = rule([big, a, b])
        self.assertEqual(kept, [a, b])
        self.assertEqual(kept[0], [5, 10, 25, 50, "det", 0.9, "x"])
        self.assertEqual(dropped, [[0, 0, 100, 60, "interp"]])
        self.assertEqual(rule([[1, 2, 3, 4]]), ([[1, 2, 3, 4]], []))     # boxes without extra fields
        self.assertEqual(rule([(5, 10, 25, 50, "det")])[0], [[5, 10, 25, 50, "det"]])   # tuples come back as lists

    def test_input_is_not_mutated_and_output_is_independent(self):
        boxes = [[0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"], [-5, -5, 3, 3, "det"]]
        before = copy.deepcopy(boxes)
        kept, dropped = rule(boxes)
        self.assertEqual(boxes, before)
        for out in kept + dropped:
            self.assertFalse(any(out is b for b in boxes))
        kept[0][0] = 999
        dropped[0][0] = 999
        self.assertEqual(boxes, before)

    def test_deterministic(self):
        boxes = [[0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"]]
        self.assertEqual(rule(boxes), rule(boxes))

    def test_zero_area_boxes(self):
        big, a = [0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"]
        for z in ([50, 20, 50, 40, "det"], [50, 20, 60, 20, "det"], [30, 30, 30, 30, "det"], [60, 40, 55, 20, "det"]):   # no width, no height, a point, inverted
            self.assertEqual(rule([big, a, z], min_inner=2), ([big, a, z], []))        # a zero-area box is not an inner box
            self.assertEqual(rule([big, a, z], min_inner=1), ([a, z], [big]))
            self.assertEqual(rule([z]), ([z], []))                                      # and it is never a giant
            self.assertEqual(rule([z, [0, 0, 5, 5, "det"]]), ([z, [0, 0, 5, 5, "det"]], []))
        self.assertEqual(rule([big, [50, 20, 50, 40, "det"]], min_inner=1), ([big, [50, 20, 50, 40, "det"]], []))
        self.assertEqual(rule([[50, 20, 50, 40, "det"]], min_share=1e-12), ([[50, 20, 50, 40, "det"]], []))

    def test_empty_frame(self):
        self.assertEqual(rule([]), ([], []))
        self.assertEqual(rule([], min_inner=1, min_share=0.01), ([], []))

    def test_clipping_boxes_partly_outside_the_image(self):
        # raw area 200 x 110 = 22000 (220% of the image), the part inside the image is 100 x 60 = 6000 (60%)
        out = [-50, -50, 150, 60, "det"]
        a, b = [5, 10, 25, 50, "det"], [70, 10, 90, 50, "det"]
        self.assertAlmostEqual(br.share(out, W, H), 0.6)
        self.assertEqual(rule([out, a, b]), ([a, b], [out]))                 # a giant by its visible part; returned with its original coordinates
        self.assertEqual(rule([out, a, b], min_share=0.7), ([out, a, b], []))            # the raw area alone does not make a giant
        big = [0, 0, 100, 60, "det"]
        stick = [-20, 10, 20, 50, "det"]                                      # raw 40 x 40, half outside the image; the visible 20 x 40 lies inside big
        self.assertEqual(rule([big, a, stick])[1], [big])
        self.assertEqual(rule([big, a, stick])[0], [a, stick])
        # a box completely outside the image has no area: it is not an inner box
        self.assertEqual(rule([big, a, [120, 10, 160, 50, "det"]])[1], [])
        self.assertEqual(rule([[120, 10, 260, 160, "det"]], min_share=0.01), ([[120, 10, 260, 160, "det"]], []))
        # the share of a dropped box never exceeds 1
        t = dict(width=W, height=H, params={}, frames=[[1, [[-50, -50, 150, 150, "det"], a, b]]])
        self.assertEqual(br.apply_rules_to_table(t)[1]["dropped_share_max"], 1.0)

    def test_parameter_validation(self):
        big, a = [0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"]
        for kw in (dict(min_inner=0), dict(min_inner=-1), dict(min_inner=1.5), dict(min_share=0), dict(min_share=-0.1),
                   dict(inner_frac=0), dict(inner_frac=1.01)):
            with self.assertRaises(ValueError, msg=str(kw)):
                rule([big, a], **kw)
        with self.assertRaises(ValueError):
            br.giant_rule([big], 0, 100)
        with self.assertRaises(ValueError):
            br.giant_rule([big], 100, -1)

    def test_scan_of_parameters_is_monotone(self):
        """a larger min_share or a larger min_inner never drops more boxes (fuzz)"""
        rnd = random.Random(7)
        for _ in range(300):
            boxes = random_frame(rnd)
            for k in (1, 2, 3):
                prev = None
                for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70):
                    d = [tuple(x) for x in rule(boxes, min_share=t, min_inner=k)[1]]
                    if prev is not None:
                        self.assertTrue(set(d) <= set(prev))
                    prev = d
            for t in (0.30, 0.50):
                prev = None
                for k in (1, 2, 3):
                    d = [tuple(x) for x in rule(boxes, min_share=t, min_inner=k)[1]]
                    if prev is not None:
                        self.assertTrue(set(d) <= set(prev))
                    prev = d

    def test_manual_boxes_are_never_dropped(self):
        big, a, b = [0, 0, 100, 60, "manual"], [5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"]
        self.assertEqual(rule([big, a, b]), ([big, a, b], []))
        self.assertEqual(rule([big, a, b], keep_kinds=()), ([a, b], [big]))         # without the exemption it is a giant
        # a manual box still counts as an inner box of another giant
        giant = [0, 0, 100, 90, "det"]
        self.assertEqual(rule([giant, [5, 10, 25, 50, "manual"], b]), ([[5, 10, 25, 50, "manual"], b], [giant]))


def random_frame(rnd, w=W, h=H, n_max=7):
    """random boxes, some big, some nested in others, a few sticking out of the image; kind 'det' / 'interp'"""
    boxes = []
    for _ in range(rnd.randint(0, n_max)):
        r = rnd.random()
        if r < 0.3 and boxes:                                            # inside (or nearly inside) an earlier box
            p = rnd.choice(boxes)
            pw, ph = p[2] - p[0], p[3] - p[1]
            x1 = p[0] + rnd.uniform(0, 0.7) * pw
            y1 = p[1] + rnd.uniform(0, 0.7) * ph
            x2 = min(x1 + rnd.uniform(0.1, 0.6) * pw * (1.3 if rnd.random() < 0.3 else 1.0), p[2] + (0.2 * pw if rnd.random() < 0.3 else 0))
            y2 = min(y1 + rnd.uniform(0.1, 0.6) * ph, p[3])
        else:
            bw = rnd.uniform(0.02, 1.0) * w * (1 if r < 0.6 else 0.3)
            bh = rnd.uniform(0.02, 1.0) * h * (1 if r < 0.6 else 0.3)
            x1 = rnd.uniform(-0.05 * w, w - bw * 0.9)
            y1 = rnd.uniform(-0.05 * h, h - bh * 0.9)
            x2, y2 = x1 + bw, y1 + bh
        boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), rnd.choice(["det", "interp"])])
    return boxes


def reference_giant_rule(boxes, w, h, min_share, min_inner, inner_frac=0.8):
    """the rule written out literally, on the boxes clipped to the image"""
    def area_(b): return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    def inter_(a, b): return max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    c = [[min(max(b[0], 0), w), min(max(b[1], 0), h), min(max(b[2], 0), w), min(max(b[3], 0), h)] for b in boxes]
    out = []
    for i, B in enumerate(boxes):
        inner = [c[j] for j in range(len(boxes)) if j != i and area_(c[j]) > 0 and area_(c[j]) < area_(c[i]) and inter_(c[i], c[j]) / area_(c[j]) >= inner_frac]
        if area_(c[i]) >= min_share * w * h and len(inner) >= min_inner:
            continue
        out.append(B)
    return out


class TestFuzz(unittest.TestCase):
    def test_against_the_literal_rule_idempotence_and_structure(self):
        rnd = random.Random(2026)
        n_frames_dropped = 0
        for _ in range(3000):
            boxes = random_frame(rnd)
            for t in (0.30, 0.40, 0.60):
                for k in (1, 2, 3):
                    before = copy.deepcopy(boxes)
                    kept, dropped = rule(boxes, min_share=t, min_inner=k)
                    self.assertEqual(boxes, before)
                    self.assertEqual(kept, reference_giant_rule(boxes, W, H, t, k))
                    self.assertEqual(len(kept) + len(dropped), len(boxes))
                    self.assertEqual(sorted(map(tuple, kept + dropped)), sorted(map(tuple, boxes)))     # nothing lost, nothing invented
                    self.assertEqual(rule(kept, min_share=t, min_inner=k), (kept, []))                  # idempotent
                    self.assertEqual(rule(list(reversed(boxes)), min_share=t, min_inner=k)[0], list(reversed(kept)))   # order does not matter
                    if boxes:
                        self.assertTrue(kept)                                                            # a frame is never emptied
                    for d in dropped:                                                                    # only a box with boxes inside it is dropped
                        self.assertGreaterEqual(br.share(d, W, H), t)
                        cd = clip_box(d, W, H)
                        inside = [b for b in boxes if b is not d and 0 < area(clip_box(b, W, H)) < area(cd)
                                  and inter(cd, clip_box(b, W, H)) >= 0.8 * area(clip_box(b, W, H)) - 1e-9]
                        self.assertGreaterEqual(len(inside), k)
                    n_frames_dropped += bool(dropped)
        self.assertGreater(n_frames_dropped, 100)                                                         # the fuzz does exercise the rule


def small_table():
    a, b = [5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"]
    return dict(video="v", kind="video", visit="V", source="s", fps=25.0, width=W, height=H, time_base=[1, 90000], first_pts=0,
                start_local="2024-01-01T00:00:00", params={"base_s": 1.0, "chunk_s": 60.0}, chunks=[[0.0, "full", 100, 4]],
                frames=[[10, [[0, 0, 100, 60, "det"], a, b]],          # two people apart + a big box: the big box goes
                        [20, [[0, 0, 100, 100, "det"]]],               # lone big box: stays
                        [30, [a, b]],                                  # no giant
                        [40, [[0, 0, 100, 70, "interp"], a, b, [40, 60, 60, 69, "interp"]]],     # a larger giant (70%) with 3 inner boxes
                        [50, [[0, 0, 100, 60, "det"], a]],             # one inner box only: stays at k = 2
                        [60, []]])                                     # an empty frame (does not occur in real tables)


class TestTable(unittest.TestCase):
    def test_stats_and_content(self):
        t = small_table()
        before = copy.deepcopy(t)
        new, st = br.apply_rules_to_table(t, min_share=0.4)
        self.assertEqual(t, before)                                                     # input untouched
        self.assertEqual(st, dict(frames_total_with_boxes=5, frames_changed=2, boxes_total=3 + 1 + 2 + 4 + 2, boxes_dropped=2,
                                  frames_emptied=0, dropped_share_max=0.7))
        self.assertEqual([f[0] for f in new["frames"]], [10, 20, 30, 40, 50])           # the empty frame is removed
        self.assertEqual(new["frames"][0][1], [[5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"]])
        self.assertEqual(new["frames"][1][1], [[0, 0, 100, 100, "det"]])
        self.assertEqual(new["frames"][3][1], [[5, 10, 25, 50, "det"], [70, 10, 90, 50, "interp"], [40, 60, 60, 69, "interp"]])
        self.assertEqual(new["frames"][4][1], [[0, 0, 100, 60, "det"], [5, 10, 25, 50, "det"]])
        self.assertEqual(new["params"], {"base_s": 1.0, "chunk_s": 60.0, "box_rules": record(0.4)})
        for key in t:                                                                   # everything else is copied as it is
            if key not in ("frames", "params"):
                self.assertEqual(new[key], t[key])
        self.assertEqual(list(new), list(t))                                            # same key order
        self.assertIsNot(new["chunks"], t["chunks"])
        new["chunks"][0][0] = 5
        self.assertEqual(t, before)

    def test_stats_with_k1_and_nothing_to_do(self):
        _, st = br.apply_rules_to_table(small_table(), min_share=0.4, min_inner=1)
        self.assertEqual((st["frames_changed"], st["boxes_dropped"], st["frames_emptied"]), (3, 3, 0))
        t = small_table()
        t["frames"] = [[1, [[5, 10, 25, 50, "det"]]], [2, [[0, 0, 100, 100, "det"]]]]
        new, st = br.apply_rules_to_table(t)
        self.assertEqual(st, dict(frames_total_with_boxes=2, frames_changed=0, boxes_total=2, boxes_dropped=0, frames_emptied=0,
                                  dropped_share_max=0.0))
        self.assertEqual(new["frames"], t["frames"])
        new, st = br.apply_rules_to_table(dict(t, frames=[]))
        self.assertEqual((new["frames"], st["frames_total_with_boxes"], st["dropped_share_max"]), ([], 0, 0.0))

    def test_threshold_in_the_table_stats(self):
        t = small_table()
        _, st = br.apply_rules_to_table(t, min_share=0.65)
        self.assertEqual((st["frames_changed"], st["boxes_dropped"], st["dropped_share_max"]), (1, 1, 0.7))
        _, st = br.apply_rules_to_table(t, min_share=0.75)
        self.assertEqual((st["frames_changed"], st["boxes_dropped"], st["dropped_share_max"]), (0, 0, 0.0))

    def test_idempotence_and_guard(self):
        t = small_table()
        once, s1 = br.apply_rules_to_table(t, min_share=0.4)
        twice, s2 = br.apply_rules_to_table(once, min_share=0.4)
        self.assertEqual(twice, once)
        self.assertEqual((s2["boxes_dropped"], s2["frames_changed"]), (0, 0))
        with self.assertRaises(ValueError):
            br.apply_rules_to_table(once, min_share=0.3)                                 # on top of other parameters: refused
        again, _ = br.apply_rules_to_table(once, min_share=0.4, min_inner=2)
        self.assertEqual(again, once)
        no_params = {k: v for k, v in t.items() if k != "params"}
        self.assertEqual(br.apply_rules_to_table(no_params, min_share=0.4)[0]["params"], {"box_rules": record(0.4)})

    def test_fuzz_table_idempotence(self):
        rnd = random.Random(5)
        frames = [[i, random_frame(rnd)] for i in range(400)]
        t = dict(small_table(), frames=frames)
        for k in (1, 2, 3):
            once, s = br.apply_rules_to_table(t, min_inner=k)
            twice, s2 = br.apply_rules_to_table(once, min_inner=k)
            self.assertEqual(twice, once)
            self.assertEqual(s2["boxes_dropped"], 0)
            self.assertEqual(s["frames_emptied"], 0)
            self.assertEqual(sum(len(f[1]) for f in once["frames"]), s["boxes_total"] - s["boxes_dropped"])
            self.assertLessEqual(len(once["frames"]), s["frames_total_with_boxes"])

    def test_file_roundtrip(self):
        """replaces the command-line round trip of the production tool: save -> load -> rule -> save -> load"""
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.json.gz"), os.path.join(d, "sub", "out.json.gz")
            save_table(small_table(), src)
            self.assertNotIn("source", read_table_dict(src))                            # a path is not written by default
            loaded = load_table(src)
            new, st = br.apply_rules_to_table(loaded, min_share=0.4)
            self.assertIsInstance(new, BoxTable)
            self.assertEqual(st["boxes_dropped"], 2)
            save_table(new, dst)
            back = load_table(dst)
            expect, _ = br.apply_rules_to_table(small_table(), min_share=0.4)
            self.assertEqual(back.frames, expect["frames"])
            self.assertEqual(back.params["box_rules"], record(0.4))
            # a rule that changes nothing keeps the table and records the parameters
            same, st = br.apply_rules_to_table(loaded, min_share=5)
            self.assertEqual(st["boxes_dropped"], 0)
            self.assertEqual(same.frames, [f for f in small_table()["frames"] if f[1]])
            self.assertEqual(same.params["box_rules"], record(5))
            with self.assertRaises(ValueError):
                br.apply_rules_to_table(loaded, min_inner=0)                             # a rule that would drop lone boxes

    def test_synthetic_table_invariants(self):
        """the invariants that were checked on a real table, on a large synthetic one"""
        rnd = random.Random(11)
        frames = [[1000 + 40 * i, random_frame(rnd, n_max=9)] for i in range(1500)]
        frames = [f for f in frames if f[1]]
        t = dict(small_table(), frames=frames)
        snapshot = copy.deepcopy(t)
        new, st = br.apply_rules_to_table(t, min_share=0.4)
        self.assertEqual(t, snapshot)
        n_in = len(frames)
        self.assertEqual(st["frames_total_with_boxes"], n_in)
        self.assertEqual(st["frames_emptied"], 0)
        self.assertEqual(len(new["frames"]), n_in)
        self.assertEqual(st["boxes_total"], sum(len(f[1]) for f in frames))
        self.assertEqual(sum(len(f[1]) for f in new["frames"]), st["boxes_total"] - st["boxes_dropped"])
        self.assertGreater(st["boxes_dropped"], 0)
        self.assertGreaterEqual(st["dropped_share_max"], 0.4)
        self.assertLessEqual(st["dropped_share_max"], 1.0)
        by_pts = {f[0]: f[1] for f in new["frames"]}
        changed = 0
        for pts, boxes in frames:
            out = by_pts[pts]
            it = iter(boxes)                                                              # the output is a subsequence of the input
            self.assertTrue(all(any(b == c for c in it) for b in out))
            if len(out) != len(boxes):
                changed += 1
                self.assertGreaterEqual(len(boxes), 3)                                     # a dropped box needs >= 2 boxes inside it
        self.assertEqual(changed, st["frames_changed"])
        again, st2 = br.apply_rules_to_table(new, min_share=0.4)
        self.assertEqual((st2["boxes_dropped"], again), (0, new))


class TestEpisodes(unittest.TestCase):
    def table(self):
        giant = [0, 0, 100, 60, "interp"]                 # 60% of a 100 x 100 image
        left, right = [5, 10, 25, 50, "det"], [70, 10, 90, 50, "det"]
        frames = [[100, [giant, left, right]], [101, [giant, left, right]], [102, [left]],             # episode 1: t = 0 and 1 s
                  [110, [giant, left, right]],                                                         # 9 s later: episode 2
                  [111, [[0, 0, 100, 10, "det"], [1, 1, 40, 9, "det"], [50, 1, 99, 9, "det"], [0, 0, 100, 55, "det"]]]]      # a second giant in one frame
        return dict(video="v", width=100, height=100, fps=1.0, time_base=[1, 1], first_pts=100, frames=frames, params={})

    def test_episodes(self):
        eps = br.firing_episodes(self.table(), min_share=0.5, min_inner=2)
        self.assertEqual(len(eps), 2)
        e = eps[0]
        self.assertEqual((e["start_s"], e["end_s"], e["frames"], e["boxes"], e["max_share"], e["n_inner_at_min"], e["kind_at_min"]),
                         (0.0, 1.0, 2, 2, 0.6, 2, "interp"))
        self.assertEqual((eps[1]["start_s"], eps[1]["frames"]), (10.0, 2))
        self.assertTrue(0.45 < e["min_covered_after"] < 0.6, e["min_covered_after"])       # two 20 x 40 people enlarged to 28 x 56 inside a 100 x 60 box: 0.52
        self.assertEqual(br.firing_episodes(self.table(), min_share=0.9, min_inner=2), [])
        self.assertEqual(br.firing_episodes(self.table(), min_share=0.5, min_inner=2, gap_s=20.0)[0]["frames"], 4)
        self.assertEqual(br.firing_episodes(BoxTable.from_dict(self.table()), min_share=0.5), eps)   # a BoxTable works the same

    def test_covered_share(self):
        self.assertEqual(br.covered_share([0, 0, 100, 100], [[0, 0, 100, 100]], 100, 100), 1.0)
        self.assertEqual(br.covered_share([0, 0, 100, 100], [], 100, 100), 0.0)
        half = br.covered_share([0, 0, 100, 100], [[0, 0, 50, 100]], 100, 100)                # the left half, enlarged by 20% of 50 px on the right
        self.assertTrue(0.55 < half < 0.65, half)

    def test_frames_table_seconds(self):
        t = self.table()
        t["time_base"] = None
        t["fps"] = 5.0
        eps = br.firing_episodes(t, min_share=0.5, min_inner=2, gap_s=1.0)
        self.assertEqual((eps[0]["start_s"], eps[0]["end_s"]), (0.0, 0.2))


class TestDetectionRules(unittest.TestCase):
    def test_candidates(self):
        boxes = [[0, 0, 10, 10, 0.04], [0, 0, 10, 10, 0.5], [1, 0, 11, 10, 0.6], [50, 50, 60, 60, 0.05], [20, 20, 30, 30, 0.3]]
        out = br.candidates(boxes, thr=0.05, merge_iou=0.6)
        # 0.6 kept; 0.5 overlaps it at IoU 9/11 = 0.82 -> merged; below thr dropped; best first
        self.assertEqual(out, [[1, 0, 11, 10, 0.6], [20, 20, 30, 30, 0.3], [50, 50, 60, 60, 0.05]])
        # equal scores keep their input order (stable sort)
        tie = [[0, 0, 5, 5, 0.5], [40, 40, 45, 45, 0.5]]
        self.assertEqual(br.candidates(tie), tie)

    def test_dedupe_keeps_loose_overlaps(self):
        tight, loose, dup = [10, 10, 30, 50, 0.8], [5, 5, 35, 55, 0.9], [10, 10, 30, 50.5, 0.7]
        out = br.dedupe([tight, loose, dup], nms=0.85)
        self.assertEqual(out, [loose, tight])                                            # IoU(tight, loose) = 0.53: both stay; dup goes

    def test_container_rule(self):
        a, b = [10, 10, 20, 40, 0.9], [60, 10, 70, 40, 0.8]
        wrapper = [0, 0, 100, 100, 0.95]                                                 # 10000 >= 3 x union (60 x 30 = 1800)
        out = br.container_rule([wrapper, a, b])
        self.assertEqual(out, [a, b, [10, 10, 70, 40, 0.95]])
        tight = [5, 5, 75, 45, 0.95]                                                     # 2800 < 3 x 1800: not loose enough
        self.assertEqual(br.container_rule([tight, a, b]), [tight, a, b])
        self.assertEqual(br.container_rule([wrapper, a]), [wrapper, a])                  # one inner box is not enough
        self.assertEqual(br.postprocess([wrapper, a, b, [10, 10, 20, 40.2, 0.5]]), [a, b, [10, 10, 70, 40, 0.95]])
        self.assertEqual(br.postprocess([wrapper, a, b], container=False), [wrapper, a, b])

    def test_format_stats(self):
        _, st = br.apply_rules_to_table(small_table(), min_share=0.4)
        text = br.format_stats(st)
        self.assertIn("boxes_dropped", text)
        self.assertIn("share of boxes dropped", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
