# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Verifier training data without transformers: candidates, labels by coverage / IoU,
video-level folds, frame selection, epoch sampling and augmentation, and the out-of-fold
evaluation maths. The training loop itself runs only when transformers and peft are installed."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from openveil.detect.verifier import IMAGENET_MEAN as E_MEAN
from openveil.detect.verifier import IMAGENET_STD as E_STD
from openveil.train import classifier as C
from openveil.train import evaluate as E
from openveil.train import labels as L
from openveil.train import select as S

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAVE_TRAINING = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft"))

PHRASES = ["worker", "person", "cow", "calf"]
POS, COMP = ["worker", "person"], ["cow", "calf"]


def det(box, worker=0.0, person=0.0, cow=0.0, calf=0.0):
    return list(box) + [worker, person, cow, calf]


class TestCandidates(unittest.TestCase):
    def test_threshold_merge_and_animal_score(self):
        boxes = [
            det([10, 10, 50, 90], worker=0.50),
            det([11, 10, 51, 90], person=0.30),               # same place, weaker: merged away
            det([10, 11, 50, 91], calf=0.70),                 # same place, scored as an animal
            det([200, 10, 240, 90], person=0.08),             # below thr 0.10
            det([300, 10, 340, 90], person=0.20, cow=0.10),
        ]
        c = L.phrase_candidates(boxes, PHRASES, POS, COMP, thr=0.10)
        self.assertEqual([r["box"] for r in c], [[10, 10, 50, 90], [300, 10, 340, 90]])
        self.assertEqual([r["id"] for r in c], [1, 2])
        self.assertEqual(c[0]["person"], 0.5)
        self.assertEqual(c[0]["animal"], 0.7)                 # raised from the overlapping animal box
        self.assertEqual(c[1]["animal"], 0.1)
        self.assertEqual(c[0]["scores"]["worker"], 0.5)
        self.assertEqual(len(L.phrase_candidates(boxes, PHRASES, POS, COMP, thr=0.05)), 3)
        # without competitor phrases the animal score is 0
        self.assertEqual(L.phrase_candidates(boxes, PHRASES, POS)[0]["animal"], 0.0)
        with self.assertRaises(ValueError):
            L.phrase_candidates(boxes, PHRASES, ["pedestrian"])

    def test_bucket(self):
        self.assertEqual(L.bucket([dict(person=0.40, animal=0.20)]), "strong")
        self.assertEqual(L.bucket([dict(person=0.40, animal=0.35)]), "medium")
        self.assertEqual(L.bucket([dict(person=0.15, animal=0.10)]), "medium")
        self.assertEqual(L.bucket([dict(person=0.15, animal=0.50)]), "animal_like")
        self.assertEqual(L.bucket([dict(person=0.11, animal=0.0)]), "random")
        self.assertEqual(L.bucket([]), "random")


class TestLabels(unittest.TestCase):
    def test_label_by_coverage_and_iou(self):
        person = [[100, 100, 140, 200]]
        tight = L.label_candidate([98, 98, 142, 202], person)
        self.assertEqual((tight["label"], tight["kind"], tight["covers"]), (1, "person", [0]))
        loose = L.label_candidate([0, 0, 400, 400], person)       # covers the person, IoU tiny
        self.assertEqual((loose["label"], loose["kind"]), (1, "mixed"))
        half = L.label_candidate([100, 100, 140, 149], person)     # 49 % of the person
        self.assertEqual((half["label"], half["kind"]), (0, "none"))
        upper = L.label_candidate([100, 100, 140, 151], person)    # 51 % of the person, IoU 0.51 -> person
        self.assertEqual((upper["label"], upper["kind"]), (1, "person"))
        # tight rule: the person must fill at least 25 % of the box
        self.assertEqual(L.label_candidate([0, 0, 400, 400], person, tight=0.25)["label"], 0)
        self.assertEqual(L.label_candidate([90, 90, 150, 210], person, tight=0.25)["label"], 1)
        self.assertEqual(L.label_candidate([0, 0, 10, 10], [])["label"], 0)

    def _frames(self):
        frames, labels = [], []
        for v in range(6):
            cam = f"cam{v % 2}"
            for k in range(5):
                key = f"v{v}:{k:.2f}"
                person = [20 + 10 * k, 20, 60 + 10 * k, 120]
                boxes = [det([0, 0, 10, 10], person=0.04)]                         # never a candidate at thr 0.05
                if k % 2 == 0:
                    boxes.append(det([person[0] - 2, 18, person[2] + 2, 122], worker=0.6))
                boxes.append(det([200, 50, 260, 110], person=0.2, calf=0.5))        # an animal
                frames.append(dict(key=key, video=f"v{v}", camera=cam, day=f"2024-06-{10 + v}", image=f"img/{key}.jpg",
                                   width=320, height=240, night=(k == 4), boxes=boxes))
                labels.append(dict(key=key, persons=[person] if k < 4 else []))
        frames.append(dict(key="unlabelled", video="v0", camera="cam0", boxes=[]))
        return frames, labels

    def test_build_crop_table_with_video_folds(self):
        frames, labels = self._frames()
        T = L.build_crop_table(frames, labels, PHRASES, POS, COMP, thr=0.05, k=3, split="video")
        rows = T["candidates"]
        self.assertEqual(T["dropped"], 1)
        self.assertEqual([r["id"] for r in rows], list(range(len(rows))))
        self.assertEqual(len(T["frames"]), 30)
        self.assertEqual(len(rows), 6 * (3 + 5))                  # 3 person boxes + 5 animal boxes per video
        self.assertEqual(sum(r["label"] for r in rows), 6 * 2)    # frames 0 and 2 have a covered person (frame 4 is empty)
        animal = [r for r in rows if r["box"] == [200, 50, 260, 110]]
        self.assertTrue(all(r["label"] == 0 and r["kind"] == "none" and r["animal"] == 0.5 for r in animal))
        # folds: no video in two folds, every fold used, frames and people carry their video's fold
        by_video = {}
        for r in rows:
            by_video.setdefault(r["video"], set()).add(r["fold"])
        self.assertTrue(all(len(f) == 1 for f in by_video.values()))
        self.assertEqual(sorted({r["fold"] for r in rows}), [0, 1, 2])
        self.assertEqual([sum(1 for r in rows if r["fold"] == j) for j in range(3)], [16, 16, 16])
        for f in T["frames"]:
            self.assertEqual(f["fold"], next(iter(by_video[f["video"]])))
        # people: 4 per video; the ones in frames 1 and 3 have no candidate
        self.assertEqual(len(T["people"]), 24)
        self.assertEqual(sum(bool(p["cand"]) for p in T["people"]), 12)
        for p in T["people"]:
            for i in p["cand"]:
                self.assertEqual(rows[i]["key"], p["key"])
        self.assertAlmostEqual(sum(T["bucket_share"].values()), 1.0)
        self.assertEqual(T["frames"][0]["n_cands"], 2)

    def test_other_splits(self):
        frames, labels = self._frames()
        T = L.build_crop_table(frames, labels, PHRASES, POS, COMP, thr=0.05, split="camera")
        self.assertEqual(sorted({(r["camera"], r["fold"]) for r in T["candidates"]}), [("cam0", 0), ("cam1", 1)])
        T = L.build_crop_table(frames, labels, PHRASES, POS, COMP, thr=0.05, k=2, split="day")
        days = {}
        for r in T["candidates"]:
            days.setdefault((r["camera"], r["day"]), set()).add(r["fold"])
        self.assertTrue(all(len(f) == 1 for f in days.values()))
        with self.assertRaises(ValueError):
            L.add_folds([dict(video="a", label=1)], 2, split="frame")

    def test_assign_folds_and_always_train_rows(self):
        fold, load = L.assign_folds({"a": [5, 50], "b": [3, 10], "c": [3, 30], "d": [0, 5]}, 2)
        # largest first into the fold with the fewest positives (then rows)
        self.assertEqual(fold, {"a": 0, "c": 1, "b": 1, "d": 0})
        self.assertEqual(load, [[5, 55], [6, 40]])
        rows = [dict(video="x", label=1), dict(video="y", label=0, fold=-1)]
        L.add_folds(rows, 2)
        self.assertEqual(rows[1]["fold"], -1)

    def test_holdout_videos(self):
        vids = [dict(name=f"c{c}_{i}", camera=f"c{c}") for c in range(3) for i in range(15)]
        h = L.holdout_videos(vids, share=0.10, seed=13)
        self.assertEqual(h, L.holdout_videos(vids, share=0.10, seed=13))
        self.assertEqual(h, sorted(h))
        self.assertEqual([sum(n.startswith(f"c{c}_") for n in h) for c in range(3)], [2, 2, 2])
        self.assertEqual(len(L.holdout_videos(vids[:3], share=0.10)), 1)        # at least one per camera

    def test_grey_frames(self):
        grey = np.full((180, 320, 3), 90, np.uint8)
        colour = grey.copy()
        colour[:, :, 0] = 200
        self.assertTrue(L.is_grey(grey))
        self.assertFalse(L.is_grey(colour))


class TestSelectFrames(unittest.TestCase):
    def test_quota_spacing_and_holdout(self):
        mined = []
        for v in range(4):
            for k in range(60):
                t = 10.0 * k
                kind = k % 4
                if kind == 0:
                    boxes = [det([0, 0, 40, 90], worker=0.6)]                  # strong
                elif kind == 1:
                    boxes = [det([0, 0, 40, 90], person=0.2)]                  # medium
                elif kind == 2:
                    boxes = [det([0, 0, 40, 90], person=0.2, calf=0.6)]        # animal-like
                else:
                    boxes = []                                                 # random
                mined.append(dict(video=f"v{v}", camera="cam", t=t, boxes=boxes, image=f"v{v}_{k}.jpg"))
        mined.append(dict(video="v0", camera="cam", t=1.0, boxes=None))
        chosen, rep = S.select_frames(mined, PHRASES, POS, COMP, per_camera=20, spacing_s=30.0, holdout=["v3"],
                                      exclude=lambda r: r["t"] == 0.0)
        again, _ = S.select_frames(mined, PHRASES, POS, COMP, per_camera=20, spacing_s=30.0, holdout=["v3"],
                                   exclude=lambda r: r["t"] == 0.0)
        self.assertEqual([c["key"] for c in chosen], [c["key"] for c in again])      # deterministic
        self.assertEqual(len(chosen), 20)
        self.assertEqual(len({c["key"] for c in chosen}), 20)
        self.assertNotIn("v3", {c["video"] for c in chosen})
        self.assertEqual(rep["excluded"], 3)
        got = {b: sum(c["bucket"] == b for c in chosen) for b in S.ORDER}
        self.assertEqual(got, dict(strong=8, medium=7, animal_like=3, random=2))
        for c in chosen:                                                      # spacing within video and bucket
            same = [o["t"] for o in chosen if o["video"] == c["video"] and o["bucket"] == c["bucket"] and o is not c]
            self.assertTrue(all(abs(o - c["t"]) >= 30.0 for o in same))
            self.assertNotIn("boxes", c)
            self.assertIn("cands", c)
        self.assertEqual(rep["per_camera"]["cam"]["strong"][0], 8)

    def test_shortfall_moves_to_the_next_bucket(self):
        mined = [dict(video="v", camera="c", t=100.0 * k, boxes=[det([0, 0, 40, 90], worker=0.6)]) for k in range(3)]
        mined += [dict(video="v", camera="c", t=100.0 * k + 50, boxes=[]) for k in range(30)]
        chosen, rep = S.select_frames(mined, PHRASES, POS, COMP, per_camera=10, spacing_s=1.0)
        self.assertEqual(len(chosen), 10)
        self.assertEqual(rep["per_camera"]["c"]["strong"], [3, 3])
        self.assertEqual(rep["per_camera"]["c"]["random"][0], 7)


class TestEvaluate(unittest.TestCase):
    def test_auc_and_ap(self):
        y = [0, 0, 1, 1, 0, 1, 0, 1]
        p = [0.1, 0.4, 0.35, 0.8, 0.4, 0.9, 0.05, 0.4]
        if HAVE_SKLEARN:
            from sklearn.metrics import average_precision_score, roc_auc_score

            self.assertAlmostEqual(E.auc(y, p), roc_auc_score(y, p))
            self.assertAlmostEqual(E.average_precision(y, p), average_precision_score(y, p))
        self.assertAlmostEqual(E.auc([0, 1], [0.2, 0.8]), 1.0)
        self.assertAlmostEqual(E.auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertTrue(np.isnan(E.auc([1, 1], [0.2, 0.8])))
        self.assertAlmostEqual(E.average_precision([1, 0, 1], [0.9, 0.8, 0.7]), (1 + 2 / 3) / 2)

    def _table(self):
        # 3 people: two covered by candidates with scores 0.9 and 0.3, one never covered;
        # 2 person-free frames, one with a false candidate of score 0.5
        rows = [dict(id=0, key="f1", label=1), dict(id=1, key="f2", label=1), dict(id=2, key="f3", label=0),
                dict(id=3, key="f1", label=0)]
        people = [dict(key="f1", cand=[0], camera="a"), dict(key="f2", cand=[1], camera="b"), dict(key="f2", cand=[], camera="b")]
        frames = [dict(key="f1", n_people=1, bucket="strong", camera="a"), dict(key="f2", n_people=2, bucket="medium", camera="b"),
                  dict(key="f3", n_people=0, bucket="random", camera="a", night=True),
                  dict(key="f4", n_people=0, bucket="random", camera="b", ir=True)]
        return rows, people, frames

    def test_operating_points(self):
        rows, people, frames = self._table()
        score = {0: 0.9, 1: 0.3, 2: 0.5, 3: 0.05}
        w = E.frame_weights(frames, None)
        ops = E.operating_points(rows, people, frames, score, w, targets=(0.6, 0.9))
        self.assertTrue(ops[0.6]["reachable"])
        self.assertAlmostEqual(ops[0.6]["t"], 0.3)                    # 2 of 3 people need t <= 0.3
        self.assertAlmostEqual(ops[0.6]["recall"], 2 / 3)
        self.assertAlmostEqual(ops[0.6]["fp_all"], 0.5)                # f3 has a kept false box at 0.3
        self.assertAlmostEqual(ops[0.6]["fp_clock_night"], 1.0)
        self.assertAlmostEqual(ops[0.6]["fp_grey"], 0.0)
        self.assertAlmostEqual(ops[0.6]["recall_a"], 1.0)
        self.assertAlmostEqual(ops[0.6]["recall_b"], 0.5)
        self.assertFalse(ops[0.9]["reachable"])
        self.assertAlmostEqual(ops[0.9]["ceiling"], 2 / 3)
        # re-weighting to other bucket shares
        w2 = E.frame_weights(frames, {"strong": 0.5, "medium": 0.5, "random": 0.0})
        self.assertEqual(w2["f1"], 0.5)
        self.assertEqual(w2["f3"], 0.0)

    def test_area_share_and_ordinary_burden(self):
        self.assertAlmostEqual(E.area_share([[0, 0, 640, 720]], 1280, 720), 0.5)
        self.assertAlmostEqual(E.area_share([[0, 0, 640, 720], [0, 0, 640, 720]], 1280, 720), 0.5)
        O = dict(frames=[dict(fid=i, video=f"v{i % 3}", camera="a", night=i % 2 == 1) for i in range(6)],
                 candidates=[dict(id=0, fid=0, box=[0, 0, 128, 72]), dict(id=1, fid=1, box=[0, 0, 1280, 720])])
        b = E.ordinary_burden(O, {"0": 0.8, "1": 0.05}, 0.1, boot=50)
        self.assertEqual(b["all"]["n"], 6)
        self.assertAlmostEqual(b["all"]["with_mosaic"][0], 1 / 6)
        self.assertAlmostEqual(b["all"]["area"][0], 0.01 / 6)
        self.assertLessEqual(b["all"]["with_mosaic"][1], b["all"]["with_mosaic"][0])
        self.assertEqual(b["clock_night"]["with_mosaic"][0], 0.0)

    def test_load_oof_scores(self):
        tmp = tempfile.mkdtemp(prefix="ov_oof_")
        try:
            p = os.path.join(tmp, "oof.json")
            with open(p, "w") as f:
                json.dump(dict(fingerprint={}, folds={"0": dict(scores={"0": 0.2, "2": 0.9}), "1": dict(scores={"1": 0.4})}), f)
            s, st = E.load_oof_scores(p, 4)
            np.testing.assert_allclose(s[:3], [0.2, 0.4, 0.9])
            self.assertTrue(np.isnan(s[3]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTrainingData(unittest.TestCase):
    def test_epoch_rows(self):
        pos, neg = list(range(5)), list(range(100, 200))
        e0 = C.epoch_rows(pos, neg, 10, seed=0, epoch=0)
        self.assertEqual(len(e0), 55)
        self.assertEqual(e0[:5], pos)
        self.assertEqual(e0, C.epoch_rows(pos, neg, 10, seed=0, epoch=0))
        self.assertNotEqual(e0, C.epoch_rows(pos, neg, 10, seed=0, epoch=1))     # fresh negatives every epoch
        self.assertEqual(len(C.epoch_rows(pos, neg[:7], 10, 0, 0)), 12)          # fewer negatives than wanted

    def test_schedule(self):
        cfg = C.TrainConfig(epochs=2, batch=4, neg_per_pos=3)
        self.assertEqual(C.plan_steps(10, 100, cfg), (20, 1))
        self.assertAlmostEqual(C.lr_factor(0, 10, 100), 0.1)
        self.assertAlmostEqual(C.lr_factor(9, 10, 100), 1.0)
        self.assertAlmostEqual(C.lr_factor(10, 10, 100), 1.0)
        self.assertAlmostEqual(C.lr_factor(100, 10, 100), 0.1)
        self.assertAlmostEqual(C.lr_factor(55, 10, 100), 0.55)

    def test_recipe_defaults_match_production(self):
        r = C.TrainConfig().recipe()
        self.assertEqual((r["lora_r"], r["lora_alpha"], r["lora_dropout"], r["lr"], r["wd"], r["epochs"], r["batch"]),
                         (32, 64, 0.05, 1e-4, 0.01, 8, 32))
        self.assertEqual((r["neg_per_pos"], r["size"], r["pad"], r["hidden"], r["smoothing"]), (10.0, 224, 0.10, 1024, 0.1))

    def test_dataset_items_and_augmentation(self):
        img = np.zeros((120, 160, 3), np.uint8)
        img[20:100, 40:80] = (200, 100, 50)
        cfg = C.TrainConfig(size=32, seed=3)
        rows = [dict(id=7, image="a.jpg", box=[40, 20, 80, 100], label=1), dict(id=8, image="a.jpg", box=[0, 0, 30, 30], label=0)]
        train = C.CropDataset(rows, True, cfg, epoch=2, loader=lambda p: img)
        x, y, i = train[0]
        self.assertEqual((x.shape, x.dtype, y, i), ((3, 32, 32), np.float32, 1, 0))
        np.testing.assert_array_equal(x, C.CropDataset(rows, True, cfg, epoch=2, loader=lambda p: img)[0][0])
        self.assertFalse(np.array_equal(x, C.CropDataset(rows, True, cfg, epoch=3, loader=lambda p: img)[0][0]))
        ev = C.CropDataset(rows, False, cfg, loader=lambda p: img)
        xe, ye, _ = ev[1]
        self.assertEqual(ye, 0)
        e1 = C.eval_crop(img, rows[0]["box"], cfg.pad, cfg.size)
        np.testing.assert_allclose(ev[0][0], ((e1 - E_MEAN) / E_STD).transpose(2, 0, 1), rtol=1e-6)
        rng = np.random.default_rng(0)
        for _ in range(20):
            c = C.augment_crop(img, rows[0]["box"], 0.1, 32, rng)
            self.assertTrue(((c >= 0) & (c <= 1)).all())

    def test_row_path_and_fingerprint(self):
        self.assertEqual(C.row_path(dict(image="x/a.jpg"), "root"), os.path.join("root", "x/a.jpg"))
        self.assertEqual(C.row_path(dict(image="a.jpg", path="/p/a.jpg"), "root"), "/p/a.jpg")
        rows = [dict(id=0, label=1, box=[0, 0, 1, 1], fold=0, image="a")]
        f1 = C.table_fingerprint(rows)
        rows[0]["label"] = 0
        self.assertNotEqual(f1, C.table_fingerprint(rows))

    def test_write_meta_matches_verifier_check(self):
        from openveil.detect.verifier import CropVerifier

        tmp = tempfile.mkdtemp(prefix="ov_meta_")
        try:
            C.write_meta(tmp, C.TrainConfig())
            self.assertEqual(CropVerifier(adapter_dir=tmp).check_meta_file()["pad"], 0.10)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestTrainingLoopWithStandInModel(unittest.TestCase):
    """The training, out-of-fold and saving code with a tiny torch model in place of DINOv3 + LoRA."""

    def setUp(self):
        import cv2  # noqa: F401  (images are written with OpenCV)

        self.tmp = tempfile.mkdtemp(prefix="ov_train_")
        rng = np.random.default_rng(0)
        self.rows = []
        for v in range(4):
            for k in range(6):
                img = rng.integers(0, 60, (64, 96, 3), dtype=np.uint8)
                label = int(k % 3 == 0)
                if label:
                    img[10:50, 30:50] = 230                               # a bright "person"
                name = f"v{v}_{k}.png"
                import cv2

                cv2.imwrite(os.path.join(self.tmp, name), img)
                self.rows.append(dict(id=len(self.rows), video=f"v{v}", image=name, box=[30, 10, 50, 50], label=label,
                                      fold=v % 2))
        self.cfg = C.TrainConfig(device="cpu", dtype="float32", workers=0, epochs=2, batch=4, neg_per_pos=2, size=32,
                                 hidden=16, image_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def stand_in(cfg):
        import torch.nn as nn

        from openveil.detect.verifier import build_head, save_state_dict

        class Out:
            def __init__(self, x):
                self.pooler_output = x

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, stride=2)

            def forward(self, pixel_values):
                return Out(self.conv(pixel_values).mean(dim=(2, 3)))

            def save_pretrained(self, d):                     # what a PEFT model writes: adapter only
                save_state_dict(self.state_dict(), os.path.join(d, "adapter_model.safetensors"))
                with open(os.path.join(d, "adapter_config.json"), "w") as f:
                    json.dump(dict(r=cfg.lora_r), f)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.dim = 8
                self.head = build_head(8, cfg.hidden)

            def forward(self, x):
                return self.head(self.backbone(pixel_values=x).pooler_output.float())

        return Model()

    def test_cross_validate_resume_and_final(self):
        from unittest import mock

        from openveil.detect.verifier import load_head

        logs = []
        out = os.path.join(self.tmp, "run")
        with mock.patch.object(C, "build_classifier", self.stand_in):
            st = C.cross_validate(self.rows, out, self.cfg, log=logs.append)
            self.assertEqual(sorted(st["folds"]), ["0", "1"])
            scores, _ = E.load_oof_scores(os.path.join(out, "oof_scores.json"), len(self.rows))
            self.assertFalse(np.isnan(scores).any())
            # a fold scores only its own rows, never a row of a video it trained on
            for f, rec in st["folds"].items():
                self.assertEqual({self.rows[int(i)]["fold"] for i in rec["scores"]}, {int(f)})
            # resumed: nothing is trained again
            n_logs = len(logs)
            C.cross_validate(self.rows, out, self.cfg, log=logs.append)
            self.assertEqual(logs[n_logs:], ["fold 0 already done", "fold 1 already done"])
            # other settings: the old scores are kept aside and the run starts clean
            cfg2 = C.TrainConfig(**{**self.cfg.__dict__, "epochs": 1})
            C.cross_validate(self.rows, out, cfg2, folds=[0], log=logs.append)
            self.assertTrue(any(n.startswith("oof_scores_replaced_") for n in os.listdir(out)))
            # a video in two folds is refused
            bad = [dict(r) for r in self.rows]
            bad[0]["fold"] = 1 - bad[0]["fold"]
            with self.assertRaises(ValueError):
                C.cross_validate(bad, os.path.join(self.tmp, "run_bad"), self.cfg, log=logs.append)
            final = C.train_final(self.rows, out, self.cfg, log=logs.append)
        self.assertEqual(sorted(os.listdir(final)), ["adapter_config.json", "adapter_model.safetensors", "head.safetensors",
                                                     "meta.json"])
        head = load_head(os.path.join(final, "head.safetensors"))
        self.assertEqual(head[0].in_features, 8)
        with open(os.path.join(final, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual((meta["size"], meta["pad"], meta["classes"]), (32, 0.10, ["not person", "person"]))


@unittest.skipUnless(HAVE_TRAINING, "torch, transformers and peft are needed for training")
class TestTrainingLoop(unittest.TestCase):
    def test_build_classifier_needs_a_backbone(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            m = C.build_classifier(C.TrainConfig(backbone="facebook/dinov3-vits16-pretrain-lvd1689m", device="cpu", hidden=64))
        except OSError as e:
            self.skipTest(f"backbone weights are not cached locally: {e}")
        self.assertGreater(C._n_trainable(m), 0)


if __name__ == "__main__":
    unittest.main()
