# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Proposer and verifier logic without the real models: phrase groups, box decoding, crop
geometry (padding, clipping, letterbox), frame preprocessing, head weights in safetensors or
legacy .pt, and the scoring path with a stand-in backbone. Model loading is tested only when
transformers (and peft) are installed."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

from openveil.detect import proposer as P
from openveil.detect import verifier as V

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAVE_PEFT = importlib.util.find_spec("peft") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None

CASE_TOKENS = ["[CLS]", "worker", ".", "person", ".", "cow", ".", "calf", ".", "[SEP]"]


class TestPhrases(unittest.TestCase):
    def test_groups_of_the_case_study_prompt(self):
        groups, names = P.phrase_groups(CASE_TOKENS)
        self.assertEqual(groups, [[1], [3], [5], [7]])
        self.assertEqual(names, ["worker", "person", "cow", "calf"])
        self.assertEqual(P.phrase_indices(names, ["worker", "person"]), [0, 1])
        self.assertEqual(P.phrase_indices(names, ["calf", "cow"], "competitor"), [2, 3])

    def test_multi_token_phrases_and_word_pieces(self):
        toks = ["[CLS]", "farm", "worker", ".", "tract", "##or", ".", "dog", "[SEP]"]
        groups, names = P.phrase_groups(toks)
        self.assertEqual(groups, [[1, 2], [4, 5], [7]])
        self.assertEqual(names, ["farm worker", "tractor", "dog"])

    def test_missing_phrase_is_an_error(self):
        with self.assertRaises(ValueError):
            P.phrase_indices(["person"], ["worker"])
        with self.assertRaises(ValueError):
            P.GroundingDinoProposer(prompt="person.", positive_phrases=["worker"])
        with self.assertRaises(ValueError):
            P.GroundingDinoProposer(prompt="person. calf.", positive_phrases=["person"], competitor_phrases=["cow"])
        with self.assertRaises(ValueError):
            P.GroundingDinoProposer(prompt="person. calf.", positive_phrases=["person"], competitor_phrases=["person"])
        with self.assertRaises(ValueError):
            P.GroundingDinoProposer(prompt="person.", positive_phrases=[])

    def test_any_prompt_names_are_accepted(self):
        # no assertion on particular phrase names: any prompt with matching phrases works
        p = P.GroundingDinoProposer(prompt="pedestrian. cyclist. horse", positive_phrases=["pedestrian", "cyclist"],
                                    competitor_phrases=["horse"])
        self.assertEqual(p.prompt, "pedestrian. cyclist. horse.")
        p.set_tokens(["[CLS]", "pedestrian", ".", "cyclist", ".", "horse", ".", "[SEP]"])
        self.assertEqual(p.positive_idx, [0, 1])
        self.assertEqual(p.competitor_idx, [2])

    def test_decode_boxes_scores_positive_groups_only(self):
        groups = [[1], [3], [5], [7]]
        probs = np.zeros((3, 10), np.float32)
        probs[0, 1] = 0.40           # worker
        probs[0, 5] = 0.90           # cow: a competitor never raises the score
        probs[1, 3] = 0.06           # person, just above cand_thr
        probs[2, 7] = 0.80           # calf only: no candidate
        boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], np.float32)
        out = P.decode_boxes(probs, boxes, 200, 100, groups, [0, 1], 0.05)
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(out[0], [80, 30, 120, 70, 0.40], rtol=1e-6)
        np.testing.assert_allclose(out[1], [40, 20, 60, 30, 0.06], rtol=1e-6)
        det = P.decode_boxes(probs, boxes, 200, 100, groups, [0, 1], 0.05, detailed=True)
        np.testing.assert_allclose(det[0][4:], [0.40, 0.0, 0.90, 0.0], rtol=1e-6)
        self.assertEqual(P.decode_boxes(probs[:0], boxes[:0], 200, 100, groups, [0, 1], 0.05), [])

    def test_make_proposer_from_config(self):
        from openveil.config import ProposerConfig

        pc = ProposerConfig(prompt="worker. person. cow. calf.", positive_phrases=["worker", "person"],
                            competitor_phrases=["cow", "calf"], fp16=False, batch=2, cand_thr=0.07)
        p = P.make_proposer(pc, device="cpu")
        self.assertEqual((p.dtype, p.batch, p.cand_thr, p.device), ("float32", 2, 0.07, "cpu"))
        self.assertEqual(p.model_id, "IDEA-Research/grounding-dino-base")
        self.assertEqual((p.shortest_edge, p.longest_edge), (800, 1333))
        self.assertEqual(P.make_proposer(ProposerConfig()).prompt, "person.")


class TestCropGeometry(unittest.TestCase):
    def test_window_padding_and_clipping(self):
        # 10 % of the box size on every side, rounded outwards
        self.assertEqual(V.crop_window([100, 50, 200, 250], 0.10, 640, 480), (90, 30, 210, 270))
        self.assertEqual(V.crop_window([100.5, 50.5, 200.2, 250.2], 0.10, 640, 480), (90, 30, 211, 271))
        # clipped to the image
        self.assertEqual(V.crop_window([-20, -10, 30, 40], 0.10, 640, 480), (0, 0, 35, 45))
        self.assertEqual(V.crop_window([600, 400, 660, 500], 0.10, 640, 480), (594, 390, 640, 480))
        # too small after clipping: a 4 x 4 window at the box corner, inside the image
        self.assertEqual(V.crop_window([700, 10, 710, 20], 0.10, 640, 480), (636, 10, 640, 14))
        self.assertEqual(V.crop_window([5, 5, 6, 6], 0.0, 640, 480), (5, 5, 9, 9))
        self.assertEqual(V.crop_window([10, 10, 20, 20], 0.0, 640, 480, corner=[3, 4]), (10, 10, 20, 20))
        self.assertEqual(V.crop_window([10, 10, 11, 11], 0.0, 640, 480, corner=[3, 4]), (3, 4, 7, 8))

    def test_letterbox_tall_crop(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[100:300, 200:250] = (255, 0, 0)                 # a 50 x 200 red box
        c = V.make_crop(img, [200, 100, 250, 300], pad=0.0, size=224)
        self.assertEqual(c.shape, (224, 224, 3))
        self.assertEqual(c.dtype, np.uint8)
        # 50 x 200 -> 56 x 224, centred: grey bars of 84 px left and right
        self.assertTrue((c[:, :80] == V.LETTERBOX_GREY).all())
        self.assertTrue((c[:, -80:] == V.LETTERBOX_GREY).all())
        np.testing.assert_array_equal(c[112, 112], [255, 0, 0])
        self.assertTrue((c[:, 88:136, 0] == 255).all())

    def test_letterbox_wide_crop_with_context(self):
        img = np.full((100, 400, 3), 30, np.uint8)
        c = V.make_crop(img, [100, 40, 300, 60], pad=0.10, size=64)
        # window 80..320 x 38..62 = 240 x 24 -> 64 x 6 (rounded), centred vertically
        rows = np.nonzero((c != V.LETTERBOX_GREY).any(axis=(1, 2)))[0]
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0], (64 - 6) // 2)

    def test_jitter_is_reproducible(self):
        rng_img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        a = V.make_crop(rng_img, [40, 30, 80, 100], 0.1, 32, np.random.default_rng([0, 1, 2]), jitter=0.10)
        b = V.make_crop(rng_img, [40, 30, 80, 100], 0.1, 32, np.random.default_rng([0, 1, 2]), jitter=0.10)
        c = V.make_crop(rng_img, [40, 30, 80, 100], 0.1, 32)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_prepare_frame_resize_and_jpeg(self):
        bgr = np.zeros((360, 640, 3), np.uint8)
        bgr[:, :, 2] = 200                                     # red in BGR
        rgb, s = V.prepare_frame(bgr, 1280, 90)
        self.assertEqual(rgb.shape, (720, 1280, 3))
        self.assertAlmostEqual(s, 2.0)
        self.assertLess(abs(int(rgb[300, 600, 0]) - 200), 4)   # RGB order, JPEG error small on a flat frame
        self.assertLess(int(rgb[300, 600, 2]), 6)
        same, s1 = V.prepare_frame(bgr, None, None)
        self.assertEqual(s1, 1.0)
        np.testing.assert_array_equal(same, bgr[:, :, ::-1])

    def test_normalize(self):
        x = V.normalize_crops([np.full((4, 4, 3), 255, np.uint8)])
        self.assertEqual(x.shape, (1, 3, 4, 4))
        np.testing.assert_allclose(x[0, :, 0, 0], (1.0 - V.IMAGENET_MEAN) / V.IMAGENET_STD, rtol=1e-6)


class TestSafetensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_st_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        t = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([1, 2], np.int64)}
        p = os.path.join(self.tmp, "x.safetensors")
        V.save_state_dict(t, p, metadata={"format": "pt"})
        back = V.load_state_dict(p)
        self.assertEqual(sorted(back), ["a.weight", "b"])
        np.testing.assert_array_equal(back["a.weight"], t["a.weight"])
        np.testing.assert_array_equal(back["b"], t["b"])
        with open(p, "rb") as f:                                # standard layout: u64 header length + JSON header
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        self.assertEqual(header["a.weight"]["dtype"], "F32")
        self.assertEqual(header["a.weight"]["shape"], [3, 4])

    def test_reads_bf16(self):
        vals = np.array([1.0, -2.5, 0.15625], np.float32)
        raw = (vals.view(np.uint32) >> 16).astype("<u2").tobytes()
        hb = json.dumps({"w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(raw)]}}).encode()
        p = os.path.join(self.tmp, "bf16.safetensors")
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + raw)
        try:
            back = V.load_state_dict(p)
        except Exception as e:  # pragma: no cover - the safetensors package may refuse BF16 in numpy
            self.skipTest(f"safetensors package cannot read BF16 into numpy: {e}")
        np.testing.assert_allclose(back["w"], vals)


@unittest.skipUnless(HAVE_TORCH, "torch is not installed")
class TestHeadAndScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ov_head_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _head(self, dim=16, hidden=8):
        import torch

        torch.manual_seed(0)
        return V.build_head(dim, hidden).eval()

    def test_head_layout(self):
        h = self._head(1024, 1024)
        shapes = [tuple(p.shape) for p in h.parameters()]
        self.assertEqual(shapes, [(1024, 1024), (1024,), (512, 1024), (512,), (2, 512), (2,)])

    def test_head_from_safetensors_and_legacy_pt(self):
        import torch

        h = self._head()
        st = os.path.join(self.tmp, "head.safetensors")
        pt = os.path.join(self.tmp, "head.pt")
        V.save_head(h, st)
        torch.save(h.state_dict(), pt)
        x = torch.randn(5, 16)
        for path in (st, pt):
            h2 = V.load_head(path)
            torch.testing.assert_close(h2(x), h(x))
        # safetensors is preferred when both exist
        self.assertEqual(V._find_head(self.tmp, None), st)
        os.remove(st)
        self.assertEqual(V._find_head(self.tmp, None), pt)
        with self.assertRaises(FileNotFoundError):
            V._find_head(os.path.join(self.tmp, "nothing"), None)

    def test_load_head_rejects_other_weights(self):
        import torch

        p = os.path.join(self.tmp, "other.pt")
        torch.save({"weight": torch.zeros(2)}, p)
        with self.assertRaises(ValueError):
            V.load_head(p)

    def test_score_frames_with_stand_in_backbone(self):
        import torch
        import torch.nn as nn

        class Out:
            def __init__(self, x):
                self.pooler_output = x

        class Backbone(nn.Module):
            """Pooled feature = mean colour of the crop per channel, repeated to 16 dims."""

            def forward(self, pixel_values):
                m = pixel_values.mean(dim=(2, 3))
                return Out(m.repeat(1, 6)[:, :16])

        v = V.CropVerifier(device="cpu", dtype="float32", batch=2, threshold=0.5, resize_width=None, jpeg_quality=None)
        v.model = Backbone()
        v.head = self._head()
        frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 60, 0.9], [60, 20, 100, 110, 0.4], [0, 0, 160, 120, 0.2]]
        res = v.score_frames([frame, frame], [boxes, boxes[:1]])
        self.assertEqual([len(r) for r in res], [3, 1])
        # same crop, same score, whatever the batch it was scored in
        self.assertAlmostEqual(float(res[0][0]), float(res[1][0]), places=5)
        crops = v.crops_for(frame, boxes)
        direct = v.score_crops(crops)
        np.testing.assert_allclose(res[0], direct, rtol=1e-5)
        self.assertTrue(((direct >= 0) & (direct <= 1)).all())
        self.assertEqual(v.stats["crops"], 4)

    def test_adapter_must_reach_the_model(self):
        import torch.nn as nn

        class Lora(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.lora_A = nn.Linear(4, 2, bias=False)
                self.lora_B = nn.Linear(2, 4, bias=False)
                nn.init.constant_(self.lora_B.weight, b)

        loaded = nn.Sequential(Lora(0.1), Lora(-0.2))
        self.assertEqual(V.check_adapter_loaded(loaded), 2)
        # keys that did not match leave fresh LoRA layers with B = 0 (PEFT only warns): refuse
        with self.assertRaises(ValueError):
            V.check_adapter_loaded(nn.Sequential(Lora(0.1), Lora(0.0)))
        with self.assertRaises(ValueError):
            V.check_adapter_loaded(nn.Sequential(nn.Linear(4, 4)))

    def test_released_package_layout(self):
        # the released weights: adapter, head.safetensors and meta.json side by side in one folder
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            open(os.path.join(self.tmp, name), "wb").close()
        V.save_head(self._head(), os.path.join(self.tmp, "head.safetensors"))
        with open(os.path.join(self.tmp, "meta.json"), "w") as f:
            json.dump(dict(size=224, pad=0.1, threshold=0.1, classes=["not person", "person"]), f)
        v = V.make_verifier({"mode": "crop", "adapter_dir": self.tmp}, device="cpu")
        self.assertEqual(V._find_head(v.adapter_dir, v.head_path), os.path.join(self.tmp, "head.safetensors"))
        self.assertEqual(v.check_meta_file()["size"], 224)

    def test_meta_must_match(self):
        with open(os.path.join(self.tmp, "meta.json"), "w") as f:
            json.dump(dict(size=224, pad=0.10), f)
        self.assertEqual(V.CropVerifier(adapter_dir=self.tmp).check_meta_file()["size"], 224)
        with self.assertRaises(ValueError):
            V.CropVerifier(adapter_dir=self.tmp, pad=0.2).check_meta_file()
        with self.assertRaises(ValueError):
            V.CropVerifier(adapter_dir=self.tmp, size=256).check_meta_file()


class TestMakeVerifier(unittest.TestCase):
    def test_modes(self):
        from openveil.config import VerifierConfig

        none = V.make_verifier(VerifierConfig(mode="none", threshold=0.3))
        self.assertIsInstance(none, V.PassThroughVerifier)
        self.assertEqual(none.threshold, 0.3)
        sc = none.score_frames([None], [[[0, 0, 1, 1, 0.7], [0, 0, 1, 1]]])
        np.testing.assert_allclose(sc[0], [0.7, 1.0])
        with self.assertRaises(ValueError):
            V.make_verifier(VerifierConfig(mode="crop"))            # no weights: refuse instead of guessing
        v = V.make_verifier(VerifierConfig(mode="crop", adapter_dir="weights/final"), device="cpu")
        self.assertIsInstance(v, V.CropVerifier)
        self.assertEqual((v.size, v.pad, v.resize_width, v.jpeg_quality, v.threshold, v.device),
                         (224, 0.10, 1280, 90, 0.10, "cpu"))
        self.assertEqual(v.backbone_id, "facebook/dinov3-vitl16-pretrain-lvd1689m")
        self.assertIsNone(v.model)                                   # nothing is loaded before use


@unittest.skipUnless(HAVE_TRANSFORMERS and HAVE_TORCH, "transformers is not installed")
class TestModelLoading(unittest.TestCase):
    """Only runs where transformers is installed and the model is in the local Hugging Face cache."""

    def test_proposer_loads_and_runs(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            p = P.GroundingDinoProposer(prompt="person.", device="cpu", dtype="float32").load()
        except OSError as e:
            self.skipTest(f"Grounding DINO weights are not cached locally: {e}")
        out = p([np.zeros((240, 320, 3), np.uint8)])
        self.assertEqual(len(out), 1)
        self.assertEqual(p.phrases, ["person"])

    @unittest.skipUnless(HAVE_PEFT, "peft is not installed")
    def test_verifier_needs_weights(self):
        with self.assertRaises((OSError, FileNotFoundError, ValueError)):
            V.CropVerifier(backbone_id="facebook/dinov3-vits16-pretrain-lvd1689m", adapter_dir="no/such/dir",
                           device="cpu").load()


if __name__ == "__main__":
    unittest.main()
