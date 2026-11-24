# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Tests of openveil.config."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openveil import config as C  # noqa: E402

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class TestConfig(unittest.TestCase):
    def test_production_values(self):
        cfg = C.load_config()
        self.assertEqual(cfg.preset, "production")
        s = cfg.schedule
        self.assertEqual((s.chunk_s, s.base_s, s.dense_s, s.night_base_s, s.clear_gaps, s.lookahead, s.last_checkpoint, s.max_gap_s),
                         (60.0, 1.0, 0.2, 5.0, 2, 4, True, 3.0))
        self.assertEqual((s.night_gap_factor, s.gate_base, s.gate_per_s), (2.2, 1.0, 1.5))
        self.assertEqual((s.day_night.policy, s.day_night.day_start, s.day_night.day_end, s.day_night.night_mode, s.day_night.margin_s),
                         ("always_full", 6.0, 23.0, "sparse", 300.0))
        p = cfg.proposer
        self.assertEqual((p.model_id, p.prompt, p.positive_phrases, p.competitor_phrases), ("IDEA-Research/grounding-dino-base", "worker. person. cow. calf.", ["worker", "person"], ["cow", "calf"]))
        self.assertEqual((p.shortest_edge, p.longest_edge, p.fp16, p.batch, p.cand_thr, p.merge_iou), (800, 1333, True, 4, 0.05, 0.6))
        v = cfg.verifier
        self.assertEqual((v.backbone_id, v.size, v.pad, v.resize_width, v.jpeg_quality, v.threshold),
                         ("facebook/dinov3-vitl16-pretrain-lvd1689m", 224, 0.10, 1280, 90, 0.10))
        self.assertEqual((cfg.postprocess.nms, cfg.postprocess.container, cfg.postprocess.container_ratio), (0.85, True, 3.0))
        a = cfg.apply
        self.assertEqual((a.min_share, a.min_inner, a.inner_frac, a.pad, a.hold_s, a.block_divisor, a.block_min),
                         (0.55, 2, 0.8, 0.20, 2.0, 36.0, 16))
        vc = cfg.video
        self.assertEqual((vc.h264_codec, vc.h264_preset, vc.h264_crf, vc.hevc_codec, vc.hevc_crf, vc.bframes, vc.repeat_headers, vc.scenecut),
                         ("libx264", "medium", 20, "libx265", 24, 0, True, 0))
        self.assertEqual(cfg.images.jpeg_quality, 95)
        self.assertEqual(C.load_config("production"), cfg)
        self.assertIsNot(C.get_preset(), C.get_preset())                  # fresh copies

    def test_partial_json_file_and_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "barn.json")
            with open(p, "w") as f:
                json.dump({"preset": "production",
                           "proposer": {"prompt": "worker. person. cow. calf.", "positive_phrases": ["worker", "person"],
                                        "competitor_phrases": ["cow", "calf"]},
                           "schedule": {"day_night": {"policy": "clock"}},
                           "apply": {"hold_s": 1}}, f)
            cfg = C.load_config(p)
            self.assertEqual(cfg.proposer.competitor_phrases, ["cow", "calf"])
            self.assertEqual(cfg.schedule.day_night.policy, "clock")
            self.assertEqual(cfg.schedule.day_night.day_start, 6.0)              # untouched values keep the preset
            self.assertEqual(cfg.apply.hold_s, 1.0)
            self.assertIsInstance(cfg.apply.hold_s, float)
            out = os.path.join(d, "full.json")
            C.save_config(cfg, out)
            self.assertEqual(C.load_config(out), cfg)
            self.assertEqual(C.config_from_dict(cfg.to_dict()), cfg)

    @unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
    def test_yaml_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.yaml")
            with open(p, "w") as f:
                f.write("apply:\n  min_share: 0.6\nverifier:\n  mode: none\n")
            cfg = C.load_config(p)
            self.assertEqual((cfg.apply.min_share, cfg.verifier.mode), (0.6, "none"))
            out = os.path.join(d, "out.yml")
            C.save_config(cfg, out)
            self.assertEqual(C.load_config(out), cfg)

    def test_errors(self):
        for bad in ({"apply": {"min_shar": 0.5}},                      # typo
                    {"nonsense": 1},
                    {"apply": {"min_inner": 0}},                       # would drop lone big boxes
                    {"apply": {"min_inner": 1.5}},
                    {"apply": {"min_share": "big"}},
                    {"apply": {"keep_kinds": "manual"}},
                    {"postprocess": {"container": 1}},
                    {"schedule": {"day_night": {"policy": "sunrise"}}},
                    {"schedule": {"day_night": "clock"}},
                    {"verifier": {"mode": "maybe"}},
                    {"video": {"verify": "sometimes"}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                C.config_from_dict(bad)
        with self.assertRaises(ValueError):
            C.get_preset("nope")
        with self.assertRaises(FileNotFoundError):
            C.load_config("no_such_preset_or_file.yaml")

    def test_optional_values(self):
        cfg = C.config_from_dict({"verifier": {"resize_width": None, "jpeg_quality": None, "adapter_dir": "models/x"},
                                  "video": {"keyint": 25}})
        self.assertEqual((cfg.verifier.resize_width, cfg.verifier.jpeg_quality, cfg.verifier.adapter_dir, cfg.video.keyint),
                         (None, None, "models/x", 25))


if __name__ == "__main__":
    unittest.main(verbosity=2)
