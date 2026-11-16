# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Training and evaluation of the crop verifier on your own footage (optional).

Steps:

1. run the proposer with ``detailed=True`` on a sparse sample of frames and pick frames to
   label with :func:`openveil.train.select.select_frames`;
2. draw a box around every person in those frames (for example with the review tool);
3. build the crop table with :func:`openveil.train.labels.build_crop_table` (candidates
   labelled by how much of a person they cover, folds by video);
4. measure the cascade out-of-fold with :func:`openveil.train.classifier.cross_validate` and
   :mod:`openveil.train.evaluate`, then train the final adapter and head with
   :func:`openveil.train.classifier.train_final`.

Data preparation and evaluation need numpy only; training needs torch, transformers and peft
(the ``train`` extra).
"""
from .classifier import TrainConfig, cross_validate, train_final
from .evaluate import operating_points, ordinary_burden
from .labels import build_crop_table, holdout_videos, label_candidate, phrase_candidates
from .select import select_frames

__all__ = [
    "TrainConfig", "cross_validate", "train_final", "operating_points", "ordinary_burden", "build_crop_table",
    "holdout_videos", "label_candidate", "phrase_candidates", "select_frames",
]
