# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Detection: find every person in a video and write its box table.

Two models in a cascade:

* :class:`~openveil.detect.proposer.GroundingDinoProposer` proposes boxes for a text prompt
  (open vocabulary, tuned to miss as little as possible);
* :class:`~openveil.detect.verifier.CropVerifier` keeps the proposed boxes that show a person
  (verifier mode ``"none"``: :class:`~openveil.detect.verifier.PassThroughVerifier`).

:func:`~openveil.detect.run.detect_video` runs them on the frames chosen by the scan schedule of
one video; :func:`~openveil.detect.run.run_many` runs many videos with worker processes,
resume, a failure ledger and a stall watchdog.

Only numpy is needed to import this package; torch, transformers, peft and PyAV are imported
when a model is loaded or a video is decoded.
"""
from .proposer import GroundingDinoProposer, make_proposer
from .run import Ledger, default_factory, detect_video, run_many
from .verifier import CropVerifier, PassThroughVerifier, make_crop, make_verifier

__all__ = [
    "GroundingDinoProposer", "make_proposer", "CropVerifier", "PassThroughVerifier", "make_verifier", "make_crop",
    "detect_video", "run_many", "default_factory", "Ledger",
]
