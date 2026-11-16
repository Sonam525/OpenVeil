# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Apply step: from box tables to masked videos and images.

:mod:`openveil.apply.plan` (which boxes to mosaic in which frame) and
:mod:`openveil.apply.mosaic` (the mosaic itself) need only numpy. The video and image writers
(:mod:`openveil.apply.video`, :mod:`openveil.apply.images`, :mod:`openveil.apply.quality`) need
the ``video`` / ``images`` extras and are imported only when used.
"""
from .mosaic import block_size, mosaic_array, mosaic_yuv420
from .plan import MaskPlan

__all__ = ["MaskPlan", "block_size", "mosaic_array", "mosaic_yuv420"]
