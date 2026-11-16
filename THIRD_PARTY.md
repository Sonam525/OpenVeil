# Third-party components

OpenVeil's own code is licensed under Apache-2.0 (see `LICENSE`). It ships only
Python source, an HTML page and documentation (including three masked example
frames of the case-study dataset, shown with the data owner's approval). It does
**not** include or redistribute any third-party binary, model weight or dataset.
The components below are installed by the user (`pip`) or downloaded by model
id, and each stays under its own licence. The OpenVeil verifier weights are
published separately (see below) and are not under Apache-2.0.

This page is a summary for convenience, not legal advice. Check each licence
yourself before you use or redistribute anything.

## Models used by the pipeline (downloaded separately, not included)

| Model | Licence | How OpenVeil uses it |
|---|---|---|
| Grounding DINO base (`IDEA-Research/grounding-dino-base`) | Apache-2.0 | Box proposer (`detect` extra), loaded with `transformers` by model id. |
| DINOv3 ViT-L/16, LVD-1689M (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | DINOv3 License (custom, not OSI-approved). The Hugging Face repository is gated (manual approval by Meta, which collects name, date of birth, country and affiliation under Meta's Privacy Policy): you must accept Meta's terms first. | Backbone of the crop verifier (`detect` extra), loaded with `transformers` by model id. |
| OpenVeil crop-verifier weights (LoRA adapter + MLP head): [`Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16`](https://huggingface.co/Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16) | DINOv3 License (derivative of DINOv3), not Apache-2.0 | Published separately on the Hugging Face Hub with a verbatim copy of the DINOv3 License: `adapter_config.json`, `adapter_model.safetensors`, `head.safetensors`, `meta.json`, `SHA256SUMS`. No backbone or merged model is distributed. See [`docs/model_card.md`](docs/model_card.md). |

DINOv3 is Copyright (c) Meta Platforms, Inc. and affiliates; the licensor is
Meta Platforms Ireland Limited in the EEA and Switzerland and Meta Platforms,
Inc. elsewhere. There are two published texts of the DINOv3 License: the
`LICENSE.md` that Meta ships in the
[GitHub repository](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
and in the gated Hugging Face repository (last updated 19 August 2025), and the
[web version on ai.meta.com](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
(dated 14 August 2025, the `license_link` of Meta's model card). OpenVeil
follows both. Among other things, the DINOv3 License:

- requires that DINO Materials and derivative works be distributed only under
  its terms, with a copy of the licence (1(b)(i));
- requires acknowledging DINOv3 in any publication of research results that
  uses it (1(b)(ii));
- requires compliance with applicable law, including trade-control, privacy and
  data-protection law (1(b)(iii));
- forbids reverse engineering, decompiling or discovering the underlying
  components of the DINO Materials (1(b)(iv));
- requires that you are not the target of trade controls, and forbids using the
  DINO Materials, or permitting others to use them, for any activities subject
  to ITAR or end uses prohibited by trade controls, including those related to
  military or warfare purposes, nuclear industries or applications, espionage,
  or the development or use of guns or illegal weapons (1(b)(v)).

Meta's published licence (the web version, 1(b)(i)(B)) also requires
prominently displaying "Built with DINOv3"; OpenVeil displays it in its README,
NOTICE and model card, and in the model card of the released weights.

## Python libraries

Version pins are the versions the production run used; newer versions are
expected to work unless noted.

| Library | Version used | Licence | Extra | Use |
|---|---|---|---|---|
| numpy | any recent | BSD-3-Clause (the wheels also bundle 0BSD, MIT, Zlib and CC0-1.0 components) | core | Array maths everywhere. |
| PyAV (`av`) | 18.1.0 (≥ 12 supported) | BSD-3-Clause (the package). See the FFmpeg note below. | `video`, `detect`, `review` | Decoding, smart-cut re-encoding, muxing. |
| scipy | any recent | BSD-3-Clause | `video`, `detect` (optional at run time) | Hungarian matching for gap interpolation (a greedy fallback is used without it). |
| PyTorch (`torch`) | not recorded | BSD-3-Clause style (the wheels also bundle Apache-2.0, BSD-2-Clause, BSL-1.0, ISC, MIT and Zlib components) | `detect`, `train` | Model inference and training. |
| transformers | 4.57.1 (≥ 4.56 needed for DINOv3; < 5 required by the released adapter) | Apache-2.0 | `detect`, `train` | Grounding DINO and DINOv3 model classes. |
| accelerate | 0.33.0 | Apache-2.0 | `detect`, `train` | Device placement helpers used by transformers. |
| peft | 0.13.2 | Apache-2.0 | `detect`, `train` | LoRA adapter loading and training. |
| huggingface_hub | any recent | Apache-2.0 | `detect` | Model downloads (via transformers) and downloading the verifier weights. |
| safetensors | any recent (≥ 0.4) | Apache-2.0 | `detect` | Reading and writing adapter and head weights (a small built-in reader is used without it). |
| Pillow | any recent | MIT-CMU (HPND) | `detect`, `images`, `review` | Image handling, demo scenes. |
| opencv-python-headless | 4.11.0.86 | Apache-2.0 (OpenCV); MIT (packaging); its wheels bundle an LGPL-2.1 FFmpeg | `images` (alternative to Pillow) | JPEG encode/decode, frame extraction for review. |
| scikit-learn | any recent | BSD-3-Clause | `train` | Metrics during classifier training. |
| bitsandbytes | 0.48.0 | MIT | optional, training only | 4-bit loading; not used for the published recipe. |
| PyYAML | any recent | MIT | `detect`, `yaml` | Reading YAML configs (JSON works without it). |

The review server, review page and export use only the Python standard library.
The page loads no external scripts, fonts or CDNs.

## FFmpeg, x264 and x265

| Component | Licence | How it is reached |
|---|---|---|
| FFmpeg | LGPL-2.1-or-later by default; LGPL-3.0-or-later when built with `--enable-version3` (as the PyAV wheels are); GPL-2.0-or-later with `--enable-gpl` | Through PyAV, never vendored |
| x264 (H.264 encoder) | GPL-2.0-or-later (commercial licence available from its authors) | Through PyAV, for smart-cut re-encoding of H.264 |
| x265 (HEVC encoder) | GPL-2.0-or-later (commercial licence available from MulticoreWare) | Through PyAV, for smart-cut re-encoding of HEVC |

- The PyAV wheels on PyPI bundle FFmpeg linked with x264 and x265
  (GPL-2.0-or-later). Treat the bundled binaries as GPL-2.0-or-later when
  redistributing them. The FFmpeg in these wheels is configured with
  `--enable-version3` and without `--enable-gpl` (pyav-ffmpeg patches FFmpeg so
  that x264 and x265 are accepted under version 3), so FFmpeg itself reports
  "LGPL version 3 or later"; the x264 and x265 code it is linked with stays
  GPL-2.0-or-later. x264 is not built for armv7l. PyAV 18.1.0 points to the
  pyav-ffmpeg tag 8.1.2-1, which no longer exists on GitHub (the tags 8.1.1-2
  and 8.1.3-1 have the same configuration). **You** install these wheels;
  OpenVeil does not redistribute them. The GPL is triggered by distributing
  binaries, not by running them.
- If you publish a container image or installer that contains such an FFmpeg
  build, you take on the GPL obligations for it (licence texts and
  corresponding source). An alternative is an LGPL-only FFmpeg build without
  x264/x265; smart cut of H.264/HEVC then falls back to whatever encoders are
  available, or cannot run.
- The GPL does not apply to your output files: masked videos are not GPL.
- **Codec patents** are separate from copyright licences. Encoding H.264 or
  HEVC may need patent licences in some jurisdictions, mainly for commercial
  use. Check this for your own situation.

## Baselines evaluated during development but not included

These were tried while designing OpenVeil (see
[`docs/design_rationale.md`](docs/design_rationale.md)). No code, weights or
outputs from them are part of this repository.

| Baseline | Licence | Why it is not included |
|---|---|---|
| YOLO11 (ultralytics) and fine-tuned YOLO11 weights | AGPL-3.0 (or Ultralytics Enterprise licence) | Lower recall in our tests; licence incompatible with a permissive repository. |
| LocateAnything-3B | NVIDIA License: academic and non-profit research only, no commercial use except by NVIDIA (built on Qwen2.5-3B-Instruct, Qwen Research License, and MoonViT-SO-400M, MIT) | Slower, no better precision; non-commercial terms. |
| SigLIP base (`google/siglip-base-patch16-224`) | Apache-2.0 | Did not separate people from animals well enough. |
| OWLv2 base ensemble (`google/owlv2-base-patch16-ensemble`) | Apache-2.0 | Part of an early pilot that was replaced. |
