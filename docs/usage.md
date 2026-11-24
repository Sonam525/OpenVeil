# Using OpenVeil

This page has everything about running OpenVeil that the
[README](../README.md#quick-start) leaves out: the install options, every
command with its options, still images and tar shards, config files and the
verifier weights. Every setting of the `production` preset is listed in
[configuration.md](configuration.md); the review tool has its own
[tutorial](review_tool.md).

## Contents

- [Install](#install)
- [Try the review tool first (no GPU, no data)](#try-the-review-tool-first-no-gpu-no-data)
- [Mask your own videos](#mask-your-own-videos)
- [Mask still images](#mask-still-images)
- [A config file](#a-config-file)
- [Verifier weights](#verifier-weights)
- [Tests](#tests)

## Install

OpenVeil needs Python ≥ 3.10. The core needs only numpy; the rest comes in
extras.

```bash
git clone https://github.com/Sonam525/OpenVeil.git openveil && cd openveil
pip install -e ".[video,detect]"      # detection and video masking
```

| Extra | Installs | Needed for |
|---|---|---|
| `video` | av, scipy | `apply video`, `verify` (scipy: optimal matching for interpolation; greedy without it) |
| `detect` | torch, transformers, peft, accelerate, safetensors, pillow, av, scipy, pyyaml | `detect` |
| `train` | `detect` + scikit-learn | training the crop verifier |
| `images` | pillow | `apply images` (OpenCV is used instead when it is installed) |
| `review` | av, pillow | `review demo` (Pillow) and `review build --videos` (PyAV) |
| `yaml` | pyyaml | YAML config files (JSON needs nothing) |
| `dev` | `video`, `images`, `review`, `yaml` | the full test suite |

Detection needs the Grounding DINO base model (downloaded automatically) and,
for the crop verifier, the DINOv3 ViT-L/16 base model, which is **gated**:
accept Meta's DINOv3 License on its Hugging Face page and log in
(`huggingface-cli login`) first. You also need the verifier weights (see
[Verifier weights](#verifier-weights)), or your own, or set the verifier mode to
`none`. A CUDA GPU is strongly recommended for detection; review and apply run
on a CPU (`--device cpu` runs detection on a CPU too, slowly).

## Try the review tool first (no GPU, no data)

```bash
pip install -e ".[review]"           # Pillow draws the synthetic frames
openveil review demo                 # synthetic frames, port 8766
```

Open <http://127.0.0.1:8766> and follow the
[review tutorial](review_tool.md).

## Mask your own videos

```bash
# 0. Look at the settings (the production preset is the default)
openveil config show --preset production

# 1. Detect people: one box table per video (<name>.json.gz)
openveil detect videos/*.mp4 --out tables/ --config my_config.yaml --workers 2 --device cuda

# 2. List where the giant rule fires (frames worth reviewing)
openveil episodes tables/ --min-share 0.55

# 3. Review the hard cases in your browser, then export the corrections
openveil review build  --tables tables/ --videos videos/ --out review/ --per-queue 100
openveil review serve  review/ --port 8765          # open http://127.0.0.1:8765
openveil review export review/ --overrides overrides.json

# 4. Mask each video (smart cut) and check the result
openveil apply video videos/cam1.mp4 --table tables/cam1.json.gz --out masked/cam1.mp4 --overrides overrides.json
openveil verify videos/cam1.mp4 masked/cam1.mp4
```

Add `--full-reencode` to `apply video` to re-encode the whole file instead of a
smart cut. Without `--config`, `detect` uses `--preset production`, whose crop
verifier needs the verifier weights (`verifier.adapter_dir`); a missing package or
weights folder is reported before any video is touched. A video that failed
`run.max_attempts` times is skipped on later runs; `detect --retry-failed`
tries it again.

Smart cut (only GOPs with a mosaic re-encoded) needs H.264 or HEVC 8-bit 4:2:0
without B-frames and with the parameter sets in every keyframe (most cameras;
`ffmpeg ... -x264-params repeat-headers=1` for files you make yourself). Other
files are re-encoded in full; `apply video` prints why.

**Try the whole pipeline without any model.** The built-in stand-in detector
boxes bright regions (it finds no people; it is for trying the commands on a
test clip, for example a bright rectangle moving over a dark background):

```bash
openveil detect clip.mp4 --out tables/ --factory openveil.detect.fake:factory --device cpu
```

`--factory MODULE:FUNCTION` takes any function `f(cfg) -> (proposer, verifier)`;
the interface is described in [how_it_works.md](how_it_works.md#14-plugging-in-other-models).

## Mask still images

```bash
# a folder of JPEG/PNG images with proposed boxes per image (apply also takes a tar)
openveil review build  --images frames/ --boxes boxes.json --out review_images/
openveil review serve  review_images/
openveil review export review_images/
openveil apply images frames/ --boxes review_images/reviewed_boxes.json --out masked_frames/
```

`boxes.json` gives the boxes in pixels of the upright image (after any EXIF
orientation), for example

```json
{"cam1/0001.jpg": [[120, 40, 210, 300], [400, 80, 460, 260, "det", 0.42]],
 "cam1/0002.jpg": []}
```

A box is `[x1, y1, x2, y2]`, optionally followed by a kind and a score. A scored
box below 0.10 (`--accept`) is only a candidate: review shows it dashed and
`apply images` does not mask it. Names are matched exactly or, when unique,
ignoring letter case, a leading `./` and leading folders; boxes for a name that
matches no image make `apply images` fail and write nothing (`--allow-missing`
goes on). `reviewed_boxes.json` marks the boxes a reviewer confirmed as
`"manual"`, so the giant rule never drops them.

In a tar, only the images that get a mosaic are re-encoded (JPEG quality 95);
all other members stay byte-identical, with their names and order.

## A config file

A config file lists only what differs from the preset. This is the case-study
setup (see also `configs/example_barn.yaml`); its prompt
`"worker. person. cow. calf."` is already the preset's default, the prompt the
released verifier was trained with:

```yaml
preset: production
verifier:
  adapter_dir: weights/OpenVeil-Verifier-DINOv3-ViT-L16                    # the downloaded weights
  head_path: weights/OpenVeil-Verifier-DINOv3-ViT-L16/head.safetensors
schedule:
  day_night:
    policy: clock          # day 06:00-23:00 by the camera clock, sparse at night
```

The default prompt names calves and cows so that Grounding DINO puts fewer of
them forward as people, and the verifier learned them as negatives. For other
footage, replace them with what is mistaken for people there and re-check the
verifier on your own frames:

```yaml
proposer:
  prompt: "worker. person. horse."
  positive_phrases: [worker, person]
  competitor_phrases: [horse]
```

## Verifier weights

**Built with DINOv3.** The crop verifier used in the case study (a LoRA adapter
and an MLP head on top of DINOv3 ViT-L/16) is published on the Hugging Face Hub
as [`Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16`](https://huggingface.co/Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16).
Its model card lists the training data, the metrics with their evaluation sets,
and the limits.

| File | Content |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | LoRA adapter (PEFT format, rank 32), 56.7 MB |
| `head.safetensors` | MLP head (1024 → 1024 → 512 → 2), 6.3 MB |
| `meta.json` | training recipe and preprocessing; OpenVeil checks `size` and `pad` at load |
| `SHA256SUMS` | checksums of the four files above |
| `LICENSE`, `NOTICE` | the DINOv3 License and the attribution |

**Licence.** The weights are a derivative of DINOv3 and are distributed under
the **DINOv3 License**, not under this repository's Apache-2.0 licence. The
DINOv3 backbone is not included and no merged model is provided: download it
yourself from Meta's gated repository
([`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)),
which means accepting the DINOv3 License. The licence restricts some uses (see
[Licences](../README.md#licences)).

**Download** (needs the `detect` extra, which brings `huggingface_hub`):

```python
from huggingface_hub import snapshot_download

snapshot_download("Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16",
                  local_dir="weights/OpenVeil-Verifier-DINOv3-ViT-L16")
```

```bash
cd weights/OpenVeil-Verifier-DINOv3-ViT-L16 && sha256sum -c SHA256SUMS && cd -
```

Then point the verifier at the folder (the head is found there by name, so
`head_path` is optional):

```yaml
verifier:
  mode: crop
  adapter_dir: weights/OpenVeil-Verifier-DINOv3-ViT-L16
  head_path: weights/OpenVeil-Verifier-DINOv3-ViT-L16/head.safetensors
```

**Tested versions.** The weights were trained and checked with
`transformers` 4.57.1 and `peft` 0.13.2. `transformers` 5 renames the DINOv3
modules, so the adapter's keys no longer match; the `detect` extra therefore
requires `transformers>=4.56,<5`, and OpenVeil refuses to run an adapter whose
weights did not load. If a download crashes on Windows, setting
`HF_HUB_DISABLE_XET=1` (the classic download path of `huggingface_hub`) helped
in our tests.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

Tests use synthetic data only and skip cleanly when an optional dependency is
missing.
