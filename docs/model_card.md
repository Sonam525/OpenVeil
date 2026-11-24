# Model card: OpenVeil crop verifier (DINOv3 ViT-L/16 + LoRA)

**Built with DINOv3.**

| | |
|---|---|
| Model | Person / not-person crop classifier ("verifier") for OpenVeil |
| Version | v2 (the model used for the case-study release) |
| Developers | Haiyu Yang and Miel Hostens, Department of Animal Science, College of Agriculture and Life Sciences, Cornell University, Ithaca, NY, USA |
| Base model | [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (DINOv3 ViT-L/16, LVD-1689M) |
| Adaptation | LoRA, rank 32, alpha 64, dropout 0.05, on the q/k/v/o and up/down projections (all linear layers); MLP head |
| Weights | [`Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16`](https://huggingface.co/Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16) on the Hugging Face Hub (adapter, head and metadata; no backbone, no merged model). Not in this repository. |
| Licence of the weights | **DINOv3 License**, not Apache-2.0 (see [Licence](#licence)) |
| Code licence | Apache-2.0 (the OpenVeil code that loads and trains the model) |
| Code | <https://github.com/Sonam525/OpenVeil> |

## What it does

Grounding DINO proposes candidate boxes with a low threshold. The verifier
looks at each candidate crop and returns p(person). OpenVeil keeps a box when
p(person) ≥ 0.10. It removes most of the detector's false alarms (animals,
posts, shadows) while keeping the boxes on people.

## Architecture and input

1. Resize the full frame to 1280 px wide and round-trip it through JPEG at
   quality 90 (this matches how the training crops were made).
2. Crop the candidate box plus 10% context on each side.
3. Letterbox to 224 × 224 on grey (114); normalise with the ImageNet mean and
   standard deviation.
4. DINOv3 ViT-L/16 with the LoRA adapter (merged into the backbone in memory at
   load time for speed) → pooled 1024-d embedding.
5. MLP head 1024 → 1024 → 512 → 2, softmax → p(person).

## Files

Published as
[`Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16`](https://huggingface.co/Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16):

| File | Content | Size |
|---|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | LoRA adapter (PEFT format) | 56.7 MB |
| `head.safetensors` | MLP head (6 float32 tensors) | 6.3 MB |
| `meta.json` | classes, image size (224), context pad (0.10), threshold (0.10), training recipe and history, preprocessing; OpenVeil checks size and pad at load | 1.9 KB |
| `SHA256SUMS` | checksums of the four files above (`sha256sum -c SHA256SUMS`) | – |
| `LICENSE` | verbatim copy of the DINOv3 License (the `LICENSE.md` Meta ships with DINOv3, last updated 19 August 2025) | – |
| `NOTICE` | attribution and use restrictions | – |
| `README.md` | the Hub model card | – |

Only the adapter and head are distributed, never a merged model: a merged model
would contain the full DINOv3 weights. You download the base model from Meta's
gated repository yourself, which requires accepting the DINOv3 License.

## How to use it with OpenVeil

```python
from huggingface_hub import snapshot_download

snapshot_download("Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16",
                  local_dir="weights/OpenVeil-Verifier-DINOv3-ViT-L16")
```

Then set the verifier in your config (section `verifier`); the head is found in
the folder by name, so `head_path` is optional:

```yaml
verifier:
  mode: crop
  adapter_dir: weights/OpenVeil-Verifier-DINOv3-ViT-L16
  head_path: weights/OpenVeil-Verifier-DINOv3-ViT-L16/head.safetensors
```

Use `transformers` ≥ 4.56 and < 5 (tested: 4.57.1 with `peft` 0.13.2).
`transformers` 5 renames the DINOv3 modules, so the adapter keys no longer
match; PEFT only warns about that, and OpenVeil refuses to load such an adapter
instead of running without it. Without the weights, set the verifier `mode` to
`none` (candidates are kept by their proposer score alone; expect many more
false mosaics), or train your own verifier with the `openveil.train` package.

## Training data

- **Source**: frames from fixed cameras in calf barns of the case-study dataset
  ([case_study.md](case_study.md)), whose masked version is the released
  dataset (version 1.2, DOI
  [10.57967/hf/11689](https://doi.org/10.57967/hf/11689); data descriptor
  [arXiv:2611.21706](https://arxiv.org/abs/2611.21706)): about 3,700 frames
  from 1,158 videos of 2 farms (Farm A and Farm B).
  The crops were cut from the original, unmasked recordings, because the model
  must see the people it learns to find. **The training crops and frames are
  not distributed.**
- **Candidates**: Grounding DINO with the prompt `worker. person. cow. calf.`,
  every box with a worker/person score ≥ 0.05, greedy NMS at IoU 0.6. This is
  the same candidate rule as at inference time, and the prompt is OpenVeil's
  default (`proposer.prompt`). The animal candidates of this prompt are among
  the negatives; with a different prompt the verifier sees different
  candidates, so check it on your footage.
- **Table**: 50,756 candidate crops, of which 1,444 are positives (people);
  negatives are calves, cows, objects and background.
- **Labels**: person boxes pre-labelled by large-language-model (LLM) reviewers
  (two blind reviewers plus an arbiter), then checked and corrected by a
  human, partly in the OpenVeil review tool. The candidates of the 731
  human-boxed HUM731 frames were added by cross-fitting (see below). Positives are crops that tightly match a
  labelled person.
- **People in the data**: barn staff and visitors, mostly in work clothes and
  boots, often partly hidden by pen walls. Colour footage by day, infrared at
  night.

## Training procedure

| Setting | Value |
|---|---|
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Epochs | 8 |
| Batch size | 32 |
| Negatives per positive (sampling) | 10 |
| Label smoothing | 0.1 |
| Box jitter (augmentation) | 0.10 |
| Image size / context pad | 224 / 0.10 |
| Head hidden size | 1024 |
| Training dtype / quantisation | bfloat16 / none |
| Seed | 0 |
| Training time | 35 min on one NVIDIA A10 (8 epochs, about 65 crops/s in training) |

Evaluation used **cross-fitting** over the HUM731 videos: they were split into
two halves, and each half was scored by a model trained on all training-label
crops plus the HUM731 crops of the other half. The released model was then
trained on all crops, so it has no held-out estimate.

## Evaluation

| Evaluation | Result |
|---|---|
| Cross-fitted ROC AUC (person vs not person, all candidates) | **0.994** |
| HUM731: 731 human-checked everyday frames (172 people in 52 frames), scored by cross-fitted models (see the caveats below), at threshold 0.10 with the duplicate and container rules | **172 / 172** people with ≥ 90% of their area under a mosaic |
| HUM731 weighted to the 2,000 sampled everyday frames, threshold 0.10 | **3.3%** of frames with a false box, **0.45%** with a big one (≥ 15% of the image) |
| HUM731 at 0.127, the highest threshold still covering all 172 people (chosen on HUM731), with the container rule | false-box frames 2.25%; big false boxes 0.35%; false area 0.32% |
| VAL261: 261 hard validation frames (145 people), full pipeline with the production schedule (replay), scored by the released model | recall 0.862 (loose matching: a person counts as found if any mosaic box matches a box drawn on any part of them); false mosaic area 0.22%. Not a held-out estimate: the released model's training data include other minutes of most of the same source videos (no overlapping frames) |
| Speed, one A10, bf16, LoRA merged | 287 crops/s on the GPU alone; 247 crops/s end to end, including JPEG decoding and crop preparation |

How these sets were built, and how the verifier compares with the alternatives
we tried, is described in [design_rationale.md](design_rationale.md).

**Caveats of the evaluation**

- HUM731 is not a held-out test. Its boxes were drawn while checking the
  pipeline's proposals, so it cannot contain a person missed by both the
  pipeline and the reviewer; each scoring model had the other half of the
  HUM731 videos in its training data; and the 0.10 threshold and the
  container rule were chosen with HUM731 in view. The results are optimistic.
- HUM731 contains only 52 frames with people, all in colour. None of the 217
  infrared frames checked contained a person, so **recall on infrared footage
  is unknown**.
- The 1,269 unchecked frames of the 2,000 everyday frames were not searched for
  people.
- VAL261 labels come from LLM reviewers with an arbiter, not from a human.
- All data come from two farms with similar barns and cameras.

### Reproducing the production scores with the published files

Before release, the published files were loaded with OpenVeil's `CropVerifier`
(adapter folder = the downloaded folder, CPU) and compared with the scores the
production run stored for the same candidate boxes. Test set: 192 of the 23,736
candidate boxes on the 2,000 sampled everyday frames, 48 from each of four
stored-score bands (≥ 0.5, 0.10-0.5, 0.07-0.10, < 0.07; seed 0), so 96 above and
96 below the 0.10 threshold. The frames were the stored 1280-px JPEG frames
that production scored, so no further resize or JPEG step was applied.

| Setting | max \|Δp\| | mean \|Δp\| | correlation | same side of 0.10 |
|---|---|---|---|---|
| bfloat16 backbone and autocast, adapter not merged (as production scored) | 0.025 | 0.0011 | 0.99997 | 192 / 192 |
| float32, adapter merged | 0.054 | 0.0020 | 0.99988 | 192 / 192 |

The remaining differences come from bfloat16 versus float32 arithmetic and CPU
versus CUDA kernels. They are largest on the steep part of the softmax (the
0.054 case moved from 0.579 to 0.525); items near the threshold differ by
about 0.002 or less. Tested with `transformers` 4.57.1 and `peft` 0.13.2. With
`transformers` 5 the adapter was silently not applied (max |Δp| 0.69, 110 / 192
on the same side), which is why OpenVeil now requires `transformers` < 5 and
refuses an adapter that did not load. Some of these frames were also training
frames: this check shows that the published files reproduce the production
model; it is not a held-out evaluation.

## Intended use

- Verifying person-box candidates from Grounding DINO inside OpenVeil, to
  **mask** people in research video datasets before release.
- As a starting point for fine-tuning a verifier on a new domain.

## Out-of-scope and prohibited uses

- **Not** for identifying, tracking or profiling people, for surveillance or for
  any decision about a person. It answers only "is there a person in this
  crop?".
- **Not** a face detector and not an anonymisation guarantee on its own: always
  combine it with human review.
- Any use that breaks applicable law, including trade-control, privacy and
  data-protection law (DINOv3 License 1(b)(iii)).
- Reverse engineering, decompiling or discovering the underlying components of
  the DINO Materials (1(b)(iv)).
- Use by anyone who is the target of trade controls, and any use, or permitting
  others' use, for activities subject to ITAR or end uses prohibited by trade
  controls, including those related to military or warfare purposes, nuclear
  industries or applications, espionage, or the development or use of guns or
  illegal weapons (1(b)(v)).

## Limitations and risks

- **Domain.** Trained only on calf-barn footage. On other scenes (streets,
  offices, fields, other animals) expect lower recall and more false alarms;
  check it with the review tool and retrain if needed.
- **Prompt.** The verifier was trained on candidates of the prompt
  `worker. person. cow. calf.` (OpenVeil's default). With another prompt, for
  example one naming other animals, it sees different candidates; re-check it.
- **Threshold.** 0.10 was chosen for recall on this domain, with the HUM731
  results in view; re-check it on yours.
- **Partial and small people.** Legs or boots behind walls, arms at the edge and
  distant figures are the hardest cases; they are where misses are most likely.
- **Night.** Untested on infrared footage with people (see above).
- **Animals.** False boxes fall mainly on calves and cows; a mosaic can hide an
  animal's behaviour.
- **Bias.** The people in the training data are a small group of barn workers
  and visitors. Performance across clothing, body types, skin tones or ages
  was not measured.
- **Training subjects' privacy.** The model was trained on crops of real people.
  The risk that a classifier of this kind memorises recognisable details is low;
  the data owners approved the release of the weights. The training crops are
  not released.

## Licence

The weights are a derivative of DINOv3 and are distributed **only under the
DINOv3 License**, with a verbatim copy of that licence (`LICENSE` in the
weights repository). DINOv3 is Copyright (c) Meta Platforms, Inc. and
affiliates. The Apache-2.0 licence of the OpenVeil code does not apply to the
weights. In short, the DINOv3 License requires you to:

- distribute DINOv3 materials and derivatives only under its terms, with a copy
  of the licence;
- acknowledge DINOv3 in any publication of research results; and, under Meta's
  published licence (the
  [web version](https://ai.meta.com/resources/models-and-libraries/dinov3-license/),
  1(b)(i)(B)), prominently display "Built with DINOv3";
- comply with applicable law, including trade-control, privacy and
  data-protection law;
- not reverse-engineer the DINOv3 materials;
- not be a target of trade controls, and not use them, or let others use them,
  for the prohibited purposes listed above.

The licence also contains a disclaimer of warranty, a limitation of liability,
termination if you sue Meta over intellectual property, an indemnity for claims
arising from your distribution, termination for breach (after which you must
delete the materials), California law, and Meta's right to change the terms.
Read the [full licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
before use. This summary is not legal advice. On the Hugging Face Hub the
weights carry `license: other`, `license_name: dinov3-license`,
`license_link: LICENSE`, `base_model: facebook/dinov3-vitl16-pretrain-lvd1689m`
and `library_name: peft`.

The training labels were produced with the help of LLM reviewers and corrected
by a human. Check that the terms of any LLM service you use for labelling allow
training a model on its output.

## Citation

See [`CITATION.cff`](../CITATION.cff) and the
[README](../README.md#citing-openveil). An arXiv preprint is forthcoming. Please also
cite DINOv3:

```bibtex
@misc{simeoni2025dinov3,
  title         = {{DINOv3}},
  author        = {Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year          = {2025},
  eprint        = {2508.10104},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2508.10104}
}
```
