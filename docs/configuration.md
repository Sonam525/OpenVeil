# Configuration reference

How to write a config file is shown in [usage.md](usage.md#a-config-file);
`configs/example_barn.yaml` is the case-study setup (the preset plus the
verifier weights and the camera-clock night policy).

Production values (the `production` preset). Print them all with
`openveil config show --preset production`; `configs/production.yaml` is the
same preset written out with comments. Unknown keys in a config file are
an error.

**Detection schedule** (`schedule`)

| Key | Value | Meaning |
|---|---|---|
| `chunk_s` | 60.0 | video is processed in chunks of this length (s) |
| `base_s` | 1.0 | checkpoint interval in full (day) mode (s) |
| `dense_s` | 0.2 | detector interval after a hit (s) |
| `clear_gaps` | 2 | checkpoint intervals without a hit before dense mode ends |
| `lookahead` | 4 | checkpoints per detector batch |
| `last_checkpoint` | true | last frame of every chunk is a checkpoint |
| `night_base_s` | 5.0 | checkpoint interval in sparse (night) mode; a hit never starts dense mode |
| `max_gap_s` | 3.0 | longest gap bridged by interpolation by day (s); at night `max(max_gap_s, night_gap_factor × night_base_s)` = 11 s |
| `night_gap_factor` | 2.2 | see above |
| `gate_base`, `gate_per_s` | 1.0, 1.5 | interpolation only if the boxes are within `gate_base + gate_per_s × gap` box sizes (or overlap, IoU > 0.05) |
| `day_night.policy` | `always_full` | `always_full`: day schedule everywhere. `clock`: by camera clock |
| `day_night.day_start`, `day_end` | 6, 23 | day hours for the `clock` policy |
| `day_night.margin_s` | 300 | a chunk is day if its start, or its end plus this margin, is in the day |
| `day_night.night_mode` | `sparse` | `sparse`, `skip` (no detection) or `full` |
| `day_night.clock_regex` | `(\d{8})[-_](\d{6})` | how to read the recording start from a file name |

**Proposer** (`proposer`)

| Key | Value | Meaning |
|---|---|---|
| `model_id` | `IDEA-Research/grounding-dino-base` | Grounding DINO model |
| `prompt` | `"worker. person. cow. calf."` | text prompt; the case-study prompt, which the released verifier was trained with (see below) |
| `positive_phrases` | `[worker, person]` | phrases whose score counts |
| `competitor_phrases` | `[cow, calf]` | phrases in the prompt that only absorb animals or objects; never scored |
| `shortest_edge`, `longest_edge` | 800, 1333 | model input size (px) |
| `fp16` | true | half-precision autocast |
| `batch` | 4 | frames per call; halved on out-of-memory |
| `cand_thr` | 0.05 | minimum score of a candidate |
| `merge_iou` | 0.6 | greedy NMS between candidates (non-maximum suppression: of boxes overlapping at IoU ≥ this, keep the best) |

The default prompt names the animals of the case-study barns. Grounding DINO
then has somewhere else to put a calf or a cow, so fewer of them become
candidates, and the released verifier was trained on the candidates of this
prompt with the animals among them as negatives. For other footage, keep the
positive phrases and replace the competitor phrases with the animals or objects
that are mistaken for people there; every phrase must occur in the prompt:

```yaml
preset: production
proposer:
  prompt: "worker. person. horse."
  positive_phrases: [worker, person]
  competitor_phrases: [horse]
```

The verifier then sees candidates from a different prompt than the one it was
trained with, so check it on your own footage (see
[limitations.md](limitations.md) and the steps for
[other footage](usage.md#other-footage-than-calf-barns)).

**Verifier** (`verifier`)

| Key | Value | Meaning |
|---|---|---|
| `mode` | `crop` | `crop` (classifier) or `none` (keep by proposer score) |
| `backbone_id` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | backbone (gated; DINOv3 License) |
| `adapter_dir`, `head_path` | – | the verifier weights ([Verifier weights](usage.md#verifier-weights)) |
| `size`, `pad` | 224, 0.10 | crop size (px) and context per side |
| `resize_width`, `jpeg_quality` | 1280, 90 | frame pre-processing, matching training |
| `threshold` | 0.10 | keep a box if p(person) ≥ this |
| `batch`, `dtype` | 128, `bfloat16` | inference settings |
| `lora_r`, `lora_alpha`, `head_hidden` | 32, 64, [1024, 512] | model shape |

**Per-frame rules** (`postprocess`)

| Key | Value | Meaning |
|---|---|---|
| `nms` | 0.85 | remove near-duplicate kept boxes |
| `container` | true | replace a loose group box by the union of its inner boxes |
| `container_min_inner`, `container_inner_frac`, `container_ratio` | 2, 0.8, 3.0 | ≥ 2 inner boxes, each ≥ 80% inside, group box ≥ 3× their union |

**Masking** (`apply`)

| Key | Value | Meaning |
|---|---|---|
| `min_share` | 0.55 | giant rule: drop a box covering ≥ 55% of the image... |
| `min_inner`, `inner_frac` | 2, 0.8 | ...that contains ≥ 2 smaller boxes, each ≥ 80% inside |
| `keep_kinds` | `[manual]` | box kinds the giant rule never drops |
| `pad` | 0.20 | enlargement per side, as a share of box width/height |
| `hold_s` | 2.0 | a frame gets all boxes within ±2 s |
| `block_divisor`, `block_min` | 36, 16 | mosaic cell = max(16, round(H/36)) px: 20 at 720p, 30 at 1080p |
| `episode_gap_s` | 1.0 | giant-rule firings closer than this form one episode |
| `override_step_s`, `override_match_iou`, `override_delete_window_s` | 0.5, 0.5, = `hold_s` | how review overrides are merged |

**Video encoding** (`video`), **decoding** (`decode`), **images** (`images`), **runs** (`run`)

| Key | Value | Meaning |
|---|---|---|
| `video.h264_codec`, `h264_preset`, `h264_crf` | libx264, medium, 20 | H.264 re-encode |
| `video.hevc_codec`, `hevc_crf` | libx265, 24 | HEVC re-encode |
| `video.other_codec` | mpeg4 | other codecs: full re-encode at the source bit rate |
| `video.bframes`, `repeat_headers`, `scenecut` | 0, true, 0 | encoder flags |
| `video.keyint` | null | null = median GOP length of the source |
| `video.verify` | `fast` | `fast`, `full` or `none` |
| `video.fallback_full_reencode` | true | re-encode in full if smart cut is not eligible or fails its check |
| `decode.first_frame_timeout_s`, `stall_timeout_s` | 300, 120 | kill a stalled decoder (s) |
| `decode.max_bad_pts` | 50 | frames with non-increasing timestamps dropped |
| `images.jpeg_quality` | 95 | quality of re-encoded JPEGs |
| `run.workers`, `max_attempts`, `stall_s` | 1, 3, 2400 | parallel workers, attempts per video, worker watchdog (s, counted once the models are built) |
| `run.device` | `cuda` | torch device of detection (`--device cpu` on a machine without a GPU) |
| `run.restart_wait_s` | 30 | wait before a worker that exited is restarted (s) |
