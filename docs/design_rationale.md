# Design rationale

OpenVeil was built to release a large calf play-behaviour video dataset (see
[case_study.md](case_study.md)). Its design is the result of about three weeks
of trials, each measured on held-out frames. This page explains what we tried,
what the numbers were, and why the final pipeline is

> **Grounding DINO proposes, a fine-tuned DINOv3 crop classifier verifies,
> a sparse-then-dense schedule decides when to look, simple box rules tidy up,
> a human reviews the hard cases, and a smart cut re-encodes as little as
> possible.**

## Summary

We tried a dozen detectors and pipelines on the case-study footage before
settling on the cascade. A summary (the full table with every number and its
evaluation set is in [section 4](#4-comparison-table)):

| Approach | Recall | False mosaics | Verdict |
|---|---|---|---|
| YOLO11x + OWLv2 + tracking (pilot) | 44 / 53 person events fully masked | 67% of masked area on animals or objects | Rejected |
| Pilot + fixed privacy zones | 53 / 53 events | masks on 32.4% of all frame area | Rejected |
| Grounding DINO alone | VAL261: 0.876 (loose) | 1.68% of the area of person-free frames (same study); in a separate evaluation (recall 0.855 IoU) 59.2% of person-free frames got a mosaic | Kept as proposer |
| LocateAnything-3B | VAL261: 0.772 | 49.3% of person-free frames; 1.99 s per frame | Rejected |
| YOLO11l fine-tuned in-domain | VAL261: 0.634 | 4.9% of person-free frames | Rejected (recall) |
| Grounding DINO + DINOv3 ViT-S verifier | 95% of labelled people | 47% of colour frames keep a false mosaic | Rejected |
| **Grounding DINO + DINOv3 ViT-L LoRA verifier (final)** | **HUM731: 172 / 172; VAL261: 0.862** | **3.3% of everyday frames** | **Adopted** |

In short: open-vocabulary detectors find people in odd poses but cannot tell a
crouching worker from a calf; detectors trained on the target data are precise
but miss people. Using the first as a proposer and a strong crop classifier as
a verifier keeps the recall of one and most of the precision of the other.

## 1. What we were optimising

1. **Recall first.** A missed person is a privacy failure; a false mosaic only
   hides some pixels. Every threshold is set low and every margin generous.
2. **But not at any price.** A mosaic on an animal can hide the very behaviour
   the dataset is about. Early pilots that put mosaics on a third of all frame
   area were not acceptable.
3. **Keep it simple.** One proposer, one verifier, one threshold to tune. No
   hand-drawn privacy zones, no per-camera heuristics. Every extra rule is
   another thing that can fail silently on the next dataset.
4. **Make human review cheap.** Store boxes (not pixels) so that rules can be
   changed after detection, and give a reviewer a fast tool for the frames that
   matter.
5. **Touch as little footage as possible.** Re-encode only the parts of a video
   that get a mosaic and copy everything else bit for bit.
6. **Affordable on one GPU type.** All costs below are in NVIDIA A10 GPU-hours.

## 2. How we measured

| Name | What it is | Used for |
|---|---|---|
| **VAL261** | 261 validation frames from 76 pilot clips with 145 people in 119 frames. Boxes were labelled by two blind large-language-model (LLM) reviewers plus an arbiter. By body part: 48 full body, 42 lower body, 26 partial at the image edge, 26 upper body, 3 arm or hand only. The set is enriched with frames that a stock YOLO detector missed, so it is harder than everyday footage and biased against YOLO. | Person recall of detectors and of the final pipeline |
| **ORD2000** | 2,000 unlabelled "everyday" frames mined from 118 held-out videos, sampled in strata. | Share of ordinary frames that would get a mosaic |
| **HUM731** | 731 of the ORD2000 frames, checked by a human reviewer in the review tool: all 131 frames an earlier classifier would mask plus 600 of the other 1,869, sampled near and below that classifier's threshold. They hold 172 people in 52 frames. Results are weighted back to all 2,000 frames. | Recall and false mosaics of the final pipeline on everyday footage |
| **Training labels** | 3,004 frames labelled by LLM reviewers: 377 frames with people, 502 person boxes; later partly corrected by a human. | Recall base for the crop-classifier trials |

Metrics used below:

- **Recall (IoU) / covers-half / loose**: the share of labelled people found, by
  three matching rules of increasing leniency: a box matches by IoU (the overlap
  of two boxes, 0 to 1, here ≥ 0.5), a kept box covers at least half of the
  person, or **loose** matching: a person counts as found if any pipeline box
  has IoU ≥ 0.5 with any labelled box drawn on any part of that person (so a box
  on a hand counts). Loose recall is the most lenient of the three.
- **Person-free frames with a mosaic**: of the VAL261 frames without a person,
  the share that would get any mosaic.
- **False-box frames**: frames with a mosaic box lying mostly off any person.
  *Big* false boxes cover at least 15% of the image.
- **Area**: the share of all frame area under a mosaic (or a false mosaic).

## 3. What we tried, in order

1. **Pilot 1: YOLO11x + OWLv2 + ByteTrack**, with ±0.6 s padding, 1 s
   interpolation and 20% enlargement. A frame audit found 44 of 53 annotated
   person events fully masked (83%), and unmasked people in 13 episodes. The
   misses came from gaps of 2.2–3.4 s between detections, crouching people
   labelled "cow" by YOLO, and people cut off at the image edge. Meanwhile 134
   of 320 audit thumbnails had masks only on calves or objects, holding 67% of
   the masked area.
2. **Pilot 2: retune plus three fixed walkway privacy zones**, six OWLv2
   queries, longer padding and hold. Privacy passed (53 of 53 events; 0 of 320
   audit thumbnails with an unmasked person) but precision failed: masks
   covered 32.4% of all frame area, 33% even in clips with no person, and the
   zones alone cost 9–14% of the frame each. Rejected as too blunt and too
   complex.
3. **Back to one detector.** We decided to keep one detector and tune only its
   threshold. Stock YOLO11x per frame put a box in 91–93% of the frames of a
   90 s clip with people (confidence 0.05–0.25) at 27.5–29.2 fps including
   encoding, but its recall was not measured in that trial.
4. **Grounding DINO base** with a phrase study. The best rule used the prompt
   `worker. person. cow. calf.`: a box counts when its *worker* score is
   ≥ 0.06 and not below the best animal score. Recall was high, but most
   person-free frames still got a mosaic, mainly on calves. Prompting with
   `person.` alone halved recall. Requiring a margin over the animal scores
   cut false mosaics but also recall. Grounding DINO was kept only as the
   **proposer**.
5. **LocateAnything-3B**, a large grounding model: lower recall than Grounding
   DINO, still boxed calves, and about 10× slower. Rejected.
6. **SigLIP box verifier and track confirmation.** Neither separated calves
   from people well enough with open-vocabulary scores (the SigLIP outputs were
   not kept as a stored evaluation, so we do not report numbers for it). Track
   confirmation cut false mosaics only by giving up recall. Rejected.
7. **In-domain YOLO11l**, fine-tuned on the 3,004 labelled frames. Very fast
   and precise, but its best recall on VAL261 was 0.634. Rejected: far below
   the 95% target.
8. **Hand-made features**: a gradient-boosted model on Grounding DINO scores,
   box geometry and colour. Better, but at 95% recall a quarter of the
   labelled colour frames still had a false positive. Rejected.
9. **Crop-classifier cascade.** Grounding DINO proposes candidates at a low
   threshold; a DINOv3 image classifier with a LoRA adapter looks at each crop
   and keeps or drops it. A small ViT-S backbone was not enough; ViT-L was.
   Lowering the candidate threshold from 0.10 to 0.05 and then retraining on
   tight positive boxes plus human-corrected labels (v2) gave the final model.
10. **Schedule.** Replaying a sparse-then-dense schedule on clips scored densely
    by the final model showed that it keeps almost all of the recall at a
    fraction of the cost (section 5).
11. **Giant rule.** A rule to drop loose boxes around groups was checked frame by
    frame in the review tool and its threshold set to 0.55 (section 6).
12. **Smart cut**, so that only the stretches of video with a mosaic are
    re-encoded.
13. **Review tool**, used throughout from the cascade stage on.

## 4. Comparison table

Numbers are as measured in each trial; the evaluation set is named in each
cell. "–" means not measured. Speeds are on one A10 GPU unless noted.

| Model / configuration | Person recall | False-mask burden | Speed or cost | Verdict |
|---|---|---|---|---|
| YOLO11x (COCO), per frame, conf 0.05–0.25 | not measured | box in 91–93% of the frames of a 90 s clip with people | 27.5–29.2 fps including encoding | Rejected (superseded by pilots) |
| Pilot 1: YOLO11x + OWLv2 + ByteTrack | 44 / 53 annotated person events fully masked (83%) | 134 / 320 audit thumbnails false-positive only, 67% of masked area | 13.26 fps end to end; 2.73 node-hours per 80 clips | Rejected (misses) |
| Pilot 2: + privacy zones and retune | 53 / 53 events; 0 / 320 audit thumbnails with an unmasked person | 162 / 320 thumbnails false-positive only (66% of mask area); masks on 32.4% of all frame area | 16.27 fps per node; 820.7 node-hours projected for the full dataset | Rejected (precision, complexity) |
| Grounding DINO base, `worker` ≥ 0.06, margin 0 | VAL261: 0.876 (loose, phrase study); 0.855 IoU / 0.869 covers-half (detector evaluation) | VAL261: 59.2% of person-free frames, 4.57% area. ORD2000: 52.4% of frames, 4.99% area | 4.84–5.20 fps measured; 2,391 A10-h for every frame of the dataset at an assumed 5.9 frames/s end to end (2,713–2,915 A10-h at the measured speeds) | Kept as proposer only |
| Grounding DINO, margin 0.05 | VAL261: 0.800 | VAL261: 33.8% / 0.77% area. ORD2000: 30.2% / 1.69% | as above | Rejected |
| Grounding DINO, prompt `person.` only (thr 0.45) | VAL261: 0.455 (loose) | VAL261: 0.88% area | as above | Rejected |
| LocateAnything-3B | VAL261: 0.772 IoU / 0.766 covers-half | VAL261: 49.3% / 4.32% area | 1.99 s per frame | Rejected (slow, still boxes calves) |
| Track confirmation on Grounding DINO | VAL261: 0.855 / 0.800 / 0.731 | VAL261: 47.2% / 34.5% / 12.0% of person-free frames | – | Rejected |
| YOLO11l in-domain, 1-class head | VAL261: 0.262 | VAL261: 2.8%. ORD2000: 1.75% of frames | 75 fps (detector only) | Rejected (recall) |
| YOLO11l in-domain, COCO head, 10 layers frozen | VAL261: 0.490 | VAL261: 2.1%. ORD2000: 2.85% | 73.1 fps | Rejected (recall) |
| YOLO11l in-domain, full fine-tune | VAL261: 0.634 IoU / 0.579 covers-half | VAL261: 4.9%. ORD2000: 4.35% of frames, 0.18% area | 75.2 fps | Rejected (recall) |
| Gradient-boosted model on GD scores, geometry, colour | 90% / 95% of labelled people | 12.7% / 25.5% false-positive frames | – | Rejected |
| GD + DINOv3 ViT-S LoRA classifier, camera held out | AUC 0.938; 95% of labelled people | 47% of colour frames keep a false mosaic | – | Rejected |
| GD + DINOv3 ViT-L LoRA r32, candidates ≥ 0.10 | AUC 0.984, AP 0.901; 95% of labelled people | 12.5% false-positive colour frames (GD alone: 82.5%). ORD2000: 8.4% [4.9–12.6] of frames, 0.27% area | – | Superseded |
| GD + DINOv3 ViT-L LoRA, candidates ≥ 0.05 | AUC 0.981; 95% of labelled people | 7.2% false-positive colour frames. ORD2000: 6.6% [3.7–10.2], 0.83% area | – | Superseded by v2 |
| **Final: GD + DINOv3 ViT-L v2, threshold 0.10, production schedule** | **HUM731: 172 / 172 people with ≥ 90% of their area under a mosaic.** VAL261: 0.862 (loose) | **HUM731: 3.3% of frames with a false box, 0.4% with a big one.** Schedule replay: 0.22% false area | 0.085 detector calls per frame in the replay (0.0508 in the real run); about 280 A10-h estimated by replay, assuming 5.9 frames/s for Grounding DINO and 160 crops/s for the classifier; classifier 276.7 crops/s measured | **Adopted** |

Square brackets are 95% confidence intervals where the trial computed them.
"95% of labelled people" means the classifier threshold was set to keep 95% of
the 502 LLM-labelled people, and the false-positive rate is read at that point.

## 5. How the final architecture was reached

**Why a cascade.** The trials split cleanly into two groups. Open-vocabulary
detectors (Grounding DINO, LocateAnything) find people in odd poses and at the
image edge, but they cannot tell a crouching worker from a calf: about half or
more of the person-free VAL261 frames got a mosaic (Grounding DINO 59%,
LocateAnything 49%). Closed-set detectors trained on
our data (YOLO11l) were precise but missed a third or more of the people. Using
Grounding DINO with a low threshold as a **proposer** keeps its recall; a
classifier that only has to answer "is this crop a person?" removes its false
alarms. At 95% recall the ViT-L verifier cut false-positive colour frames from
82.5% (Grounding DINO alone) to 12.5%, and later to 7.2%.

**Why DINOv3 with LoRA.** The verifier must learn a fine distinction from about
1,400 positive crops. A large self-supervised backbone (DINOv3 ViT-L/16)
gives strong features out of the box; LoRA (rank 32) adapts it cheaply without
over-fitting. The smaller ViT-S backbone left a false mosaic in 47% of colour
frames at the same recall.

**Why the v2 model.** The final classifier was trained on tighter positive
boxes and on labels a human had corrected in the review tool, with cross-fitting
so that every score used for evaluation came from a model that had not seen that
video. Its cross-fitted AUC was 0.994. On HUM731, scored by a model trained
without those videos, every one of the 172 people was covered at the production
threshold, and 3.3% of everyday frames had a false box. For comparison, the
previous model needed a threshold of 0.19 and then gave 3.5% false-box frames,
1.35% big false boxes and 1.19% mosaic area; v2 at threshold 0.12 gave 2.9%,
0.75% and 0.97% (2.6%, 0.40% and 0.80% with the container rule).

**Why a threshold of 0.10.** On HUM731 the v2 model still covered all 172
people at a threshold of 0.12. Production uses 0.10 for a safety margin,
because HUM731 holds only 52 frames with people.

**Why the schedule.** We replayed the production schedule (1 s checkpoints,
0.2 s dense mode after a hit, ±2 s hold) on 76 one-minute clips that the final
model had scored on every stored frame (the stored clips hold about every second
frame of the video). The comparison is a schedule that also uses 1 s
checkpoints but runs on every stored frame while it is in dense mode:

| | 1 s checkpoints, every frame in dense mode | Production schedule |
|---|---|---|
| VAL261 recall (loose) | 0.876 | 0.862 |
| Detector calls per frame | 0.229 | 0.085 |
| Estimated cost for the dataset | about 790 A10-h (547 GD + 243 classifier) | about 280 A10-h (204 GD + 75 classifier) |

The costs assume 5.9 frames/s for Grounding DINO end to end and 160 crops/s
for the classifier on one A10 (the speeds the replay script uses; the measured
detector speed was 4.84–5.20 fps, which would raise every cost by 13–22%).
Running Grounding DINO on every video frame would cost 2,391 A10-h at the same
assumed speed. The first column was recomputed by re-running the replay
script; an earlier note gave 749 A10-h for it, which could not be reproduced.

The schedule covered 99.4% of the boxes found on every stored frame; 2.1% of
frames had a box that it did not cover. On the real dataset
the schedule needed only 0.0508 calls per frame (with the `clock` policy,
sparse at night), because most footage has no people; at the same assumed
speed that is roughly 120 Grounding DINO hours. Note that the replay measures the schedule against the model, not
against people, and the night schedule was not tested this way.

**Why per-frame rules and interpolation.** Duplicate boxes (IoU ≥ 0.85) and
loose "container" boxes around a group are common outputs of open-vocabulary
detectors. The two rules are simple and apply per frame. At threshold 0.12 the
container rule lowered big false boxes from 0.75% to 0.40% of everyday frames
on HUM731.
Interpolation bridges the gaps that caused most misses in the first pilot.

## 6. The giant rule

Some frames get a box around a whole group of people. With 20% enlargement
such a box can blank most of the image. The giant rule drops a box that covers
at least `min_share` of the image and contains at least 2 smaller boxes (each
≥ 80% inside it); the inner boxes stay.

The risk is a person covered by the giant box but not by any inner box. For
every episode in which the rule fired, we reviewed in the review tool the frame
where the dropped box was least covered by the remaining boxes (plus the
biggest-box and middle frames for episodes longer than 10 s), and counted a
frame as *unsafe* if the reviewer kept the giant box, added a person, or left
a final box less than half covered by the boxes the rule keeps:

Each row counts only the frames the rule adds over the next stricter
threshold (the check at 0.50 reviewed all 91 selected frames where it fires
there, including the 45 of 0.55):

| `min_share` | Frames added over the stricter threshold | Safe | Notes |
|---|---|---|---|
| **0.55** | 45 | **45** | adopted |
| 0.50 | +46 (91 in total) | 45 of 46 | one frame where the reviewer kept a box covering 52% of the image |
| 0.40 | +139 (230 in total) | 136 of 139 | three unsafe frames, two of them people left with only 19.4% and 24.7% of their box covered |

The check used the detection tables that were finished at the time, about a
quarter of all videos. In the final release the rule fired in 15,402 frames of
20 videos, and 14 of those videos (79% of those frames) were not part of the
check. OpenVeil uses 0.55 everywhere. `openveil episodes` lists where the rule
fires so that you can review those frames on your own data.

## 7. Hold, enlargement and mosaic

- **±2 s hold and 20% enlargement** came from the pilots: the misses of pilot 1
  were gaps of 2–3 s and people at box edges. With the cascade's dense mode, a
  2 s hold covers the time between a person entering view and the next
  checkpoint.
- **A fixed mosaic grid** (cell size max(16, round(H/36)), anchored at the
  image corner) gives the same coarse pixelation at every resolution and does
  not flicker as boxes move.

## 8. Smart cut

A full re-encode of the case-study videos would have changed every pixel of
every frame. With smart cut only 12.0% of the video packets were re-encoded;
everything else is bit-exact. Two bugs found on the way (a missing packet
duration that dropped the last frame of a run, and repeated HEVC timestamps from
one camera model) are fixed in OpenVeil and covered by the automatic check.

## 9. Notes on conflicting numbers

Some results were recorded more than once with different values. The tables
above always give the value that has a stored evaluation output, and name the
evaluation set. In particular:

- **Grounding DINO recall** for the same rule is 0.876 under the phrase study's
  loose matching and 0.855 (IoU) or 0.869 (covers-half) in the detector
  evaluation, all on VAL261. They are different matching rules, not different
  runs. Confidence intervals for these were not stored and are not given.
- **`person.` alone**: 0.455 recall at 0.88% area (stored phrase-study table).
- **In-domain YOLO11l with a frozen backbone, ORD2000**: 2.85% of frames at
  confidence 0.05.
- **Previous classifier at threshold 0.19 on HUM731**: 3.5% / 1.35% / 1.19%
  (re-run of the evaluation script).
- **Giant rule at 0.40**: worst covers 0.194 and 0.247 (re-run of the check).
- **Cost**: about 280 A10-h is the schedule-replay estimate. The actual billed
  GPU hours were not recorded, so no measured cost is given. The 820.7
  node-hours of pilot 2 is a projection for a different pipeline.
- **Dataset totals**: 1,173 videos (two unplayable files were removed from an
  initial 1,175) and 50,794,625 frames summed from the detection tables.
- **Review counts** in [case_study.md](case_study.md) are recomputed from the
  latest decision for every frame in the review log, which supersedes counts
  made earlier during the review.
- **LoRA rank** of the final adapter is 32 (from its stored configuration).
- **Duplicate suppression** after verification is at IoU ≥ 0.85, a mild rule;
  candidate NMS before verification is at IoU 0.6.
