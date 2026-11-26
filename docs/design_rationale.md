# Design rationale

OpenVeil was built to release a large calf play-behaviour video dataset (see
[case_study.md](case_study.md)). Its design is the result of about three weeks
of trials on that footage, followed by a comparison of the final pipeline with
single detectors under one protocol. This page explains what was compared,
what the numbers were, and why the final pipeline is

> **a detector that proposes broadly (Grounding DINO at a low threshold), a
> classifier that judges each proposed crop (DINOv3 with a LoRA adapter), and a
> person who reviews the frames where errors are likely; around them a
> sparse-then-dense scan schedule, simple box rules and a smart cut that
> re-encodes as little as possible.**

All numbers on this page come from the case study, calf barns on two farms.
They are not results for other kinds of footage.

## Summary

OpenVeil does not ask one detector to be perfect. In the case-study footage,
no score threshold on a single detector both kept every person and rejected
every calf that looked like a crouching worker. OpenVeil splits the work: the
detector only has to propose, so its threshold can be low enough to miss
almost no one, and a classifier that sees one crop at a time decides which
proposals show a person.

### The comparison under one protocol

The table compares OpenVeil with five detectors and a face detector on the
**test set**: 731 human-checked frames of everyday footage with 172 people
([section 2](#2-how-we-measured) defines the sets and the measures). Every
method was scored on the same frames and boxes. A person counts as covered when
at least 90% of their box lies under the method's boxes, each enlarged by 20%
per side as in masking. A budget is an upper limit on the share of everyday
frames with a false box; at each budget a detector is shown at the threshold
that covered the most people within the limit.

| Method | Most people covered (false-box frames at that threshold) | Within 3.3% | Within 5% | Within 10% | Within 20% |
|---|---|---|---|---|---|
| RetinaFace, face box | 0 | 0 | 0 | 0 | 0 |
| RetinaFace, body box drawn from the face | 34 (22.6%) | 16 | 21 | 25 | 30 |
| YOLO11x, pre-trained on COCO | 137 (51.0%) | 82 | 93 | 99 | 106 |
| YOLO11l, fine-tuned on the training set | 128 (5.0%) | 126 | 128 | 128 | 128 |
| OWLv2, query "a person" | 142 (52.7%) | 35 | 55 | 81 | 119 |
| LocateAnything-3B (no score, one operating point) | 151 (50.9%) | – | – | – | – |
| Grounding DINO alone | 172 (100%) | 71 | 98 | 123 | 128 |
| **OpenVeil (Grounding DINO with the verifier at 0.10)** | **172 (3.3%)** | **172** | **172** | **172** | **172** |

What the table shows:

- **Grounding DINO alone has the people among its candidates, but no usable
  threshold.** Keeping every candidate (score ≥ 0.05) covers all 172 people and
  puts a false box on every everyday frame. At thresholds of 0.2, 0.3 and 0.4
  the false-box frames fall to 42.3%, 14.8% and 3.7%, and the people covered
  fall to 138, 128 and 88.
- **The fine-tuned YOLO11l makes few false boxes but leaves people uncovered at
  every threshold.** Even at its lowest confidence threshold it covered 128 of
  the 172 people. It mostly misses parts of people, not whole people: at a
  threshold of 0.02 its boxes covered at least half of the box of 163 people.
- **A face detector cannot do this work.** RetinaFace found a face for 66 of
  the 172 people at a score threshold of 0.05 (33 at 0.5), and no face box
  covered a person.
- **With the verifier, false boxes can be removed without giving up coverage.**

Read the table with these limits in mind:

- The thresholds of the single detectors were chosen on the test set, in their
  favour.
- The test set favours methods built on Grounding DINO: its boxes were drawn
  while checking the pipeline's proposals, and 168 of the 172 boxes are
  Grounding DINO candidates. A rule that asks for 90% of such a box is strict
  for a detector whose boxes have another shape.
- The verifier scores on the test set are cross-fitted, and the threshold of
  0.10 was chosen with the test-set results in view
  ([section 2](#2-how-we-measured)).
- YOLO11l was fine-tuned on the LLM labels of the training set before their
  review and saw no test-set video; the cross-fitted verifier was trained on
  the reviewed labels and on half of the test-set videos.

### The development trials that led there

Before this comparison, a dozen detectors and pipelines were tried during
development. The numbers in the next table come from those trials. They were
measured on pilot clips and on the **validation set** (261 hard frames, 145
people), with different matching rules, so the rows are not all comparable
with each other or with the table above. The full table, with every number and
the set it was measured on, is in [section 4](#4-comparison-table).

| Approach (development trial) | Recall | False mosaics | Verdict |
|---|---|---|---|
| YOLO11x + OWLv2 + tracking (pilot) | 44 / 53 person events fully masked | 67% of masked area on animals or objects | Rejected |
| Pilot + fixed privacy zones | 53 / 53 events | masks on 32.4% of all frame area | Rejected |
| Grounding DINO alone (`worker` score ≥ 0.06) | validation set: 0.876 (loose) | 59.2% of person-free frames got a mosaic (4.57% of their area) | Kept as proposer |
| LocateAnything-3B | validation set: 0.772 | 49.3% of person-free frames; 1.99 s per frame | Rejected |
| YOLO11l fine-tuned in-domain | validation set: 0.634 | 4.9% of person-free frames | Rejected (recall) |
| Grounding DINO + DINOv3 ViT-S verifier | 95% of labelled people | 47% of colour frames keep a false mosaic | Rejected |
| **Grounding DINO + DINOv3 ViT-L LoRA verifier (final)** | **test set: 172 / 172 covered; validation set: 0.862 (loose)** | **3.3% of everyday frames** | **Adopted** |

In short: open-vocabulary detectors find people in odd poses but cannot tell a
crouching worker from a calf; detectors trained on the target data are precise
but miss people. Using the first as a proposer and a crop classifier as a
verifier keeps the coverage of one and most of the precision of the other.

## 1. What we were optimising

1. **Coverage first.** A missed person is a privacy failure; a false mosaic
   only hides some pixels. Every threshold is set low and every margin
   generous.
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

Frames a fraction of a second apart show almost the same scene, so training
and test data were separated by whole videos, not by frames. Of the videos of
each recording, 10% (118 in all) were held out: no training frame was taken
from them, and they supplied the test set. Footage within 5 min of a
validation clip was kept out of the training set as well.

| Set | What it is | Used for |
|---|---|---|
| **Training set** | 3,004 frames from 1,041 videos, 751 per recording, chosen from a proposer-only scan of all videos. Person boxes were pre-labelled by large language model (LLM) agents running Claude Opus 5.5 in Claude Code: two agents labelled each frame blind, and a third, the arbiter, relabelled the 149 frames on which they disagreed. These LLM labels held 502 person boxes in 377 frames. One of the authors then reviewed all 3,004 frames in the review tool (the tool logged a decision for 622 of them; the others needed no change). After review the labels hold 486 people in 361 frames. | Training the verifier; rules for choosing its threshold |
| **Validation set** | 261 frames from 76 one-minute pilot clips: 145 people in 119 frames, and 142 frames without a person. By body part: 48 full body, 42 lower body, 26 partial at the image edge, 26 upper body, 3 arm or hand only. The set was enriched with frames in which a pre-trained YOLO detector had missed people, so it is harder than everyday footage. LLM agents pre-labelled it by the same procedure, here with Claude Sonnet 5.5, and one of the authors then reviewed all 261 frames. | Comparing detectors during development; setting the scan schedule |
| **Dense validation clips** | The same 76 clips with about every other frame stored (50,777 frames), scored on every stored frame by the released verifier. One of the authors checked 484 of these frames in the review tool: 200 from 12 random daytime clips and 284 from 12 night clips. | Replaying the schedule, the hold and the enlargement without a GPU |
| **Everyday sample** | 2,000 unlabelled frames drawn from the 118 held-out videos, in three strata by the score of a preliminary verifier (131, 721 and 1,148 frames). | Weighting test-set rates back to everyday footage |
| **Test set** | 731 frames of the everyday sample, checked by one of the authors in the review tool, who boxed every person: all 131 frames that the preliminary verifier would have masked, 300 of the 721 near its threshold and 300 of the 1,148 below it. It holds 172 people in 52 frames, all in colour; none of its 217 infrared frames shows a person. | Evaluating OpenVeil and the alternatives |

**What the test set can and cannot show.** The test set was not built
independently of the method it tests:

- The reviewer drew its boxes while checking the pipeline's own proposals, with
  the Grounding DINO candidates offered for one-click adoption. It therefore
  cannot contain a person whom both the pipeline and the reviewer missed, and
  the shapes of its boxes favour Grounding DINO.
- The candidate crops of the test set also entered the training of the
  released verifier. Test-set scores therefore come from two **cross-fitted**
  models: the test videos were split into two halves, and each half was scored
  by a model trained on the training set and on the other half. Every test
  frame is scored by a model that had not seen its video, but the other half of
  the test videos was in that model's training data.
- The released threshold of 0.10 and the container rule were chosen with
  test-set results in view.
- Coverage in infrared could not be measured: no infrared frame of the test
  set, and none of the 284 reviewed night-clip frames, shows a person.

The validation set is not held out either: the released verifier was trained
on other minutes of most of the videos that its clips come from. Its labels
favour methods built on Grounding DINO, because most of its person boxes were
chosen from Grounding DINO candidates.

Measures used below:

- **Covered** (test set): at least 90% of the person's box lies under the kept
  boxes, after each kept box is enlarged by 20% per side as in masking.
- **Recall (IoU) / covers-half / loose**: the share of labelled people found, by
  three matching rules of increasing leniency: a kept box matches by IoU (the
  overlap of two boxes, 0 to 1, here ≥ 0.5), a kept box covers at least half of
  the person, or **loose** matching: a person counts as found if any kept box
  has IoU ≥ 0.5 with any labelled box drawn on any part of that person (so a
  box on a hand counts). Loose recall is the most lenient of the three.
- **False-box frames**: frames with a false box, that is, a kept box that after
  enlargement has less than 20% of its area on any single person box. *Big*
  false boxes cover at least 15% of the image.
- **Person-free frames with a mosaic**: of the 142 validation frames without a
  person, the share that would get any mosaic.
- **Area**: the share of frame area under a mosaic (or a false mosaic).
- **Weighting and intervals**: test-set rates are weighted back to the 2,000
  frames of the everyday sample (a frame from a stratum of N frames of which n
  were checked has weight N/n). Square brackets are 95% intervals.

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
   threshold. YOLO11x pre-trained on COCO, run per frame, put a box in 91–93%
   of the frames of a 90 s clip with people (confidence 0.05–0.25) at
   27.5–29.2 fps including encoding, but its recall was not measured in that
   trial.
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
7. **In-domain YOLO11l**, fine-tuned on the LLM labels of the training set
   before their review. Very fast and precise, but its recall on the
   validation set in this trial was at most 0.634. Rejected: far below the 95%
   target. In the later comparison under one protocol it covered 126 of the
   172 test-set people with no more false-box frames than the final pipeline
   ([Summary](#summary)).
8. **Hand-made features**: a gradient-boosted model on Grounding DINO scores,
   box geometry and colour. Better, but at 95% recall a quarter of the
   labelled colour frames still had a false positive. Rejected.
9. **Proposer and crop verifier.** Grounding DINO proposes candidates at a low
   threshold; a DINOv3 image classifier with a LoRA adapter looks at each crop
   and keeps or drops it. A small ViT-S backbone was not enough in these
   trials; ViT-L was. Lowering the candidate threshold from 0.10 to 0.05, and
   then retraining on tighter positive boxes and on the labels after their
   review, gave the final verifier.
10. **Schedule.** Replaying a sparse-then-dense schedule on clips scored densely
    by the final verifier showed that it keeps almost all of the recall at a
    fraction of the cost (section 5).
11. **Giant rule.** A rule to drop loose boxes around groups was checked frame by
    frame in the review tool and its threshold set to 0.55 (section 6).
12. **Smart cut**, so that only the stretches of video with a mosaic are
    re-encoded.
13. **Review tool**, used throughout from the proposer-and-verifier stage on.

## 4. Comparison table

These are the development trials, as measured in each trial; the set is named
in each cell. They were made on different sets and with different matching
rules, so the rows are not all comparable with each other. The comparison
under one protocol is the first table of the [Summary](#summary). "–" means
not measured. Speeds are on one A10 GPU unless noted.

| Model / configuration | Person recall | False-mask burden | Speed or cost | Verdict |
|---|---|---|---|---|
| YOLO11x (COCO), per frame, conf 0.05–0.25 | not measured | box in 91–93% of the frames of a 90 s clip with people | 27.5–29.2 fps including encoding | Rejected (superseded by pilots) |
| Pilot 1: YOLO11x + OWLv2 + ByteTrack | 44 / 53 annotated person events fully masked (83%) | 134 / 320 audit thumbnails false-positive only, 67% of masked area | 13.26 fps end to end; 2.73 node-hours per 80 clips | Rejected (misses) |
| Pilot 2: + privacy zones and retune | 53 / 53 events; 0 / 320 audit thumbnails with an unmasked person | 162 / 320 thumbnails false-positive only (66% of mask area); masks on 32.4% of all frame area | 16.27 fps per node; 820.7 node-hours projected for the full dataset | Rejected (precision, complexity) |
| Grounding DINO base, `worker` ≥ 0.06, margin 0 | validation set: 0.876 (loose, phrase study); 0.855 IoU / 0.869 covers-half (detector evaluation) | validation set: 59.2% of person-free frames, 4.57% area. Everyday sample: 52.4% of frames, 4.99% area | 4.84–5.20 fps measured; 2,391 A10-h for every frame of the dataset at an assumed 5.9 frames/s end to end (2,713–2,915 A10-h at the measured speeds) | Kept as proposer only |
| Grounding DINO, margin 0.05 | validation set: 0.800 | validation set: 33.8% / 0.77% area. Everyday sample: 30.2% / 1.69% | as above | Rejected |
| Grounding DINO, prompt `person.` only (thr 0.45) | validation set: 0.455 (loose) | validation set: 0.88% area | as above | Rejected |
| LocateAnything-3B | validation set: 0.772 IoU / 0.766 covers-half | validation set: 49.3% / 4.32% area | 1.99 s per frame | Rejected (slow, still boxes calves) |
| Track confirmation on Grounding DINO (three settings) | validation set: 0.855 / 0.800 / 0.731 | validation set: 47.2% / 34.5% / 12.0% of person-free frames | – | Rejected |
| YOLO11l in-domain, 1-class head | validation set: 0.262 | validation set: 2.8%. Everyday sample: 1.75% of frames | 75 fps (detector only) | Rejected (recall) |
| YOLO11l in-domain, COCO head, 10 layers frozen | validation set: 0.490 | validation set: 2.1%. Everyday sample: 2.85% | 73.1 fps | Rejected (recall) |
| YOLO11l in-domain, full fine-tune | validation set: 0.634 IoU / 0.579 covers-half | validation set: 4.9%. Everyday sample: 4.35% of frames, 0.18% area | 75.2 fps | Rejected (recall) |
| Gradient-boosted model on GD scores, geometry, colour | 90% / 95% of labelled people | 12.7% / 25.5% false-positive frames | – | Rejected |
| GD + DINOv3 ViT-S LoRA classifier, camera held out | AUC 0.938; 95% of labelled people | 47% of colour frames keep a false mosaic | – | Rejected |
| GD + DINOv3 ViT-L LoRA r32, candidates ≥ 0.10 | AUC 0.984, AP 0.901; 95% of labelled people | 12.5% false-positive colour frames (GD alone: 82.5%). Everyday sample: 8.4% [4.9–12.6] of frames, 0.27% area | – | Superseded |
| GD + DINOv3 ViT-L LoRA, candidates ≥ 0.05 | AUC 0.981; 95% of labelled people | 7.2% false-positive colour frames. Everyday sample: 6.6% [3.7–10.2], 0.83% area | – | Superseded by the final verifier |
| **OpenVeil as released: GD + DINOv3 ViT-L (final verifier), threshold 0.10** | **Test set: 172 / 172 people covered.** Validation set: 0.862 (loose) | **Test set: 3.3% of frames with a false box, 0.45% with a big one.** Schedule replay: 0.22% false area | 0.086 detector calls per frame in the replay (0.0508 in the real run); about 296 A10-h estimated for the day schedule at all hours, about 122 Grounding DINO A10-h for the real call count, assuming 5.9 frames/s for Grounding DINO and 160 crops/s for the verifier; verifier 287 crops/s measured on the GPU alone, 247 end to end | **Adopted** |

GD is Grounding DINO. Square brackets are 95% confidence intervals where the
trial computed them. "95% of labelled people" means the classifier threshold
was set to keep 95% of the 502 people of the LLM labels of the training set,
before their review, and the false-positive rate is read at that point.

## 5. How the final architecture was reached

**Why a proposer and a verifier.** The trials split cleanly into two groups.
Open-vocabulary detectors (Grounding DINO, LocateAnything) find people in odd
poses and at the image edge, but they cannot tell a crouching worker from a
calf: in the development trials about half or more of the person-free
validation frames got a mosaic (Grounding DINO 59%, LocateAnything 49%).
Detectors trained on our data (YOLO11l) were precise but missed a third or
more of the validation people. The comparison under one protocol says the same
([Summary](#summary)): no single detector covered more than 128 of the 172
test-set people at any false-box budget up to 20%.

A verifier can only remove boxes, so the proposer has to reach near-full
coverage at some threshold. Grounding DINO was the only detector compared
whose low-threshold candidates included nearly every person: its candidates
with a score of at least 0.05 covered all 172 test-set people (partly by
construction, see section 2), all 145 validation people and 481 of the 486
people of the reviewed training labels. The last two figures are favoured by
the way the labels were made: most validation and training boxes were chosen
from the same candidates. Its own score separates people from
the other candidates poorly: over the 9,260 test-set candidates its AUC was
0.713 [0.685–0.743], against 0.994 [0.985–0.998] for the cross-fitted
verifier. A classifier that only has to answer "does this crop show a person?"
removes the false alarms without giving up the coverage.

**Why DINOv3 with LoRA, and how much the choice matters.** The verifier must
learn a fine distinction from about 1,400 positive crops (1,444 of 50,756). A
large self-supervised backbone (DINOv3 ViT-L/16) gives strong features out of
the box, and a LoRA adapter (rank 32) adapts it cheaply; training took 35 min
on one A10.

Having a verifier matters much more than which one. Seven variants were
trained on the same crops, with the same recipe and the same two video folds,
one seed each: the released ViT-L with a rank-32 adapter, the same backbone
with a rank-8 adapter, ViT-B, ViT-S and DINOv2 ViT-L with a rank-32 adapter, a
frozen ViT-L with an MLP head, and a logistic regression on frozen ViT-L
features. All seven ranked the test-set candidates far better than Grounding
DINO's score did (AUC 0.989–0.997).

| Variant | False-box frames at the highest threshold that still covers all 172 test-set people (set on the test set) |
|---|---|
| **ViT-L, LoRA rank 32 (released)** | **2.25%** |
| ViT-L, LoRA rank 8 | 1.60% |
| ViT-B, LoRA rank 32 | 1.85% |
| ViT-L frozen, MLP head | 1.63% |
| DINOv2 ViT-L, LoRA rank 32 | 26.9% |
| ViT-S, LoRA rank 32 | 94.9% |

DINOv2 ViT-L and ViT-S end up there because one or two hard people forced a
low threshold. With each threshold set without the test set, to the highest
value that still covered 95% of the coverable people of the training set,
Grounding DINO's score covered 142 of the 172 test-set people with false boxes
in 76.3% of everyday frames; the logistic regression covered 171 with 20.1%;
and the other six variants covered 170–172 with 1.5–2.0%.

The comparison does not place the released model ahead of rank 8, ViT-B or the
frozen ViT-L with an MLP head. ViT-L with a rank-32 adapter is released
because it is the model that masked the case-study dataset. It was chosen
during development, before this comparison was run: in the development trials
it left far fewer false boxes than a ViT-S verifier, which was tested with
cameras held out (section 4). On this test set the adapter was not needed.
Where verifier time matters, ViT-B is a reasonable choice: it ran at 574 crops
per second end to end on one A10, against 247 for the released ViT-L, and with
its own threshold it left a similar share of false-box frames.

**Why the final verifier replaced the preliminary one.** A preliminary
verifier was trained on the LLM labels before their review, with a looser rule
for positive crops and without test-set crops. The final verifier was trained
on the reviewed labels, with a tighter rule for positives (a candidate is
positive when it covers at least half of a labelled person and that person
fills at least 20% of the candidate box) and with the test-set crops by
cross-fitting (section 2). On the test set, with the container rule, the
preliminary verifier covered every person up to a threshold of 0.19, where it
left false boxes in 3.1% of frames (0.95% big). The final verifier covered
every person up to 0.127, with 2.25% (0.35% big). The two differ in more than
their labels, so the lower rates cannot be attributed to the reviewed labels
alone.

**Why a threshold of 0.10.** On the test set the final verifier covered all
172 people at thresholds from 0.07 to 0.127, 171 from 0.15 to 0.425 and 170 at
0.5. The false-box frames fell from 3.3% at 0.10 to 2.4% at 0.125 and 1.1% at
0.5. The released value of 0.10 was set with this curve in view, so we also
chose thresholds by rules that do not use the frames on which the result is
scored:

| How the threshold was chosen | Threshold | Test-set people covered | False-box frames |
|---|---|---|---|
| **Released, with test-set results in view** | **0.10** | **172 / 172** | **3.3% [2.6–4.1]** |
| Training set, highest threshold covering 99% of the coverable people | 0.067 | 172 / 172 | 24.9% |
| Training set, 98% | 0.091 | 172 / 172 | 5.6% |
| Training set, 95% | 0.351 | 171 / 172 | 1.5% |
| Chosen on one half of the test videos, applied to the other | 0.12 / 0.56 | 170 / 172 | 1.6% |

Coverage stayed at 170 to 172 people under every rule tried, but the share of
false-box frames ranged from 1.5% to 100% (a rule that asks for every coverable
person gives a threshold at the bottom of the score scale). The half-and-half
choice is unstable, because each half holds only 86 people. **Set the
threshold on your own reviewed frames.**

**Why the schedule.** We replayed the released schedule (1 s checkpoints,
0.2 s dense mode after a hit, ±2 s hold) on the dense validation clips: 76
one-minute clips that the released verifier had scored on every stored frame
(about every other frame of the video). The comparison is a schedule that also
uses 1 s checkpoints but scans every stored frame while it is in dense mode:

| | Dense mode on every stored frame | Released schedule (dense step 0.2 s) |
|---|---|---|
| Validation set, loose recall | 0.876 | 0.862 |
| Validation people covered after hold and enlargement (of 145) | 140 | 140 |
| Boxes of the dense scan covered by the mask plan | 99.6% | 99.4% |
| Detector calls per frame | 0.229 | 0.086 |
| Estimated cost for the dataset | 548 A10-h for Grounding DINO + 243 for the verifier | 205 + 91, about 296 A10-h |

The released schedule made 37% of the detector calls. Longer dense steps save
more calls but lose boxes: with a dense step of 1 s the loose recall fell to
0.807. Without a dense mode at all, a day scan at strides of 1, 2 and 5 s would
have a loose recall of 0.786, 0.669 and 0.497, which is why the day schedule
needs its dense mode.

The costs assume 5.9 frames/s for Grounding DINO end to end and 160 crops/s
for the verifier on one A10. The verifier figure is conservative (247 crops/s
were measured end to end). The Grounding DINO figure is not: at the measured
speed of 4.84–5.20 fps its hours would be 13–22% higher and the totals about
9–15% higher. Running Grounding DINO on every video frame would cost 2,391
A10-h at the same assumed speed.

On the real dataset the schedule needed only 0.0508 calls per frame (with the
`clock` policy, sparse at night), because most footage has no people; at the
same assumed speeds that is about 122 A10-hours for Grounding DINO and about
54 for the verifier. Note that the replay measures the schedule mostly against
the model's own boxes, not against confirmed people.

At night the case study checked one frame every 5 s without a dense mode, a
cost decision made because people rarely enter the barns at night. Replayed on
the 12 night clips, this scan covered 82.2% of the boxes of a dense scan,
against 94.9% for a 1 s stride and 97.4% for the day schedule, which OpenVeil
uses by default. A 1 s stride would have cost about 20 more A10-hours (about
25 instead of 5 for the 11.5 million night frames). These boxes are verifier
output, not confirmed people: the review of 284 frames from these clips found
no person.

**Why per-frame rules and interpolation.** Duplicate boxes (IoU ≥ 0.85) and
loose "container" boxes around a group are common outputs of open-vocabulary
detectors. The two rules are simple and apply per frame. At the released
threshold the container rule lowered the share of everyday frames with a big
false box from 0.80% to 0.45% on the test set; one person lost the IoU match
and no person lost coverage. On the dense validation clips it reduced the
mosaic area by 27% (0.48% against 0.66% of the image), with 140 of the 145
validation people covered with or without it. The rule was chosen with
test-set results in view. Interpolation bridges the gaps that caused most
misses in the first pilot.

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
a person's box less than half covered by the boxes the rule keeps.

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
  were gaps of up to 3.4 s between detections of one person, and people at box
  edges. The replay on the dense validation clips later measured them. With
  neither, the mask plan covered 120 of the 145 validation people; the 20%
  enlargement alone covered 137, a ±2 s hold alone 136, and both together 140.
  Beyond 20% and ±1 s the coverage did not rise further, but the mosaic on
  person-free frames kept growing with the hold: at 20% enlargement it took
  0.67% of the image at ±1 s, 1.07% at ±2 s and 1.33% at ±3 s. Where a small
  mosaic area matters more than a wide margin, ±1 s is the better choice. On
  the 200 human-checked daytime frames of the dense validation clips the
  released setting covered all 27 people; without enlargement it covered 26,
  and without enlargement or hold 25.
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
set it was measured on. In particular:

- **Grounding DINO recall** for the same rule is 0.876 under the phrase study's
  loose matching and 0.855 (IoU) or 0.869 (covers-half) in the detector
  evaluation, all on the validation set. They are different matching rules,
  not different runs.
- **`person.` alone**: 0.455 recall at 0.88% area (stored phrase-study table).
- **In-domain YOLO11l with a frozen backbone, everyday sample**: 2.85% of
  frames at confidence 0.05.
- **Preliminary verifier at threshold 0.19 on the test set**: 3.1% false-box
  frames, 0.95% big, 0.53% false area, with the container rule (later re-run).
  An earlier evaluation recorded 3.5% / 1.35% / 1.19%.
- **Giant rule at 0.40**: worst covers 0.194 and 0.247 (re-run of the check).
- **Cost**: about 296 A10-h (205 Grounding DINO + 91 verifier, at 0.0858 calls
  per frame) is the estimate of the schedule replay with the container rule.
  An earlier replay without the rule gave 0.0853 calls per frame and about 295
  A10-h (204 + 91), and 547 instead of 548 Grounding DINO hours for the
  schedule whose dense mode scans every stored frame; in that earlier replay
  2.1% of frames had a box of the dense scan that the released schedule did
  not cover. Earlier versions of this page quoted those values. GPU time was
  not metered per step, so no measured cost is given. The 820.7 node-hours of
  pilot 2 is a projection for a different pipeline.
- **Dataset totals**: 1,173 videos (two unplayable files were removed from an
  initial 1,175) and 50,794,625 frames summed from the detection tables.
- **Review counts** in [case_study.md](case_study.md) are recomputed from the
  latest decision for every frame in the review log, which supersedes counts
  made earlier during the review.
- **LoRA rank** of the final adapter is 32 (from its stored configuration).
- **Duplicate suppression** after verification is at IoU ≥ 0.85, a mild rule;
  candidate NMS before verification is at IoU 0.6.
