# Limitations

What OpenVeil does and does not guarantee. The numbers come from the case study
([case_study.md](case_study.md)); how they were measured is in
[design_rationale.md](design_rationale.md#2-how-we-measured).

- **Not a guarantee.** No method can guarantee that no person remains in
  millions of frames. **Human review remains necessary** before any release,
  and a private channel for reporting a missed person is strongly recommended.
- **One case study.** OpenVeil is meant for any footage in which people are
  not the subject, but it has been tested on one kind only: four recordings of
  calf barns on two farms with similar barns and cameras. There are no results
  for factories, laboratories, warehouses, wildlife cameras or other farms.
- **The released weights are a starting point elsewhere.** The verifier was
  trained on calf-barn footage only, and the default prompt names cows and
  calves. Other footage needs its own reviewed frames and its own threshold,
  and possibly a verifier trained on those frames
  ([usage.md](usage.md#other-footage-than-calf-barns)).
- **What the test set can show.** Coverage was measured on the test set: 731
  human-checked frames of everyday footage, in which all 172 people were
  covered. Read this figure with its limits:
  - The reviewer drew the test-set boxes while checking the pipeline's own
    proposals. A person missed by both the pipeline and the reviewer is not
    counted anywhere.
  - The verifier scores on the test set are cross-fitted: each frame was scored
    by a model that had not seen its video, but the other half of the test
    videos was in that model's training data. The released model was trained
    on all the data and has no held-out estimate of its own.
  - The threshold of 0.10 was chosen with the test-set results in view.
    Thresholds chosen without the frames they were scored on still covered 170
    to 172 of the 172 people, but the share of frames with a false box ranged
    from 1.5% to 100%.
  - The 172 people appear in only 52 frames. With no miss among them, the
    one-sided 95% upper bound on the miss rate is about 1.7% per person, and
    about 5.6% with the 52 frames as the unit.
  - It is a single-frame test, without the schedule and the hold. Those were
    tested by a replay on the validation clips, where the whole pipeline
    covered 140 of 145 people (loose recall 0.862); that figure is not held out
    either, because the released verifier was trained on other minutes of most
    of the same videos.
- **Infrared.** Coverage in infrared could not be measured: none of the 217
  infrared frames of the test set and none of the 284 reviewed frames of the
  night clips shows a person.
- **Hardest cases.** Brief or partial appearances (limbs or boots at the image
  edge, people behind walls) and small distant figures. With a sparse night
  schedule, a person in view for less than about 5 s can be missed; the default
  `always_full` policy avoids that at a higher cost.
- **False mosaics** fall on animals and objects and can hide behaviour of
  interest. Annotations made on unmasked footage are not changed by OpenVeil.
- **Giant rule.** Where it fires, a person boxed both as a whole and in parts may
  be only partly covered. Review its episodes (`openveil episodes`).
- **Smart-cut assumptions.** Smart cut needs 8-bit 4:2:0 H.264/HEVC streams
  without B-frames and with parameter sets in band; files that fail the
  eligibility probe or the check are re-encoded in full. In re-encoded GOPs,
  pixels outside the mosaic change slightly. H.264 and HEVC are re-encoded with
  x264/x265; other codecs are re-encoded in full.
- **Night schedule.** With the `clock` policy the night chunks are scanned
  every 5 s; a person who stays in view is masked throughout (the last frame of
  every chunk is a checkpoint too). The code that masked the case-study
  dataset did not have this checkpoint: 3,216 night frames at 134 chunk
  boundaries were left outside the hold. The boxes on either side of these
  gaps are verifier output, not confirmed people. OpenVeil closes these gaps.
- **Pixels only.** OpenVeil masks people in the image. It does not handle
  audio, metadata, file names, burned-in text or other identifying information.
  Mosaic cells are coarse (1/36 of the image height), but no formal
  re-identification test was done.
- **Licences.** The DINOv3 base model and the verifier weights come with use
  restrictions (DINOv3 License); the FFmpeg in the PyAV wheels is linked with
  the GPL encoders x264 and x265; encoding H.264/HEVC may need patent licences
  for commercial use. See [THIRD_PARTY.md](../THIRD_PARTY.md).
