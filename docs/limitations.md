# Limitations

What OpenVeil does and does not guarantee. The numbers come from the case study
([case_study.md](case_study.md)); how they were measured is in
[design_rationale.md](design_rationale.md#2-how-we-measured).

- **Not a guarantee.** Recall was measured on a 261-frame validation set
  (VAL261, recall 0.862 under loose matching) and on a 731-frame human-reviewed
  sample of everyday frames (HUM731, 172 of 172 people covered, but only 52
  frames with people). These are small samples from one domain. The HUM731
  boxes were drawn while checking the pipeline's proposals, so it cannot
  contain a person missed by both the pipeline and the reviewer; the verifier
  was cross-fitted over the HUM731 videos, and the 0.10 threshold was chosen
  with HUM731 in view, so these results are optimistic. **Human review
  remains necessary** before any release, and a private channel for reporting a
  missed person is strongly recommended.
- **Hardest cases.** Brief or partial appearances (limbs or boots at the image
  edge, people behind walls), small distant figures, and night footage. Recall
  on infrared footage with people was not measured. With a sparse night
  schedule, a person in view for less than about 5 s can be missed; the default
  `always_full` policy avoids that at a higher cost.
- **Domain.** The verifier was trained on calf-barn footage only. On other
  scenes, review a sample and retrain or adjust the threshold.
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
  boundaries were left outside the hold (whether those boxes are people is
  not known). OpenVeil closes these gaps.
- **Pixels only.** OpenVeil masks people in the image. It does not handle
  audio, metadata, file names, burned-in text or other identifying information.
  Mosaic cells are coarse (1/36 of the image height), but no formal
  re-identification test was done.
- **Licences.** The DINOv3 base model and the verifier weights come with use
  restrictions (DINOv3 License); the FFmpeg in the PyAV wheels is linked with
  the GPL encoders x264 and x265; encoding H.264/HEVC may need patent licences
  for commercial use. See [THIRD_PARTY.md](../THIRD_PARTY.md).
