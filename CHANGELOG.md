# Changelog

All notable changes to OpenVeil are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-24

The corrected release. It supersedes 1.0.0, which was published from an
earlier commit before these changes were pushed. Each release is archived on
Zenodo, which assigns it a version DOI; the concept DOI
[10.5281/zenodo.24505557](https://doi.org/10.5281/zenodo.24505557) cites all
versions. See [Citing OpenVeil](README.md#citing-openveil).

### What 1.1.0 contains

- **Detection** (`openveil detect`): a scan schedule over each video (day and
  night policies), Grounding DINO proposals, a crop verifier (LoRA adapter and
  MLP head on DINOv3 ViT-L/16; weights on Hugging Face under the DINOv3
  License), per-frame rules and gap interpolation, written to one box table per
  video. `openveil episodes` lists where the giant rule fires.
- **Review tool** (`openveil review build|serve|demo|export`): a local web page
  for checking the hard frames, with a synthetic demo.
- **Masking** (`openveil apply video|images`): mosaics people in videos with a
  smart cut that re-encodes only the parts that contain a mosaic, and in image
  folders and tar shards. `openveil verify` checks a masked video against its
  source.
- **Verifier training** (the `openveil.train` package): train your own crop
  verifier.
- The `production` preset (`openveil config show --preset production`), the
  documentation in `docs/` (usage, configuration, how it works, design
  rationale, limitations, review tool, case study, model card), the
  project page and a test suite on synthetic data.

### Changed

- The default Grounding DINO prompt is now the case-study prompt
  `worker. person. cow. calf.` (positive phrases `worker` and `person`,
  competitor phrases `cow` and `calf`) instead of `person.`. It is the prompt
  the released crop verifier was trained with, so a run with the defaults uses
  the evaluated configuration; only the day/night policy still differs from the
  case study (`always_full` instead of `clock`). For footage with other
  animals, replace the competitor phrases and re-check the verifier
  ([docs/configuration.md](docs/configuration.md)).
  `configs/example_barn.yaml` no longer repeats the prompt. This change of a
  default is why this release is a minor version and not a patch.
- The package reports its version correctly (`openveil --version` prints
  1.1.0; the code archived as 1.0.0 still reported 0.1.0), and the
  development status is Production/Stable.

### Added

- Release metadata: `.zenodo.json` for the Zenodo GitHub integration, this
  changelog, and `CITATION.cff` with both authors' ORCID iDs, the release
  date, the concept DOI and the version DOI of 1.0.0, and the crop verifier
  weights as a reference.
- The citation sections of the README ("Citing OpenVeil") and of the project
  page now give the release and the concept DOI, and the author lists in the
  README, the project page and `CITATION.cff` give Miel Hostens's ORCID iD.

### Fixed

- The documentation now states the caveats of the HUM731 evaluation: the
  verifier was cross-fitted over the HUM731 videos (each scoring model had the
  other half in its training data), the HUM731 boxes were drawn while checking
  the pipeline's proposals, and the 0.10 threshold and the container rule were
  chosen with HUM731 in view, so full coverage holds partly by construction.
- Corrected figures in the documentation: the schedule-replay cost (about
  295 A10-hours: 204 Grounding DINO + 91 classifier), the verifier speed
  (287 crops/s on the GPU alone, 247 end to end), the share of frames with a big
  false box (0.45%), the Grounding DINO false-mosaic area (4.57%) and the
  comparison of the previous and final verifiers.
- The case study and the limitations now note that the code that masked the
  case-study dataset left 3,216 night frames at 134 chunk boundaries outside
  the hold; OpenVeil closes these gaps.
- The table of example stills in `docs/img/README.md` renders as one table.

## [1.0.0] - 2026-11-24

Superseded by 1.1.0. The GitHub release v1.0.0 was published from commit
`16d1513`, before the commits that make up 1.1.0 were pushed. Zenodo archived
it as [10.5281/zenodo.24505558](https://doi.org/10.5281/zenodo.24505558);
that archive cannot be changed. Its code, `CITATION.cff`, README and project
page still report version 0.1.0, and the last three lack Miel Hostens's ORCID
iD. It uses the default prompt `person.`, and its documentation does not yet
have the corrections listed under 1.1.0. Please use 1.1.0.

### Changed

- The README was redesigned; long reference material moved to
  `docs/usage.md`, `docs/configuration.md` and `docs/limitations.md`, and the
  results and design summary tables moved to `docs/case_study.md` and
  `docs/design_rationale.md`.

### Added

- The project page (`docs/index.html`, served by GitHub Pages) and the drawn
  banner and pipeline figures (`figures/make_figures.py`).
- More approved, masked example stills from the case study (FarmA_R2, FarmA_R3,
  and small, partly visible people).
- Citations of the case-study dataset (version 1.2, DOI 10.57967/hf/11689) and
  its data descriptor (arXiv:2611.21706) in the README, project page, case
  study, model card and `CITATION.cff`.

## [0.1.0] - 2026-11-16

The first public version: detection, the review tool, masking, verifier
training, the documentation and the test suite.

[1.1.0]: https://github.com/Sonam525/OpenVeil/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Sonam525/OpenVeil/compare/9ad4d1a...v1.0.0
[0.1.0]: https://github.com/Sonam525/OpenVeil/commit/9ad4d1a
