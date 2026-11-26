# Images for the documentation

This folder holds the images the documentation refers to: the review-tool
screenshots, a screen recording of the review tool, a synthetic mosaic
illustration and eight masked example stills of the released dataset. This
file lists every image and the rules each one must follow.

## Rules

1. **No recognisable person, ever.** Images from real footage may only show
   frames that are already masked by OpenVeil, and each one needs written
   approval from the data owner before it is added.
2. Review-tool screenshots and the screen recording are taken from the
   **synthetic demo** (`openveil review demo`), which contains no real people.
3. No file names, farm names or other identifying text may be added to an
   image. The masked example stills are full, uncropped frames of the public
   masked release: a burned-in camera date/time overlay, where the camera
   shows one, appears exactly as it does in the released dataset, and nothing
   in these stills has been cropped or mosaicked beyond the release's own
   masking. Any other readable on-screen text (for example a camera or site
   name) must be cropped or mosaicked before a still is added; the data owner
   confirms this by eye when approving a still.
4. Prefer PNG for screenshots and JPEG for video stills; keep each file under
   about 500 KB (the two Farm A stills, at their source resolution of
   2304 x 1296, are about 630 to 660 KB each; the screen recording of the
   review tool is larger, about 1.3 MB as MP4 and 1.7 MB as GIF).
5. Strip all metadata (EXIF, ICC profiles, comments) from stills before adding
   them.

## Review-tool screenshots (from the synthetic demo)

Used in [review_tool.md](../review_tool.md); `review_adopt_candidate.png` and
`review_magnifier.png` are also shown on the project page (`docs/index.html`),
and `review_overview.png` is its link-preview image (`og:image`). All five are
**synthetic demo data, no real people**: the frames are drawn by
`openveil review demo` (grey silhouettes, calf-like shapes, fences), so they
need no data-owner approval.

| File | Shows | Status |
|---|---|---|
| `review_overview.png` | the whole page: queue selector with progress, a field frame with a green person box and a dashed candidate (score 0.22) on an already pixelated figure, side panel. Synthetic demo data, no real people. | done |
| `review_adopt_candidate.png` | the dashed candidate on a figure just adopted: it is now a selected green box, and another dashed candidate (on a calf, score 0.07) is still visible. Synthetic demo data, no real people. | done |
| `review_edit_box.png` | a box that is too large, selected, with its 8 resize handles. Synthetic demo data, no real people. | done |
| `review_magnifier.png` | the hold-`Z` 4x magnifier over a figure half hidden behind a fence, with its dashed candidate. Synthetic demo data, no real people. | done |
| `review_unsure_note.png` | the same fence frame marked unsure (amber Unsure button, "Marked unsure"), with a note typed in. Synthetic demo data, no real people. | done |

### How they were made

Start the demo (`openveil review demo --port 8766 --no-browser`) and open each
page state with the `?state=` URL parameter (or single parameters such as
`queue`, `fid`, `sel`, `adopt`, `lens`, `unsure`, `note`, `minscore`, `theme`)
in a headless Chromium browser at a 1440x900 window, device scale factor 1,
with a fresh temporary profile, for example
`msedge --headless=new --window-size=1440,900 --virtual-time-budget=8000 --screenshot=FILE URL`.
The page sets `data-ready="1"` on `<html>` once a scripted state is applied.

| File | URL (after `http://127.0.0.1:8766/`) |
|---|---|
| `review_overview.png` | `?queue=all&fid=M003&filter=all&theme=light` |
| `review_adopt_candidate.png` | `?state=adopted&minscore=0.05&theme=light` |
| `review_edit_box.png` | `?state=loose&theme=light` |
| `review_magnifier.png` | `?queue=missed_person&fid=M002&filter=all&lens=330,195&theme=light` |
| `review_unsure_note.png` | `?state=unsure&theme=light` |

## Screen recording of the review tool (from the synthetic demo)

Used in [review_tool.md](../review_tool.md), in the README and on the project
page (`docs/index.html`). All three files are **synthetic demo data, no real
people**, so they need no data-owner approval.

| File | Shows | Status |
|---|---|---|
| `review_demo.mp4` | the review tool in use on the synthetic demo: 45 s, 1280 x 800, 15 frames per second, H.264 (High profile, Level 4.0), no sound. Shown on the project page. | done |
| `review_demo.gif` | the same recording at 864 x 540 and half the frame rate, looping. For Markdown pages on GitHub, which show a GIF in place but only link to a video file in the repository. | done |
| `review_demo_poster.png` | one frame of the recording (the box on a calf selected), 1280 x 800: the poster shown before the video plays. | done |

What the recording shows, in order, each step named by a caption above the
page: a correct proposal accepted with `Enter`; a box on a calf deleted with
`X`; the queue selector with the progress of each queue; a missed person fixed
with one click on a dashed candidate, and a second box drawn by dragging; the
magnifier (hold `Z`) over a figure half hidden behind a fence; that frame
marked unsure with a note; a loose box resized by its handles.

### How it was made

`python figures/record_review_demo.py` makes all three files (it needs Pillow,
PyAV with libx264, and Chrome, Chromium or Edge). The script

1. draws the demo frames with `openveil.review.demo`, without the seed
   decisions, so that every queue starts at 0, and serves them with the demo
   guard on; it refuses to go on unless the served manifest says `demo` and
   every frame is `synthetic`;
2. starts a headless Chromium browser with a fresh temporary profile and
   drives the real page through the Chrome DevTools Protocol, with real mouse
   and key input. After each step it checks on the page, and at the end in
   `decisions.jsonl`, that the step had its effect;
3. takes one screenshot per video frame. A headless screenshot contains no
   mouse pointer, so the pointer, a ripple on each click, a key cap for each
   key, the caption bar above the page and a ring around the queue selector
   are drawn onto the frames with Pillow. The caption bar is at the top
   because a video player lays its controls over the bottom of the picture
   when the film is paused.

Two things differ from what a user sees. The queue is chosen with the Down and
Enter keys (shown as key caps), because the drop-down list of a headless
browser takes no mouse input. And the recording starts from an empty demo,
whereas `openveil review demo` starts with a few frames already decided: untick
"skip reviewed" there to see the first two frames of the recording. Running
the script again gives the same content, but the MP4 and the GIF are not
byte-identical from run to run, because the frames are captured in real time.

## Masked example stills (real footage, already masked)

Used in [case_study.md](../case_study.md), the README and the project page
(`docs/index.html`). **Each needs
data-owner approval before it is added.** The stills are frames of the masked
release, never of the unmasked footage. Each file is a byte-for-byte copy of
the still the data owner approved (not re-encoded); those files carry no
metadata (only the JFIF header, with no thumbnail, and no EXIF, XMP, ICC
profile or comment segments).

| File | Shows | Status |
|---|---|---|
| `example_farmA_1.jpg` | Farm A, a frame of the released dataset after masking (2304 x 1296) | approved by the data owner, added |
| `example_farmA_2.jpg` | Farm A, a second frame of the released dataset after masking (2304 x 1296) | approved by the data owner, added |
| `example_farmB_1.jpg` | Farm B, a frame of the released dataset after masking (1280 x 720) | approved by the data owner, added |
| `example_farmA_r3.jpg` | FarmA_R3, two people masked (1920 x 1080) | approved by the data owner, added |
| `example_farmA_r2_small.jpg` | FarmA_R2, a partly visible person of 40 x 29 px masked (1280 x 720) | approved by the data owner, added |
| `example_farmA_r2_small_annotated.jpg` | The same frame with the mosaic outlined and a 3x inset of it, cut from the same masked pixels | approved by the data owner, added |
| `example_farmA_r2_lowscore.jpg` | FarmA_R2, a partly visible person of 29 x 58 px that the proposer scored only 0.25 in a separate scan, masked (1280 x 720) | approved by the data owner, added |
| `example_farmB_small.jpg` | FarmB_R1, a partly visible person of 47 x 84 px masked (1280 x 720) | approved by the data owner, added |

The data owner checked every still by eye for any person left unmasked and
for readable on-screen text (rules 1 and 3) before approving it. In the five
FarmA_R2 / FarmA_R3 / hard-case stills, a human reviewer had also confirmed
in the review tool that every person is covered.

## Diagrams

The README and the project page use drawn SVG figures (the banner and the
pipeline), made by `figures/make_figures.py` with the standard library only and
copied to `docs/assets/` for the project page. They are not in this folder.
All of them are drawn: they show no footage and no real person, and the people
of the banner exist only as the mosaic (no figure of a person is written to
the file). The pipeline diagram in [how_it_works.md](../how_it_works.md) is
written in Mermaid and needs no image.

| File | Shows | Status |
|---|---|---|
| `figures/banner.svg` (copy: `docs/assets/banner.svg`) | four drawn scenes side by side, each watched by a camera: a calf barn, a factory line with a robot arm, a laboratory bench and a meadow with a deer. In each scene the one person present is covered by a pixel mosaic with a green tick, while the calves, the machine, the instruments and the deer stay visible; a dashed box with a cross on one calf and on the deer marks a candidate the verifier rejected. It shows the kinds of footage OpenVeil is meant for; only the calf barn has been tested. 1280 x 420. | done |
| `figures/pipeline.svg`, `figures/pipeline_narrow.svg` (copies in `docs/assets/`) | the eight steps of the pipeline in two bands, detection on a GPU and review and apply without one, for wide (1000 x 352) and narrow (340 x 832) screens. The proposer and the verifier are marked as a pair: a detector proposes broadly, then a classifier verifies each candidate. | done |
| `mosaic_grid.png` | a **drawn, synthetic** frame (stylised figure, calf shape, fence; no footage) before and after the mosaic: the detector box, the box enlarged by 20% per side, and the grid cells inside it filled with their mean. Made with the OpenVeil mosaic code; no approval needed. | done |
