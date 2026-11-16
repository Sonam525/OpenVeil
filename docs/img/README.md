# Images for the documentation

This folder holds the images the documentation refers to: the review-tool
screenshots, a synthetic mosaic illustration and three masked example stills of
the released dataset. This file lists every image and the rules each one must
follow.

## Rules

1. **No recognisable person, ever.** Images from real footage may only show
   frames that are already masked by OpenVeil, and each one needs written
   approval from the data owner before it is added.
2. Review-tool screenshots are taken from the **synthetic demo**
   (`openveil review demo`), which contains no real people.
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
   2304 x 1296, are about 630 to 660 KB each).
5. Strip all metadata (EXIF, ICC profiles, comments) from stills before adding
   them.

## Review-tool screenshots (from the synthetic demo)

Used in [review_tool.md](../review_tool.md) and the README. All five are
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

## Masked example stills (real footage, already masked)

Used in [case_study.md](../case_study.md) and the README. **Each needs
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
| `example_farmA_r2_lowscore.jpg` | FarmA_R2, a partly visible person of 29 x 58 px with detector score 0.25, masked (1280 x 720) | approved by the data owner, added |
| `example_farmB_small.jpg` | FarmB_R1, a partly visible person of 47 x 84 px masked (1280 x 720) | approved by the data owner, added |

The data owner checked every still by eye for any person left unmasked and
for readable on-screen text (rules 1 and 3) before approving it. In the five
FarmA_R2 / FarmA_R3 / hard-case stills, a human reviewer had also confirmed
in the review tool that every person is covered.

## Diagrams

The pipeline diagram is written in Mermaid inside the Markdown files and needs
no image.

| File | Shows | Status |
|---|---|---|
| `mosaic_grid.png` | a **drawn, synthetic** frame (stylised figure, calf shape, fence; no footage) before and after the mosaic: the detector box, the box enlarged by 20% per side, and the grid cells inside it filled with their mean. Made with the OpenVeil mosaic code; no approval needed. | done |
