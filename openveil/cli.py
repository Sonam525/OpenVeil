# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Command line interface: ``openveil <command> ...``.

::

    openveil detect   VIDEO... --out DIR [--config FILE|--preset production] [--workers N] [--device cuda]
                      [--retry-failed] [--factory MODULE:FUNCTION]
    openveil episodes DIR_OF_TABLES [--min-share 0.55]
    openveil review build   --tables DIR --videos DIR --out REVIEW_DIR [--per-queue N]
    openveil review build   --images DIR --boxes BOXES.json --out REVIEW_DIR
    openveil review serve   REVIEW_DIR [--port 8765]
    openveil review demo    [--out DIR] [--port 8766]
    openveil review export  REVIEW_DIR [--overrides OUT.json]
    openveil apply video    VIDEO --table T.json.gz --out OUT.mp4 [--overrides O.json] [--full-reencode]
    openveil apply images   (DIR|TAR) --boxes BOXES.json --out OUT [--accept 0.10] [--allow-missing]
    openveil verify   SRC.mp4 OUT.mp4
    openveil config show [--preset production]

Every command returns exit code 0 on success and 1 when its result is not usable (a video
without a box table, a masked file that failed its check, ...); argument errors give 2.
Optional dependencies are imported only by the command that needs them.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from typing import Any, Callable, List, Optional, Sequence

from . import __version__

VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".mkv", ".avi", ".ts", ".mts")
REVIEW_COMMANDS = {
    "build": ("openveil.review.build", "build a review folder from box tables + videos, or images + boxes"),
    "serve": ("openveil.review.server", "serve a review folder on 127.0.0.1"),
    "demo": ("openveil.review.demo", "build and serve the synthetic demo (no data needed)"),
    "export": ("openveil.review.export", "export decisions: human labels, summary, overrides"),
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _err(msg: str) -> None:
    print(f"openveil: {msg}", file=sys.stderr)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--config", metavar="FILE", help="config file (YAML or JSON) with the values that differ from its preset")
    g.add_argument("--preset", metavar="NAME", default=None, help="built-in preset (default: production)")


def _load_cfg(args: argparse.Namespace):
    from .config import load_config  # noqa: PLC0415

    return load_config(getattr(args, "config", None) or getattr(args, "preset", None) or "production")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=1, default=str))


# optional packages -> the extra that installs them
_EXTRA_OF = {"av": "video", "PIL": "images", "cv2": "images", "scipy": "video", "yaml": "yaml", "torch": "detect",
             "transformers": "detect", "peft": "detect", "safetensors": "detect", "sklearn": "train"}


def _install_hint(e: ImportError) -> str:
    mod = (getattr(e, "name", None) or "").split(".")[0]
    extra = _EXTRA_OF.get(mod)
    if extra == "images" and mod == "PIL":
        extra = "review"
    return f"{e} (install it with  pip install \"openveil[{extra}]\")" if extra else str(e)


def _expand_videos(items: Sequence[str]) -> List[str]:
    """Video paths from files, folders (their video files, not recursive) and glob patterns
    (expanded here as well, because the Windows shell does not expand them)."""
    out: List[str] = []
    for it in items:
        if os.path.isdir(it):
            out.extend(sorted(os.path.join(it, n) for n in os.listdir(it) if n.lower().endswith(VIDEO_EXTENSIONS)))
        elif os.path.exists(it):
            out.append(it)
        elif glob.has_magic(it):
            out.extend(sorted(p for p in glob.glob(it) if os.path.isfile(p)))
        else:
            raise FileNotFoundError(f"no such video or folder: {it}")
    seen, uniq = set(), []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _resolve_factory(spec: Optional[str]) -> Optional[Callable]:
    """``module:function`` -> the function (a picklable model factory ``f(cfg) -> (proposer, verifier)``)."""
    if not spec:
        return None
    mod, sep, name = spec.partition(":")
    if not sep or not mod or not name:
        raise ValueError(f"--factory must be MODULE:FUNCTION, got {spec!r}")
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise ValueError(f"--factory {spec!r} is not callable")
    return fn


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def cmd_detect(args: argparse.Namespace) -> int:
    from .detect.run import SetupError, run_many  # noqa: PLC0415

    cfg = _load_cfg(args)
    if args.device:
        cfg.run.device = args.device
    videos = _expand_videos(args.videos)
    if not videos:
        _err("no video files found")
        return 1
    try:
        summ = run_many(videos, args.out, cfg, workers=args.workers, factory=_resolve_factory(args.factory),
                        in_process=args.in_process, retry_failed=args.retry_failed)
    except SetupError as e:
        _err(f"cannot detect: {e}")
        return 1
    print(f"{summ['done']} of {summ['videos']} videos have a box table in {args.out}")
    return 0 if not summ["missing"] and not summ["abandoned"] else 1


def _table_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no such table or folder: {path}")
    return sorted(os.path.join(path, n) for n in os.listdir(path)
                  if n.endswith((".json.gz", ".json")) and not n.startswith(("summary", "overrides")))


def cmd_episodes(args: argparse.Namespace) -> int:
    from .rules import firing_episodes  # noqa: PLC0415
    from .table import TableError, load_table  # noqa: PLC0415

    cfg = _load_cfg(args)
    a = cfg.apply
    min_share = a.min_share if args.min_share is None else args.min_share
    report, n_tables, n_eps = {}, 0, 0
    for p in _table_files(args.tables):
        try:
            t = load_table(p)
        except (TableError, ValueError, KeyError, OSError) as e:
            _err(f"skipped {os.path.basename(p)}: {e}")
            continue
        n_tables += 1
        eps = firing_episodes(t, min_share=min_share, min_inner=a.min_inner, inner_frac=a.inner_frac,
                              gap_s=a.episode_gap_s, keep_kinds=a.keep_kinds, pad=a.pad)
        report[t.video] = eps
        n_eps += len(eps)
        if args.json:
            continue
        for e in eps:
            print(f"{t.video}  {e['start_s']:9.2f}-{e['end_s']:9.2f} s  frames {e['frames']:4d}  "
                  f"max share {e['max_share']:.2f}  min covered {e['min_covered_after']:.2f}  "
                  f"look at t={e['t_at_min_s']:.2f} s (pts {e['pts_at_min']})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(min_share=min_share, episodes=report), f, indent=1)
    print(f"{n_eps} giant-rule episode(s) in {n_tables} table(s) at min_share {min_share}"
          + (f" -> {args.json}" if args.json else ""))
    return 0 if n_tables else 1


def cmd_apply_video(args: argparse.Namespace) -> int:
    from .apply.video import mask_video  # noqa: PLC0415
    from .table import apply_overrides, load_overrides, load_table  # noqa: PLC0415

    cfg = _load_cfg(args)
    table = load_table(args.table, validate=True)
    if args.overrides:
        ov = load_overrides(args.overrides)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        entry = None
        if "add" in ov or "delete" in ov or "keep" in ov:
            entry = ov
        else:
            for key in (table.video, stem, os.path.basename(args.video)):
                if key in ov:
                    entry = ov[key]
                    break
        if entry is None:
            print(f"no overrides for {table.video} in {os.path.basename(args.overrides)}")
        else:
            table, st = apply_overrides(table, entry, cfg)
            print("overrides: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    res = mask_video(args.video, table, args.out, cfg, full_reencode=args.full_reencode)
    if args.result:
        with open(args.result, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
    _print_json({k: v for k, v in res.items() if not k.startswith("attempt")})
    stats = next((res[k]["stats"] for k in sorted(res, reverse=True)
                  if k.startswith("attempt") and isinstance(res[k], dict) and "stats" in res[k]), {})
    if stats.get("mode") == "full" and stats.get("full_reason") and stats.get("full_reason") != "requested":
        print(f"smart cut not possible ({stats['full_reason']}); the video was re-encoded in full")
    if not res.get("ok"):
        _err(f"masking {os.path.basename(args.video)} failed its check; nothing was written to {args.out}")
        return 1
    return 0


def cmd_apply_images(args: argparse.Namespace) -> int:
    from .apply.images import load_boxes_json, mask_images  # noqa: PLC0415

    cfg = _load_cfg(args)
    res = mask_images(args.src, load_boxes_json(args.boxes, accept=args.accept), args.out, cfg, rules=not args.no_rules,
                      allow_absent=args.allow_missing)
    _print_json(res)
    if res.get("planned_absent"):
        _err(f"boxes for {len(res['planned_absent'])} image(s) match no image: {res['planned_absent'][:10]}")
    if not res.get("ok"):
        _err(f"masking the images failed: {res.get('why', 'check failed')}")
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .apply.video import verify_pair  # noqa: PLC0415

    res = verify_pair(args.src, args.out)
    _print_json(res)
    print("OK" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


def cmd_config_show(args: argparse.Namespace) -> int:
    from .config import to_dict  # noqa: PLC0415

    data = to_dict(_load_cfg(args))
    fmt = args.format
    if fmt == "yaml":
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            fmt = "json"
            _err("PyYAML is not installed; showing JSON")
        else:
            sys.stdout.write(yaml.safe_dump(data, sort_keys=False))
            return 0
    _print_json(data)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Only reached for ``openveil review`` without a sub-command (the sub-commands are dispatched in :func:`main`)."""
    args.parser.print_help()
    return 2


# ---------------------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="openveil",
        description="Detect, verify and mosaic people in video datasets before release, "
                    "with an interactive human review tool.")
    ap.add_argument("--version", action="version", version=f"openveil {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("detect", help="detect people; one box table per video")
    p.add_argument("videos", nargs="+", metavar="VIDEO", help="video files, folders of videos or glob patterns")
    p.add_argument("--out", required=True, metavar="DIR", help="folder for the box tables (<name>.json.gz)")
    _add_config_args(p)
    p.add_argument("--workers", type=int, default=None, metavar="N", help="worker processes sharing the device (default: run.workers)")
    p.add_argument("--device", default=None, help="torch device, e.g. cuda, cuda:1 or cpu (default: run.device)")
    p.add_argument("--in-process", action="store_true", help="run in this process (no restarts or stall watchdog; for debugging)")
    p.add_argument("--retry-failed", action="store_true",
                   help="forget earlier failures (OUT/failed/<name>.json) of these videos and try them again")
    p.add_argument("--factory", default=None, metavar="MODULE:FUNCTION",
                   help="build the models with this function f(cfg) -> (proposer, verifier) instead of Grounding DINO "
                        "+ the crop verifier; openveil.detect.fake:factory runs without any model (see docs/how_it_works.md)")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("episodes", help="list where the giant rule fires (frames worth reviewing)")
    p.add_argument("tables", metavar="DIR_OF_TABLES", help="folder of box tables (or one table)")
    p.add_argument("--min-share", type=float, default=None, help="giant rule share (default: apply.min_share, 0.55)")
    _add_config_args(p)
    p.add_argument("--json", default=None, metavar="FILE", help="also write the episodes to this JSON file")
    p.set_defaults(func=cmd_episodes)

    p = sub.add_parser("review", help="human review: build | serve | demo | export",
                       description="Human review of person boxes. Sub-commands: "
                                   + "; ".join(f"{k}: {v[1]}" for k, v in REVIEW_COMMANDS.items())
                                   + ". Run 'openveil review <sub-command> --help' for their options.")
    p.add_argument("review_command", nargs="?", choices=sorted(REVIEW_COMMANDS), metavar="{build,serve,demo,export}")
    p.set_defaults(func=cmd_review, parser=p)

    p = sub.add_parser("apply", help="mask videos or images")
    asub = p.add_subparsers(dest="apply_command", metavar="{video,images}")
    p.set_defaults(func=lambda a: (a.parser.print_help(), 2)[1], parser=p)
    q = asub.add_parser("video", help="mask one video (smart cut) and check the result")
    q.add_argument("video", metavar="VIDEO")
    q.add_argument("--table", required=True, metavar="T.json.gz", help="its box table")
    q.add_argument("--out", required=True, metavar="OUT.mp4", help="masked video (written only when its check passes)")
    q.add_argument("--overrides", default=None, metavar="O.json", help="reviewer corrections (from 'openveil review export')")
    q.add_argument("--full-reencode", action="store_true", help="re-encode the whole video instead of a smart cut")
    q.add_argument("--result", default=None, metavar="FILE", help="also write the full result JSON here")
    _add_config_args(q)
    q.set_defaults(func=cmd_apply_video)
    q = asub.add_parser("images", help="mask a folder or tar of JPEG/PNG images")
    q.add_argument("src", metavar="DIR|TAR")
    q.add_argument("--boxes", required=True, metavar="BOXES.json",
                   help="{image name: [[x1, y1, x2, y2(, kind)(, score)], ...]} in pixels of the upright image "
                        "(the layouts of 'openveil review build --boxes' work too)")
    q.add_argument("--out", required=True, metavar="OUT", help="output folder (or tar when the input is a tar)")
    q.add_argument("--no-rules", action="store_true", help="the boxes are final: no giant rule and no enlargement")
    q.add_argument("--accept", type=float, default=0.10,
                   help="a scored box below this is a candidate, not a person, and is not masked (default 0.10)")
    q.add_argument("--allow-missing", action="store_true",
                   help="go on when boxes are given for names that match no image (default: fail, write nothing)")
    _add_config_args(q)
    q.set_defaults(func=cmd_apply_images)

    p = sub.add_parser("verify", help="check a masked video against its source")
    p.add_argument("src", metavar="SRC.mp4")
    p.add_argument("out", metavar="OUT.mp4")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("config", help="show the settings")
    csub = p.add_subparsers(dest="config_command", metavar="{show}")
    p.set_defaults(func=lambda a: (a.parser.print_help(), 2)[1], parser=p)
    q = csub.add_parser("show", help="print a preset or a config file with every value")
    _add_config_args(q)
    q.add_argument("--format", choices=("yaml", "json"), default="yaml", help="output format (default yaml; json without PyYAML)")
    q.set_defaults(func=cmd_config_show)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # The review sub-commands have their own parsers; hand them the rest of the line.
    if len(argv) >= 2 and argv[0] == "review" and argv[1] in REVIEW_COMMANDS:
        try:
            mod = importlib.import_module(REVIEW_COMMANDS[argv[1]][0])
            return int(mod.main(argv[2:]) or 0)
        except ImportError as e:
            _err(_install_hint(e))
            return 1
        except (FileNotFoundError, ValueError) as e:
            _err(str(e))
            return 1
        except KeyboardInterrupt:
            _err("interrupted")
            return 130
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except ImportError as e:
        _err(_install_hint(e))
        return 1
    except (FileNotFoundError, ValueError) as e:
        _err(str(e))
        return 1
    except KeyboardInterrupt:
        _err("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
