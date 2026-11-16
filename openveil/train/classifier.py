# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Train the crop verifier: a DINOv3 backbone with LoRA adapters and a small MLP head.

The model sees one letterboxed crop per candidate box (:func:`openveil.detect.verifier.make_crop`)
and predicts "person" / "not person". Recipe of the production model (:class:`TrainConfig`
defaults): LoRA rank 32, alpha 64, dropout 0.05 on all linear layers of the backbone; head
``1024 -> 1024 -> 512 -> 2`` on the CLS embedding (``pooler_output``); AdamW lr 1e-4, weight
decay 0.01, 8 epochs, batch 32, 3 % linear warm-up then cosine decay to 10 %, label smoothing
0.1, bf16 autocast, gradient clipping at 1.0. Every epoch uses all positives and a fresh random
sample of ``neg_per_pos`` negatives per positive. Training augmentation: box edges jittered by
up to 10 %, horizontal flip, contrast x0.8-1.2, brightness +-25, grey-scale with probability
0.3 (so that colour is not the only cue; infrared frames are grey).

Evaluation is out-of-fold (:func:`cross_validate`): each fold of the crop table
(:func:`openveil.train.labels.build_crop_table`, folds by video by default) is scored by a model
trained on the other folds. The number of epochs is fixed in advance, never chosen on the held
out fold. The fold scores are saved after each fold, so an interrupted run resumes; a run with
other settings or another table starts clean and keeps the earlier file under a new name.

:func:`train_final` trains on every row and writes the folder the detector loads:
``adapter_config.json`` + ``adapter_model.safetensors`` (the LoRA adapter only),
``head.safetensors`` and ``meta.json``. The backbone itself is never saved: users download it
from its own source under its own licence.

torch, transformers and peft are imported inside the functions that need them; the data
preparation helpers (:func:`epoch_rows`, :func:`augment_crop`, :class:`CropDataset`,
:func:`lr_factor`, :func:`table_fingerprint`) need numpy only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..detect.verifier import IMAGENET_MEAN, IMAGENET_STD, make_crop, save_head

__all__ = ["TrainConfig", "read_image_rgb", "augment_crop", "eval_crop", "CropDataset", "epoch_rows", "lr_factor",
           "plan_steps", "table_fingerprint", "build_classifier", "train_model", "predict", "cross_validate", "train_final"]


@dataclasses.dataclass
class TrainConfig:
    """Training recipe (defaults = the production model)."""

    backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lr: float = 1e-4
    wd: float = 0.01
    epochs: int = 8
    batch: int = 32
    accum: int = 1
    neg_per_pos: float = 10.0
    size: int = 224
    pad: float = 0.10
    hidden: int = 1024
    smoothing: float = 0.1
    jitter: float = 0.10
    seed: int = 0
    workers: int = 4
    device: str = "cuda"
    dtype: str = "bfloat16"
    grad_ckpt: bool = False
    eval_every: int = 1
    image_root: str = ""

    def recipe(self) -> dict:
        """The settings that change the model (used in the fingerprint and in ``meta.json``)."""
        keys = ("backbone", "lora_r", "lora_alpha", "lora_dropout", "lr", "wd", "epochs", "batch", "accum", "neg_per_pos",
                "size", "pad", "hidden", "smoothing", "jitter", "seed")
        return {k: getattr(self, k) for k in keys}


# ---------------------------------------------------------------------------------------------
# data preparation (numpy only)
# ---------------------------------------------------------------------------------------------
def read_image_rgb(path: str) -> np.ndarray:
    """Read an image file as an RGB uint8 array (OpenCV, or Pillow)."""
    try:
        import cv2  # noqa: PLC0415

        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image  # noqa: PLC0415

        return np.asarray(Image.open(path).convert("RGB"))


def augment_crop(img_rgb: np.ndarray, box: Sequence[float], pad: float, size: int, rng: np.random.Generator,
                 jitter: float = 0.10) -> np.ndarray:
    """Training crop as float32 in [0, 1], HxWx3, before normalisation.

    Box jitter, horizontal flip (p 0.5), contrast x U(0.8, 1.2) and brightness + U(-25, 25)
    around mid-grey, and grey-scale (p 0.3), drawn from ``rng`` in this order.
    """
    c = make_crop(img_rgb, box, pad, size, rng, jitter=jitter)
    if rng.random() < 0.5:
        c = c[:, ::-1]
    c = c.astype(np.float32)
    c = (c - 127.5) * rng.uniform(0.8, 1.2) + 127.5 + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        c = np.repeat(c.mean(axis=2, keepdims=True), 3, axis=2)
    return np.clip(c, 0, 255) / 255.0


def eval_crop(img_rgb: np.ndarray, box: Sequence[float], pad: float, size: int) -> np.ndarray:
    """Evaluation crop as float32 in [0, 1] (no augmentation)."""
    return make_crop(img_rgb, box, pad, size).astype(np.float32) / 255.0


def _chw(c: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(((c - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)).astype(np.float32)


def row_path(row: Mapping[str, Any], image_root: str = "") -> str:
    """Image path of a crop-table row (``row["path"]``, or ``image_root`` joined with ``row["image"]``)."""
    if row.get("path"):
        return str(row["path"])
    return os.path.join(image_root, str(row["image"])) if image_root else str(row["image"])


class CropDataset:
    """Map-style dataset of crops for ``torch.utils.data.DataLoader`` (no torch import needed).

    Item ``i`` is ``(x, label, i)`` with ``x`` a normalised float32 ``3 x size x size`` array.
    Training crops are augmented with a generator seeded by ``(seed, epoch, row id)``, so an
    epoch is reproducible whatever the number of loader workers.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], train: bool, cfg: TrainConfig, epoch: int = 0,
                 loader: Callable[[str], np.ndarray] = read_image_rgb):
        self.rows, self.train, self.cfg, self.epoch, self.loader = list(rows), bool(train), cfg, int(epoch), loader

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = self.loader(row_path(r, self.cfg.image_root))
        if self.train:
            rng = np.random.default_rng([int(self.cfg.seed), int(self.epoch), int(r["id"])])
            c = augment_crop(img, r["box"], self.cfg.pad, self.cfg.size, rng, self.cfg.jitter)
        else:
            c = eval_crop(img, r["box"], self.cfg.pad, self.cfg.size)
        return _chw(c), int(r.get("label", 0)), i


def epoch_rows(pos: Sequence[Any], neg: Sequence[Any], neg_per_pos: float, seed: int, epoch: int) -> list:
    """Rows of one epoch: every positive plus a fresh sample of ``neg_per_pos`` negatives per positive."""
    rr = random.Random(seed * 1000 + epoch)
    return list(pos) + rr.sample(list(neg), min(len(neg), int(round(neg_per_pos * len(pos)))))


def plan_steps(n_pos: int, n_neg: int, cfg: TrainConfig) -> tuple[int, int]:
    """``(optimizer steps, warm-up steps)`` of a training run."""
    per_epoch = n_pos + min(n_neg, int(round(cfg.neg_per_pos * n_pos)))
    steps = max(1, cfg.epochs * (per_epoch // (cfg.batch * cfg.accum)))
    return steps, max(1, int(0.03 * steps))


def lr_factor(step: int, warm: int, steps: int) -> float:
    """Learning-rate factor: linear warm-up, then cosine decay from 1 to 0.1."""
    if step < warm:
        return (step + 1) / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))


def table_fingerprint(rows: Sequence[Mapping[str, Any]], fold_key: str = "fold") -> str:
    """Short hash of labels, boxes, folds and images of a table (detects a changed table)."""
    data = json.dumps([(r["id"], r["label"], r["box"], r.get(fold_key), r.get("image")) for r in rows])
    return hashlib.sha1(data.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# model (torch, transformers, peft)
# ---------------------------------------------------------------------------------------------
def build_classifier(cfg: TrainConfig):
    """Backbone with LoRA adapters + MLP head, as one ``torch.nn.Module``."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    from ..detect.verifier import build_head  # noqa: PLC0415

    class Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = AutoModel.from_pretrained(cfg.backbone, torch_dtype=getattr(torch, cfg.dtype)).to(cfg.device)
            if cfg.grad_ckpt:
                base.gradient_checkpointing_enable()
                base.enable_input_require_grads()
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules="all-linear",
                              bias="none", task_type="FEATURE_EXTRACTION")
            self.backbone = get_peft_model(base, lcfg)
            self.dim = int(base.config.hidden_size)
            self.head = build_head(self.dim, cfg.hidden).to(cfg.device)

        def forward(self, x):
            f = self.backbone(pixel_values=x).pooler_output
            return self.head(f.float())

    return Classifier()


def _n_trainable(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def predict(model, rows: Sequence[Mapping[str, Any]], cfg: TrainConfig, bs: int = 64) -> tuple[np.ndarray, float]:
    """p(person) for every row (no augmentation) and the crops per second."""
    import torch  # noqa: PLC0415

    model.eval()
    dl = torch.utils.data.DataLoader(CropDataset(rows, False, cfg), batch_size=bs, shuffle=False, num_workers=cfg.workers)
    out = np.zeros(len(rows), np.float32)
    dev = torch.device(cfg.device).type
    t0 = time.time()
    with torch.no_grad():
        for x, _, idx in dl:
            with torch.autocast(dev, dtype=getattr(torch, cfg.dtype), enabled=cfg.dtype != "float32"):
                logits = model(x.to(cfg.device, non_blocking=True))
            out[idx.numpy()] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
    return out, len(rows) / max(time.time() - t0, 1e-9)


def train_model(train_rows: Sequence[Mapping[str, Any]], cfg: TrainConfig, test_rows: Sequence[Mapping[str, Any]] | None = None,
                tag: str = "", log: Callable[[str], None] = print):
    """Train one model for a fixed number of epochs; returns ``(model, history)``.

    With ``test_rows`` the held-out AUC / AP is logged every ``eval_every`` epochs, for
    information only.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from .evaluate import metrics  # noqa: PLC0415

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = build_classifier(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"[{tag}] trainable parameters {_n_trainable(model) / 1e6:.1f} M | head input {model.dim}")
    pos = [r for r in train_rows if r["label"]]
    neg = [r for r in train_rows if not r["label"]]
    steps, warm = plan_steps(len(pos), len(neg), cfg)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, steps))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.smoothing)
    dev = torch.device(cfg.device).type
    hist = []
    for ep in range(cfg.epochs):
        rows = epoch_rows(pos, neg, cfg.neg_per_pos, cfg.seed, ep)
        ds = CropDataset(rows, True, cfg, epoch=ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers, drop_last=True,
                                         generator=torch.Generator().manual_seed(cfg.seed * 1000 + ep))
        model.train()
        opt.zero_grad(set_to_none=True)
        usable = (len(dl) // cfg.accum) * cfg.accum        # whole accumulation groups only
        tot, n, t0, seen = 0.0, 0, time.time(), 0
        t_first, seen_first = t0, 0
        for k, (x, y, _) in enumerate(dl):
            if k >= usable:
                break
            with torch.autocast(dev, dtype=getattr(torch, cfg.dtype), enabled=cfg.dtype != "float32"):
                logits = model(x.to(cfg.device, non_blocking=True))
            loss = crit(logits.float(), y.to(cfg.device, non_blocking=True)) / cfg.accum
            loss.backward()
            tot += loss.item() * cfg.accum
            n += 1
            seen += len(y)
            if (k + 1) % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if k == 2:
                t_first, seen_first = time.time(), seen
        rate = (seen - seen_first) / max(time.time() - t_first, 1e-9) if n > 3 else None
        h: dict[str, Any] = dict(epoch=ep + 1, loss=tot / max(n, 1), crops=seen, crops_per_s=rate, lr=sched.get_last_lr()[0],
                                 minutes=(time.time() - t0) / 60)
        if test_rows is not None and cfg.eval_every and ((ep + 1) % cfg.eval_every == 0 or ep + 1 == cfg.epochs):
            p, _ = predict(model, test_rows, cfg)
            h.update(metrics(test_rows, p), note="held-out fold, for information only")
        hist.append(h)
        log(f"[{tag}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()}))
    return model, hist


def check_split(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], split: str) -> None:
    """Raise if a video (or day, or camera) of the test fold is also in training."""
    from .labels import SPLITS  # noqa: PLC0415

    key = SPLITS[split]
    both = {key(r) for r in train} & {key(r) for r in test}
    if both:
        raise ValueError(f"{len(both)} {split} groups are in both training and test, e.g. {sorted(both)[0]}")


def cross_validate(rows: Sequence[Mapping[str, Any]], out_dir: str, cfg: TrainConfig, fold_key: str = "fold",
                   split: str = "video", folds: Sequence[int] | None = None, log: Callable[[str], None] = print) -> dict:
    """Out-of-fold scores of every row, saved to ``<out_dir>/oof_scores.json`` after each fold.

    Rows with ``row[fold_key] == -1`` are always in training. Already finished folds are skipped
    when the fingerprint (recipe + table) matches; otherwise the old file is kept as
    ``oof_scores_replaced_<time>.json`` and the run starts clean.
    """
    from .evaluate import metrics  # noqa: PLC0415

    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "oof_scores.json")
    state: dict = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    fp = dict(recipe=cfg.recipe(), split=split, fold_key=fold_key, table=table_fingerprint(rows, fold_key), rows=len(rows))
    if state.get("fingerprint") != fp:
        if state:
            stamp = int(time.time())
            shutil.copy(state_path, os.path.join(out_dir, f"oof_scores_replaced_{stamp}.json"))
            log("settings or table differ from the earlier run in this folder; earlier scores kept, starting clean")
        state = dict(fingerprint=fp, folds={})
    all_folds = sorted({int(r[fold_key]) for r in rows if int(r[fold_key]) >= 0})
    run = list(folds) if folds is not None else all_folds
    if not set(run) <= set(all_folds):
        raise ValueError(f"folds {run} are not all in the table's folds {all_folds}")
    import gc  # noqa: PLC0415

    for f in run:
        if str(f) in state["folds"]:
            log(f"fold {f} already done")
            continue
        tr = [r for r in rows if int(r[fold_key]) != f]
        te = [r for r in rows if int(r[fold_key]) == f]
        if not tr or not te:
            raise ValueError(f"fold {f} is empty")
        check_split([r for r in tr if int(r[fold_key]) >= 0], te, split)
        t0 = time.time()
        model, hist = train_model(tr, cfg, te, tag=f"fold {f}", log=log)
        p, rate = predict(model, te, cfg)
        m = metrics(te, p)
        state["folds"][str(f)] = dict(scores={str(r["id"]): round(float(s), 5) for r, s in zip(te, p)}, hist=hist, train_rows=len(tr),
                                      test_rows=len(te), infer_crops_per_s=rate, minutes=(time.time() - t0) / 60, **m)
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, state_path)
        log(f"fold {f}: AUC {m['auc']:.4f} AP {m['ap']:.4f} | {len(te)} test candidates | {(time.time() - t0) / 60:.1f} min")
        del model
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
    return state


def write_meta(out_dir: str, cfg: TrainConfig, hist: Sequence[Mapping[str, Any]] | None = None, extra: Mapping[str, Any] | None = None) -> dict:
    """Write ``meta.json`` (classes, crop ``size`` and ``pad``, backbone, recipe) next to the weights."""
    meta = dict(classes=["not person", "person"], size=cfg.size, pad=cfg.pad, backbone=cfg.backbone, recipe=cfg.recipe(),
                hist=list(hist or []), **dict(extra or {}))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def train_final(rows: Sequence[Mapping[str, Any]], out_dir: str, cfg: TrainConfig, log: Callable[[str], None] = print) -> str:
    """Train on every row and save adapter, head and ``meta.json`` to ``<out_dir>/final``.

    Only the LoRA adapter is saved (``save_pretrained`` of the PEFT model writes the adapter
    weights, not the backbone); the LoRA weights are never merged into the backbone here.
    """
    model, hist = train_model(rows, cfg, None, tag="final", log=log)
    dst = os.path.join(out_dir, "final")
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.backbone.save_pretrained(tmp)
    save_head(model.head, os.path.join(tmp, "head.safetensors"))
    write_meta(tmp, cfg, hist, dict(rows=len(rows), positives=int(sum(int(r["label"]) for r in rows)),
                                    table=table_fingerprint(rows)))
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst
