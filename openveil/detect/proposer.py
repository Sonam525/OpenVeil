# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""First stage of the detector: open-vocabulary box proposals from Grounding DINO.

Grounding DINO receives the frame and a text prompt made of phrases separated by full stops,
for example ``"person."`` or ``"worker. person. cow. calf."``. For every predicted box it
returns a probability per prompt token. OpenVeil groups the tokens into phrases and scores a box
with the highest probability over the **positive phrases** (the ones that mean "person").

**Competitor phrases** are phrases in the prompt that are *not* positive, such as the animals in
the footage. They are not used in the score: their job is to give Grounding DINO a better
answer for an object that looks a little like a person, which lowers that object's
person-phrase probability. Which phrases work best is a property of the footage, so the prompt,
the positive phrases and the competitor phrases are all configuration. The only check is that
every configured phrase occurs in the prompt.

Output per frame: ``[[x1, y1, x2, y2, score], ...]`` in pixels of the frame, for every box with
``score >= cand_thr``. With ``detailed=True`` each box is ``[x1, y1, x2, y2, s_phrase_1, ...]``
with one probability per prompt phrase (in prompt order), the format used to build training
tables.

torch, transformers and Pillow are imported when the model is loaded, so this module imports
with numpy only.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["DEFAULT_MODEL", "phrase_groups", "phrase_indices", "decode_boxes", "GroundingDinoProposer", "make_proposer"]

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")


def phrase_groups(tokens: Sequence[str]) -> tuple[list[list[int]], list[str]]:
    """Split a tokenised prompt into phrases.

    ``tokens`` is the token list of the whole prompt (special tokens included). Returns
    ``(groups, names)``: ``groups[k]`` holds the token positions of phrase ``k`` and ``names[k]``
    is its text (word pieces starting with ``##`` are joined to the previous piece). Phrases end
    at a ``"."`` token; a trailing phrase without a full stop is kept too.
    """
    groups: list[list[int]] = []
    names: list[str] = []
    cur: list[int] = []
    words: list[str] = []
    for i, t in enumerate(tokens):
        if t in SPECIAL_TOKENS:
            continue
        if t == ".":
            if cur:
                groups.append(cur)
                names.append(_join(words))
                cur, words = [], []
            continue
        cur.append(i)
        words.append(t)
    if cur:
        groups.append(cur)
        names.append(_join(words))
    return groups, names


def _join(words: Sequence[str]) -> str:
    out = ""
    for w in words:
        if w.startswith("##"):
            out += w[2:]
        else:
            out += (" " if out else "") + w
    return out


def _norm(phrase: str) -> str:
    return " ".join(phrase.lower().strip().rstrip(".").split())


def prompt_phrases(prompt: str) -> list[str]:
    """Phrases of a prompt string, as written (``"worker. person."`` -> ``["worker", "person"]``)."""
    return [_norm(p) for p in prompt.split(".") if _norm(p)]


def phrase_indices(names: Sequence[str], wanted: Sequence[str], what: str = "positive") -> list[int]:
    """Positions of the ``wanted`` phrases among the prompt phrase ``names``.

    Raises ``ValueError`` when a wanted phrase is not in the prompt (a misspelt phrase would
    otherwise silently score every box 0).
    """
    normed = [_norm(n) for n in names]
    out = []
    for w in wanted:
        if _norm(w) not in normed:
            raise ValueError(f"{what} phrase {w!r} is not in the prompt phrases {list(names)}")
        out += [i for i, n in enumerate(normed) if n == _norm(w)]
    return sorted(set(out))


def decode_boxes(probs: np.ndarray, boxes: np.ndarray, width: int, height: int, groups: Sequence[Sequence[int]],
                 positive: Sequence[int], cand_thr: float, detailed: bool = False) -> list[list[float]]:
    """Turn Grounding DINO outputs of one image into scored pixel boxes.

    ``probs``: ``(queries, tokens)`` sigmoid probabilities; ``boxes``: ``(queries, 4)`` normalised
    ``(cx, cy, w, h)``. A phrase's probability is the highest probability of its tokens; a box's
    score is the highest probability over the ``positive`` phrases. Boxes with
    ``score >= cand_thr`` are returned, in query order.
    """
    probs = np.asarray(probs, np.float32)
    boxes = np.asarray(boxes, np.float32)
    if probs.size == 0 or not groups:
        return []
    ph = np.stack([probs[:, list(g)].max(axis=-1) for g in groups], axis=-1)        # (queries, phrases)
    sc = ph[:, list(positive)].max(axis=-1)
    keep = np.nonzero(sc >= cand_thr)[0]
    cx, cy, w, h = boxes[keep, 0], boxes[keep, 1], boxes[keep, 2], boxes[keep, 3]
    x1, y1, x2, y2 = (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height
    if detailed:
        return [[float(a), float(b), float(c), float(d), *[float(v) for v in ph[q]]] for a, b, c, d, q in zip(x1, y1, x2, y2, keep)]
    return [[float(a), float(b), float(c), float(d), float(s)] for a, b, c, d, s in zip(x1, y1, x2, y2, sc[keep])]


class GroundingDinoProposer:
    """Grounding DINO box proposer.

    Parameters
    ----------
    prompt, positive_phrases, competitor_phrases
        See the module docstring. Defaults: the case-study prompt ``"worker. person. cow. calf."``
        (the one the released verifier was trained with) with positive ``["worker", "person"]``
        and competitor ``["cow", "calf"]``.
    model_id
        Hugging Face id or local folder (default ``IDEA-Research/grounding-dino-base``).
    shortest_edge, longest_edge
        Input resize of the image processor (800 / 1333).
    cand_thr
        Lowest score returned (0.05).
    device, dtype
        ``"cuda"`` and ``"float16"`` autocast by default; ``dtype="float32"`` disables autocast.
    batch
        Frames per forward pass (4); a CUDA out-of-memory error halves it for the rest of the run.
    prep_threads
        Threads that run the image processor in parallel.
    """

    def __init__(self, prompt: str = "worker. person. cow. calf.", positive_phrases: Sequence[str] = ("worker", "person"),
                 competitor_phrases: Sequence[str] = ("cow", "calf"), model_id: str = DEFAULT_MODEL, shortest_edge: int = 800,
                 longest_edge: int = 1333, cand_thr: float = 0.05, device: str = "cuda", dtype: str = "float16",
                 batch: int = 4, prep_threads: int = 4, detailed: bool = False):
        self.prompt = prompt.strip()
        if not self.prompt.endswith("."):
            self.prompt += "."
        self.positive_phrases = [str(p) for p in positive_phrases]
        self.competitor_phrases = [str(p) for p in competitor_phrases]
        if not self.positive_phrases:
            raise ValueError("at least one positive phrase is needed")
        both = {_norm(p) for p in self.positive_phrases} & {_norm(p) for p in self.competitor_phrases}
        if both:
            raise ValueError(f"phrases {sorted(both)} are both positive and competitor")
        written = prompt_phrases(self.prompt)
        phrase_indices(written, self.positive_phrases, "positive")         # early check on the prompt text
        phrase_indices(written, self.competitor_phrases, "competitor")
        self.model_id = model_id
        self.shortest_edge, self.longest_edge = int(shortest_edge), int(longest_edge)
        self.cand_thr = float(cand_thr)
        self.device, self.dtype = device, dtype
        self.batch = max(1, int(batch))
        self.prep_threads = max(1, int(prep_threads))
        self.detailed = detailed
        self.processor = None
        self.model = None
        self.groups: list[list[int]] = []
        self.phrases: list[str] = []
        self.positive_idx: list[int] = []
        self.competitor_idx: list[int] = []
        self._pool = None
        self.stats: Counter = Counter()

    def load(self) -> "GroundingDinoProposer":
        """Load processor and model (needs torch, transformers and Pillow)."""
        import torch  # noqa: PLC0415
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415

        proc = AutoProcessor.from_pretrained(self.model_id)
        proc.image_processor.size = {"shortest_edge": self.shortest_edge, "longest_edge": self.longest_edge}
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        if torch.device(self.device).type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        ids = proc.tokenizer(self.prompt)["input_ids"]
        toks = proc.tokenizer.convert_ids_to_tokens(ids)
        self.set_tokens(toks)
        self.processor, self.model = proc, model
        return self

    def set_tokens(self, tokens: Sequence[str]) -> None:
        """Set the phrase groups from the tokenised prompt and check the configured phrases."""
        self.groups, self.phrases = phrase_groups(tokens)
        self.positive_idx = phrase_indices(self.phrases, self.positive_phrases, "positive")
        self.competitor_idx = phrase_indices(self.phrases, self.competitor_phrases, "competitor")

    def _prep(self, bgr: np.ndarray):
        from PIL import Image  # noqa: PLC0415

        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return self.processor(images=[Image.fromarray(rgb)], text=[self.prompt], return_tensors="pt")

    def _forward(self, inputs):
        import torch  # noqa: PLC0415

        dev_type = torch.device(self.device).type
        with torch.no_grad(), torch.autocast(dev_type, dtype=getattr(torch, self.dtype), enabled=self.dtype != "float32"):
            out = self.model(**inputs)
        return out.logits.float().sigmoid(), out.pred_boxes.float()

    def __call__(self, frames_bgr: Sequence[np.ndarray]) -> list[list[list[float]]]:
        """Proposals for a list of BGR frames: per frame ``[[x1, y1, x2, y2, score], ...]``."""
        if self.model is None:
            self.load()
        import torch  # noqa: PLC0415

        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            self._pool = ThreadPoolExecutor(max_workers=self.prep_threads)
        res: list[list[list[float]]] = []
        k = 0
        while k < len(frames_bgr):
            sub = list(frames_bgr[k:k + self.batch])
            t0 = time.time()
            preps = list(self._pool.map(self._prep, sub))
            inputs = {a: torch.cat([p[a] for p in preps]).to(self.device) for a in preps[0].keys()}
            self.stats["prep_s"] += time.time() - t0
            t0 = time.time()
            try:
                probs, boxes = self._forward(inputs)
            except torch.cuda.OutOfMemoryError:
                import gc  # noqa: PLC0415

                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                self.stats["oom"] += 1
                if self.batch == 1:
                    time.sleep(5)
                    if self.stats["oom"] > 50:
                        raise
                self.batch = max(1, self.batch // 2)
                continue
            k += len(sub)
            probs_np, boxes_np = probs.cpu().numpy(), boxes.cpu().numpy()
            for b, im in enumerate(sub):
                h, w = im.shape[:2]
                res.append(decode_boxes(probs_np[b], boxes_np[b], w, h, self.groups, self.positive_idx, self.cand_thr, self.detailed))
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize()
            self.stats["gd_s"] += time.time() - t0
            self.stats["gd_frames"] += len(sub)
        return res


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def make_proposer(spec: Any = None, device: str | None = None, cand_thr: float | None = None) -> GroundingDinoProposer:
    """Build a :class:`GroundingDinoProposer` from a config section (mapping or dataclass).

    Recognised keys (``openveil.config.ProposerConfig``): ``model_id``, ``prompt``,
    ``positive_phrases``, ``competitor_phrases``, ``shortest_edge``, ``longest_edge``, ``batch``,
    ``fp16`` (or ``dtype``), ``cand_thr``; ``device`` comes from the run settings.
    """
    kw: dict[str, Any] = {}
    for key in ("prompt", "positive_phrases", "competitor_phrases", "shortest_edge", "longest_edge", "batch", "dtype", "prep_threads"):
        v = _get(spec, key)
        if v is not None:
            kw[key] = v
    if "dtype" not in kw and _get(spec, "fp16") is not None:
        kw["dtype"] = "float16" if _get(spec, "fp16") else "float32"
    mid = _get(spec, "model_id") or _get(spec, "model")
    if mid:
        kw["model_id"] = mid
    dev = device or _get(spec, "device")
    if dev:
        kw["device"] = dev
    thr = cand_thr if cand_thr is not None else _get(spec, "cand_thr")
    if thr is not None:
        kw["cand_thr"] = float(thr)
    return GroundingDinoProposer(**kw)
