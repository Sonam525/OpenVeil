# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Consistency checks for the documentation and licence files.

These tests read only the repository's own text files. They check that

* the documentation files exist and contain nothing that must not be published
  (e-mail addresses, absolute local paths, internal storage paths and, when a
  private word list is supplied through ``OPENVEIL_SCRUB_WORDS``, names of data
  sources);
* every ``openveil ...`` command shown in the docs follows the CLI contract;
* relative links and image references resolve, or the image is listed as planned
  in ``docs/img/README.md``;
* the configuration reference in the README names real config keys and, where
  the value is a plain number, matches the ``production`` preset;
* LICENSE is the full Apache-2.0 text and CITATION.cff is well formed.
"""
from __future__ import annotations

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_FILES = [
    "README.md",
    "docs/how_it_works.md",
    "docs/design_rationale.md",
    "docs/review_tool.md",
    "docs/case_study.md",
    "docs/model_card.md",
    "docs/img/README.md",
    "THIRD_PARTY.md",
]
OTHER_FILES = ["LICENSE", "NOTICE", "CITATION.cff"]

# Private data-source and account names are not stored in the repository, not even as
# hashes (short words are easy to reverse). Before a release, point OPENVEIL_SCRUB_WORDS
# at a private word list (one word per line, kept outside the repository) and the name
# check below runs; without it the check is skipped.
SCRUB_WORDS_ENV = "OPENVEIL_SCRUB_WORDS"

HOLDER = "Copyright 2026 Haiyu Yang and Miel Hostens"
OLD_HOLDER = "The OpenVeil " + "Authors"          # the placeholder holder used before the authors were named
WEIGHTS_REPO = "Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16"


def _private_words() -> set:
    path = os.environ.get(SCRUB_WORDS_ENV, "")
    if not path or not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip() and not w.lstrip().startswith("#")}


# The CLI contract: command words -> allowed options.
CLI = {
    ("detect",): {"--out", "--config", "--preset", "--workers", "--device", "--factory", "--retry-failed"},
    ("episodes",): {"--min-share"},
    ("review", "build"): {"--tables", "--videos", "--out", "--per-queue", "--images", "--boxes"},
    ("review", "serve"): {"--port"},
    ("review", "demo"): {"--out", "--port"},
    ("review", "export"): {"--overrides"},
    ("apply", "video"): {"--table", "--out", "--overrides", "--full-reencode"},
    ("apply", "images"): {"--boxes", "--out", "--accept", "--allow-missing", "--no-rules"},
    ("verify",): set(),
    ("config", "show"): {"--preset"},
}

PLANNED_IMAGES_README = "docs/img/README.md"


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
        return f.read()


def _slug(heading: str) -> str:
    """GitHub-style anchor of a Markdown heading."""
    s = heading.strip().lower()
    s = re.sub(r"[`*_]", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def _anchors(text: str) -> set:
    out = set()
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.add(_slug(m.group(2)))
    return out


def _code_lines(text: str):
    """Lines inside fenced code blocks, with backslash continuations joined."""
    in_code, buf = False, ""
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            continue
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        yield buf + line
        buf = ""


class TestDocFilesExist(unittest.TestCase):
    def test_all_files_present_and_non_empty(self):
        for rel in DOC_FILES + OTHER_FILES:
            with self.subTest(file=rel):
                path = os.path.join(ROOT, rel)
                self.assertTrue(os.path.isfile(path), rel)
                self.assertGreater(os.path.getsize(path), 200, rel)


class TestNothingPrivate(unittest.TestCase):
    def test_no_private_strings(self):
        email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
        drive_path = re.compile(r"(?<![\w])[A-Za-z]:[\\/]")
        internal = re.compile(r"/(?:d" r"bfs|mnt|Workspace|local_" r"disk0)/", re.IGNORECASE)
        for rel in DOC_FILES + OTHER_FILES:
            text = _read(rel)
            with self.subTest(file=rel):
                self.assertIsNone(email.search(text), f"e-mail address in {rel}")
                self.assertIsNone(drive_path.search(text), f"absolute local path in {rel}")
                self.assertIsNone(internal.search(text), f"internal storage path in {rel}")

    def test_no_private_names(self):
        private = _private_words()
        if not private:
            self.skipTest(f"no private word list ({SCRUB_WORDS_ENV} not set)")
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__") and not d.endswith(".egg-info")]
            for name in files:
                if not name.endswith((".py", ".md", ".html", ".yaml", ".toml", ".cff", ".txt", ".json"))                         and name not in ("LICENSE", "NOTICE", ".gitignore"):
                    continue
                full = os.path.join(root, name)
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    words = set(re.findall(r"[a-z0-9_]+", f.read().lower()))
                with self.subTest(file=os.path.relpath(full, ROOT)):
                    self.assertEqual(sorted(words & private), [], "private name found")

    def test_only_public_recording_pseudonyms(self):
        for rel in DOC_FILES:
            for name in re.findall(r"Farm[A-Z]_R\d+", _read(rel)):
                with self.subTest(file=rel, name=name):
                    self.assertIn(name, {"FarmA_R1", "FarmA_R2", "FarmA_R3", "FarmB_R1"})


class TestCliExamples(unittest.TestCase):
    def test_commands_follow_contract(self):
        seen = set()
        for rel in DOC_FILES:
            for line in _code_lines(_read(rel)):
                line = line.split("#", 1)[0].strip()
                if not line.startswith("openveil "):
                    continue
                words = line.split()[1:]
                key = tuple(words[:2]) if tuple(words[:2]) in CLI else tuple(words[:1])
                with self.subTest(file=rel, line=line):
                    self.assertIn(key, CLI, f"unknown command: {line}")
                    opts = {w for w in words if w.startswith("--")}
                    self.assertTrue(opts <= CLI[key], f"options {sorted(opts - CLI[key])} not in contract")
                    seen.add(key)
        # The README quickstart must show every command of the contract at least once.
        readme_keys = set()
        for line in _code_lines(_read("README.md")):
            words = line.split("#", 1)[0].split()
            if words[:1] == ["openveil"]:
                k = tuple(words[1:3]) if tuple(words[1:3]) in CLI else tuple(words[1:2])
                readme_keys.add(k)
        self.assertEqual(set(CLI) - readme_keys, set(), "README misses some commands")


class TestLinksAndImages(unittest.TestCase):
    def test_relative_links_resolve(self):
        link = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
        planned = _read(PLANNED_IMAGES_README)
        for rel in DOC_FILES:
            text = re.sub(r"```.*?```", "", _read(rel), flags=re.DOTALL)   # code blocks
            text = re.sub(r"`[^`\n]*`", "", text)                            # inline code
            base = os.path.dirname(os.path.join(ROOT, rel))
            for target in link.findall(text):
                if re.match(r"^[a-z]+:", target):
                    continue
                path, _, anchor = target.partition("#")
                with self.subTest(file=rel, target=target):
                    full = os.path.normpath(os.path.join(base, path)) if path else os.path.join(ROOT, rel)
                    is_image = path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
                    if is_image and not os.path.exists(full):
                        self.assertIn(os.path.basename(path), planned, f"image {path} neither present nor planned")
                        continue
                    self.assertTrue(os.path.exists(full), f"broken link {target}")
                    if anchor and full.endswith(".md"):
                        with open(full, "r", encoding="utf-8") as f:
                            self.assertIn(anchor, _anchors(f.read()), f"missing anchor #{anchor}")

    def test_review_screenshots_are_referenced(self):
        tutorial = _read("docs/review_tool.md")
        for name in ("review_overview.png", "review_adopt_candidate.png", "review_edit_box.png",
                     "review_magnifier.png", "review_unsure_note.png"):
            with self.subTest(image=name):
                self.assertIn(f"img/{name}", tutorial)
                self.assertIn(name, _read(PLANNED_IMAGES_README))

    def test_mermaid_diagrams(self):
        for rel in ("README.md", "docs/how_it_works.md"):
            with self.subTest(file=rel):
                self.assertIn("```mermaid\nflowchart", _read(rel))


class TestConfigReference(unittest.TestCase):
    """The README's configuration tables against openveil.config (skipped if not importable)."""

    @classmethod
    def setUpClass(cls):
        try:
            import sys
            sys.path.insert(0, ROOT)
            from openveil.config import get_preset, to_dict
        except Exception as e:  # noqa: BLE001 - any import problem means: skip
            raise unittest.SkipTest(f"openveil.config not importable: {e}")
        cls.cfg = to_dict(get_preset("production"))

    def _rows(self):
        """(section, [dotted keys], value cell) for every row of the configuration tables."""
        text = _read("README.md")
        part = text.split("## Configuration reference", 1)[1].split("\n## ", 1)[0]
        section = None
        for line in part.splitlines():
            m = re.match(r"^\*\*.*?\*\*\s*\(`(\w+)`\)", line)
            if m:
                section = m.group(1)
                continue
            if not line.startswith("| `"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            names = re.findall(r"`([\w.]+)`", cells[0])
            keys, parent = [], None
            for n in names:
                parts = n.split(".")
                if parts[0] in self.cfg and isinstance(self.cfg[parts[0]], dict):
                    path = parts
                elif len(parts) > 1:
                    path = [section] + parts
                elif parent is not None:
                    path = parent + parts
                else:
                    path = [section] + parts
                parent = path[:-1]
                keys.append(path)
            yield keys, cells[1]

    def _get(self, path):
        node = self.cfg
        for p in path:
            self.assertIsInstance(node, dict, ".".join(path))
            self.assertIn(p, node, f"README names unknown config key {'.'.join(path)}")
            node = node[p]
        return node

    def test_keys_exist_and_numbers_match(self):
        n_checked = 0
        for keys, value_cell in self._rows():
            values = [v.strip() for v in value_cell.split(",")]
            for i, path in enumerate(keys):
                with self.subTest(key=".".join(path)):
                    actual = self._get(path)
                    if len(values) == len(keys) and re.fullmatch(r"-?\d+(\.\d+)?", values[i]):
                        self.assertIsInstance(actual, (int, float))
                        self.assertAlmostEqual(float(actual), float(values[i]))
                        n_checked += 1
        self.assertGreater(n_checked, 25)


class TestLicenceFiles(unittest.TestCase):
    def test_license_is_full_apache_2(self):
        text = _read("LICENSE")
        self.assertIn("Apache License\n                           Version 2.0, January 2004", text)
        for heading in ("1. Definitions.", "2. Grant of Copyright License.", "3. Grant of Patent License.",
                        "4. Redistribution.", "5. Submission of Contributions.", "6. Trademarks.",
                        "7. Disclaimer of Warranty.", "8. Limitation of Liability.",
                        "9. Accepting Warranty or Additional Liability.", "END OF TERMS AND CONDITIONS",
                        "APPENDIX: How to apply the Apache License to your work."):
            with self.subTest(heading=heading):
                self.assertIn(heading, text)

    def test_notice_and_model_card_say_built_with_dinov3(self):
        for rel in ("NOTICE", "README.md", "docs/model_card.md"):
            with self.subTest(file=rel):
                self.assertIn("Built with DINOv3", _read(rel))

    def test_citation_cff(self):
        text = _read("CITATION.cff")
        self.assertRegex(text, r"(?m)^cff-version: 1\.2\.0$")
        self.assertNotIn(OLD_HOLDER, text)
        self.assertIn("https://github.com/Sonam525/OpenVeil", text)
        self.assertRegex(text, r"(?m)^license: Apache-2\.0$")
        self.assertNotRegex(text, r"(?mi)^\s*doi:")
        self.assertIn("arXiv preprint", text)
        try:
            import yaml
        except ImportError:
            return
        data = yaml.safe_load(text)
        for key in ("cff-version", "message", "title", "authors", "version", "license"):
            self.assertIn(key, data)
        self.assertEqual([(a["given-names"], a["family-names"]) for a in data["authors"]],
                         [("Haiyu", "Yang"), ("Miel", "Hostens")])
        for a in data["authors"]:
            self.assertIn("Cornell University", a["affiliation"])
            self.assertNotIn("email", a)
        self.assertEqual(data["authors"][0]["orcid"], "https://orcid.org/0009-0007-5125-4958")


class TestAuthorsAndWeights(unittest.TestCase):
    def test_copyright_holder_everywhere(self):
        for top in ("openveil", "tests", "configs"):
            for root, dirs, files in os.walk(os.path.join(ROOT, top)):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for name in files:
                    if not name.endswith((".py", ".yaml", ".html")):
                        continue
                    rel = os.path.relpath(os.path.join(root, name), ROOT)
                    head = "\n".join(_read(rel).splitlines()[:4])
                    with self.subTest(file=rel):
                        self.assertIn(HOLDER, head)
        self.assertIn(HOLDER, _read("NOTICE"))
        self.assertIn(HOLDER, _read("pyproject.toml"))
        for rel in DOC_FILES + OTHER_FILES + ["pyproject.toml"]:
            with self.subTest(file=rel):
                self.assertNotIn(OLD_HOLDER, _read(rel))

    def test_weights_release_is_documented(self):
        for rel in ("README.md", "docs/model_card.md", "THIRD_PARTY.md", "NOTICE"):
            text = _read(rel)
            with self.subTest(file=rel):
                self.assertIn(WEIGHTS_REPO, text)
                self.assertIn("DINOv3 License", text)
                self.assertNotRegex(text, r"(?i)weights[^.]{0,40}\bnot (yet )?(released|distributed)")
        readme = _read("README.md")
        self.assertIn("snapshot_download", readme)
        self.assertIn("transformers>=4.56,<5", _read("pyproject.toml"))

    def test_example_stills_carry_no_metadata(self):
        import struct

        img = os.path.join(ROOT, "docs", "img")
        stills = sorted(n for n in os.listdir(img) if n.lower().endswith((".jpg", ".jpeg")))
        for name in stills:
            with self.subTest(image=name):
                self.assertIn(name, _read(PLANNED_IMAGES_README))
                with open(os.path.join(img, name), "rb") as f:
                    data = f.read()
                self.assertEqual(data[:2], b"\xff\xd8")
                i, markers = 2, []
                while i + 4 <= len(data) and data[i] == 0xFF and data[i + 1] != 0xDA:
                    markers.append(data[i + 1])
                    i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
                self.assertEqual(data[i + 1], 0xDA, "no start of scan found")
                # APP1..APP15 (EXIF, XMP, ICC, ...) and COM segments must not be present
                self.assertEqual([m for m in markers if 0xE1 <= m <= 0xEF or m == 0xFE], [])


if __name__ == "__main__":
    unittest.main()
