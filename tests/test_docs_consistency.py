# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Consistency checks for the documentation and licence files.

These tests read only the repository's own text files. They check that

* the documentation files exist and contain nothing that must not be published
  (e-mail addresses, absolute local paths, internal storage paths and, when a
  private word list is supplied through ``OPENVEIL_SCRUB_WORDS``, names of data
  sources);
* every ``openveil ...`` command shown in the docs and on the project page
  follows the CLI contract, and the README with ``docs/usage.md`` shows them all;
* relative links and image references (Markdown and HTML) resolve, or the image
  is listed as planned in ``docs/img/README.md``, and in-page anchors exist;
* the configuration reference (``docs/configuration.md``) names real config keys
  and, where the value is a plain number, matches the ``production`` preset;
* the drawn figures are up to date with ``figures/make_figures.py``, accessible,
  copied unchanged to ``docs/assets/`` and not ignored by git;
* LICENSE is the full Apache-2.0 text, CITATION.cff is well formed, and the release metadata
  (CITATION.cff, .zenodo.json, CHANGELOG.md) agree with the package version and each other.
"""
from __future__ import annotations

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_FILES = [
    "README.md",
    "docs/usage.md",
    "docs/configuration.md",
    "docs/limitations.md",
    "docs/how_it_works.md",
    "docs/design_rationale.md",
    "docs/review_tool.md",
    "docs/case_study.md",
    "docs/model_card.md",
    "docs/img/README.md",
    "THIRD_PARTY.md",
    "CHANGELOG.md",
]
OTHER_FILES = ["LICENSE", "NOTICE", "CITATION.cff", ".zenodo.json"]
# The GitHub Pages project page (served from docs/) and the drawn figures.
PAGE_FILES = ["docs/index.html", "docs/style.css", "docs/site.js"]
FIGURES = ["banner.svg", "pipeline.svg", "pipeline_narrow.svg"]
FIGURE_FILES = [f"figures/{n}" for n in FIGURES] + [f"docs/assets/{n}" for n in FIGURES]
CONFIG_REFERENCE = "docs/configuration.md"

# Private data-source and account names are not stored in the repository, not even as
# hashes (short words are easy to reverse). Before a release, point OPENVEIL_SCRUB_WORDS
# at a private word list (one word per line, kept outside the repository) and the name
# check below runs; without it the check is skipped.
SCRUB_WORDS_ENV = "OPENVEIL_SCRUB_WORDS"

HOLDER = "Copyright 2026 Haiyu Yang and Miel Hostens"
OLD_HOLDER = "The OpenVeil " + "Authors"          # the placeholder holder used before the authors were named
WEIGHTS_REPO = "Sonam5/OpenVeil-Verifier-DINOv3-ViT-L16"
ORCIDS = {("Haiyu", "Yang"): "0009-0007-5125-4958", ("Miel", "Hostens"): "0000-0001-5376-976X"}
# DOIs that Zenodo has assigned (never invent one: add a DOI only once Zenodo shows it).
# The concept DOI cites all versions and resolves to the latest one. Once Zenodo shows the
# version DOI of a release, a docs-only commit adds it here, to the CITATION.cff identifiers
# (after the ones already listed) and to that version's CHANGELOG entry; the tests require all three.
ZENODO_CONCEPT_DOI = "10.5281/zenodo.24505557"
ZENODO_VERSION_DOIS = {"1.0.0": "10.5281/zenodo.24505558", "1.1.0": "10.5281/zenodo.24506134"}
ZENODO_DOIS = {ZENODO_CONCEPT_DOI, *ZENODO_VERSION_DOIS.values()}
ZENODO_DOI_RE = re.compile(r"10\.5281/zenodo\.[^\s\"'`)\]}<>,;]*")


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
        for rel in DOC_FILES + OTHER_FILES + PAGE_FILES + FIGURE_FILES + ["figures/make_figures.py"]:
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
                scanned = name.endswith((".py", ".md", ".html", ".yaml", ".toml", ".cff", ".txt", ".json",
                                         ".svg", ".css", ".js"))
                if not scanned and name not in ("LICENSE", "NOTICE", ".gitignore"):
                    continue
                full = os.path.join(root, name)
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    words = set(re.findall(r"[a-z0-9_]+", f.read().lower()))
                with self.subTest(file=os.path.relpath(full, ROOT)):
                    self.assertEqual(sorted(words & private), [], "private name found")

    def test_only_public_recording_pseudonyms(self):
        for rel in DOC_FILES + PAGE_FILES + FIGURE_FILES:
            for name in re.findall(r"Farm[A-Z]_R\d+", _read(rel)):
                with self.subTest(file=rel, name=name):
                    self.assertIn(name, {"FarmA_R1", "FarmA_R2", "FarmA_R3", "FarmB_R1"})


def _html_code_lines(text: str):
    """Lines inside <pre><code> blocks of an HTML page, with entities decoded and backslash
    continuations joined."""
    import html
    for block in re.findall(r"<pre><code>(.*?)</code></pre>", text, flags=re.DOTALL):
        buf = ""
        for line in html.unescape(re.sub(r"<[^>]+>", "", block)).splitlines():
            if line.rstrip().endswith("\\"):
                buf += line.rstrip()[:-1] + " "
                continue
            yield buf + line
            buf = ""


def _command_key(words):
    return tuple(words[:2]) if tuple(words[:2]) in CLI else tuple(words[:1])


def _commands(lines):
    """(line, command key, options) for every ``openveil ...`` line."""
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line.startswith("openveil "):
            continue
        words = line.split()[1:]
        yield line, _command_key(words), {w for w in words if w.startswith("--")}


# The commands the README's quick start must show; docs/usage.md shows the rest.
README_COMMANDS = {("detect",), ("review", "build"), ("review", "serve"), ("review", "demo"),
                   ("review", "export"), ("apply", "video"), ("verify",)}


class TestCliExamples(unittest.TestCase):
    def test_commands_follow_contract(self):
        sources = [(rel, _code_lines(_read(rel))) for rel in DOC_FILES]
        sources.append(("docs/index.html", _html_code_lines(_read("docs/index.html"))))
        for rel, lines in sources:
            for line, key, opts in _commands(lines):
                with self.subTest(file=rel, line=line):
                    self.assertIn(key, CLI, f"unknown command: {line}")
                    self.assertTrue(opts <= CLI[key], f"options {sorted(opts - CLI[key])} not in contract")

    def test_every_command_is_shown(self):
        readme = {key for _l, key, _o in _commands(_code_lines(_read("README.md")))}
        usage = {key for _l, key, _o in _commands(_code_lines(_read("docs/usage.md")))}
        page = {key for _l, key, _o in _commands(_html_code_lines(_read("docs/index.html")))}
        self.assertEqual(README_COMMANDS - readme, set(), "README quick start misses some commands")
        self.assertEqual(set(CLI) - (readme | usage), set(), "README and docs/usage.md miss some commands")
        self.assertEqual(README_COMMANDS - page, set(), "project page misses some commands")


def _html_ids(text: str) -> set:
    return set(re.findall(r'\sid="([^"]+)"', text))


class TestLinksAndImages(unittest.TestCase):
    def _check_target(self, rel, target, planned, html_ids=None):
        if re.match(r"^[a-z]+:", target) or target.startswith("//"):
            return
        path, _, anchor = target.partition("#")
        path = path.split("?", 1)[0]
        base = os.path.dirname(os.path.join(ROOT, rel))
        with self.subTest(file=rel, target=target):
            full = os.path.normpath(os.path.join(base, path)) if path else os.path.join(ROOT, rel)
            is_image = path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))
            if is_image and not os.path.exists(full):
                self.assertIn(os.path.basename(path), planned, f"image {path} neither present nor planned")
                return
            self.assertTrue(os.path.exists(full), f"broken link {target}")
            if not anchor:
                return
            if full.endswith(".md"):
                with open(full, "r", encoding="utf-8") as f:
                    self.assertIn(anchor, _anchors(f.read()), f"missing anchor #{anchor}")
            elif html_ids is not None and not path:
                self.assertIn(anchor, html_ids, f"missing id #{anchor}")

    def test_relative_links_resolve(self):
        link = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
        html_attr = re.compile(r'\s(?:src|href|srcset)="([^"]+)"')
        planned = _read(PLANNED_IMAGES_README)
        for rel in DOC_FILES:
            text = re.sub(r"```.*?```", "", _read(rel), flags=re.DOTALL)   # code blocks
            text = re.sub(r"`[^`\n]*`", "", text)                            # inline code
            for target in link.findall(text) + html_attr.findall(text):
                self._check_target(rel, target, planned)
        # In the README, the HTML anchor nav must point at real headings.
        readme = _read("README.md")
        for anchor in re.findall(r'href="#([^"]+)"', readme):
            with self.subTest(readme_anchor=anchor):
                self.assertIn(anchor, _anchors(readme))

    def test_project_page_links_resolve(self):
        planned = _read(PLANNED_IMAGES_README)
        text = _read("docs/index.html")
        ids = _html_ids(text)
        targets = re.findall(r'\s(?:src|href|srcset)="([^"]+)"', text)
        self.assertGreater(len(targets), 20)
        for target in targets:
            self._check_target("docs/index.html", target, planned, html_ids=ids)
            with self.subTest(target=target):
                # GitHub Pages serves docs/ as the site root: nothing may point outside it
                self.assertFalse(target.startswith(("../", "/")), "link leaves the docs/ site root")
                self.assertFalse(target.startswith("http:"), "insecure link")

    def test_project_page_is_self_contained(self):
        text = _read("docs/index.html")
        self.assertEqual(re.findall(r'<script[^>]*\ssrc="([^"]+)"', text), ["site.js"])
        self.assertEqual(re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"', text), ["style.css"])
        self.assertIn('name="viewport"', text)
        self.assertIn("prefers-color-scheme: dark", _read("docs/style.css"))
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "docs", ".nojekyll")))
        # every approved still is shown on the page and in the README, and is listed in docs/img/README.md
        stills = sorted(n for n in os.listdir(os.path.join(ROOT, "docs", "img")) if n.endswith(".jpg"))
        self.assertEqual(len(stills), 8)
        for name in stills:
            with self.subTest(still=name):
                self.assertIn(f'src="img/{name}"', text)
                self.assertIn(f"docs/img/{name}", _read("README.md"))
                self.assertIn(name, _read(PLANNED_IMAGES_README))
        # every image on the page has alt text
        for tag in re.findall(r"<img\b[^>]*>", text):
            with self.subTest(img=tag[:60]):
                self.assertRegex(tag, r'\salt="')

    def test_review_screenshots_are_referenced(self):
        tutorial = _read("docs/review_tool.md")
        for name in ("review_overview.png", "review_adopt_candidate.png", "review_edit_box.png",
                     "review_magnifier.png", "review_unsure_note.png"):
            with self.subTest(image=name):
                self.assertIn(f"img/{name}", tutorial)
                self.assertIn(name, _read(PLANNED_IMAGES_README))

    def test_markdown_tables_are_not_split(self):
        # A table row after a blank line, without a header separator under it, renders as a
        # paragraph of literal pipes on GitHub.
        for rel in DOC_FILES:
            text = re.sub(r"```.*?```", "", _read(rel), flags=re.DOTALL)
            lines = text.splitlines()
            for i in range(1, len(lines) - 1):
                if lines[i - 1].strip() == "" and lines[i].lstrip().startswith("|"):
                    with self.subTest(file=rel, line=lines[i][:60]):
                        self.assertRegex(lines[i + 1].strip(), r"^\|[\s:|-]*-[\s:|-]*\|$")

    def test_mermaid_diagrams(self):
        self.assertIn("```mermaid\nflowchart", _read("docs/how_it_works.md"))


class TestFigures(unittest.TestCase):
    """The drawn SVG figures: up to date, accessible, copied for the project page."""

    def test_figures_are_up_to_date(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("make_figures", os.path.join(ROOT, "figures", "make_figures.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        made = {"banner.svg": mod.banner(), "pipeline.svg": mod.pipeline(), "pipeline_narrow.svg": mod.pipeline_narrow()}
        self.assertEqual(sorted(made), sorted(FIGURES))
        for name, svg in made.items():
            for rel in (f"figures/{name}", f"docs/assets/{name}"):
                with self.subTest(file=rel):
                    self.assertEqual(_read(rel).rstrip("\n"), svg, "run python figures/make_figures.py")

    def test_figures_are_accessible(self):
        for rel in FIGURE_FILES:
            text = _read(rel)
            with self.subTest(file=rel):
                self.assertTrue(text.startswith("<svg "))
                self.assertIn('role="img"', text)
                self.assertRegex(text, r"<title[^>]*>[^<]{5,}</title>")
                self.assertRegex(text, r"<desc[^>]*>[^<]{20,}</desc>")
                self.assertNotRegex(text, r"<(?:image|script|foreignObject)\b")   # drawn only, nothing embedded

    def test_figures_are_not_ignored_by_git(self):
        import shutil
        import subprocess

        if not os.path.isdir(os.path.join(ROOT, ".git")) or not shutil.which("git"):
            self.skipTest("not a git checkout")
        paths = FIGURE_FILES + PAGE_FILES + ["docs/.nojekyll", "figures/make_figures.py"]
        r = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "", "ignored by .gitignore")


class TestConfigReference(unittest.TestCase):
    """The configuration tables of docs/configuration.md against openveil.config (skipped if not importable)."""

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
        part = _read(CONFIG_REFERENCE).split("\n## ", 1)[0]
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
        # no DOI is invented: a DOI is added only once Zenodo has assigned one to a release
        dois = [d.strip("'\"") for d in re.findall(r"(?mi)^\s*doi:\s*(\S+)", text)]
        self.assertEqual(dois, [ZENODO_CONCEPT_DOI])
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
        for a in data["authors"]:
            self.assertEqual(a["orcid"], "https://orcid.org/" + ORCIDS[(a["given-names"], a["family-names"])])
        self.assertEqual(str(data["version"]), _package_version())
        self.assertRegex(str(data["date-released"]), r"^\d{4}-\d{2}-\d{2}$")
        # the top-level DOI is the concept DOI; the identifiers list it and the archived version DOIs
        self.assertEqual(data["doi"], ZENODO_CONCEPT_DOI)
        ids = [(i["type"], str(i["value"])) for i in data["identifiers"]]
        self.assertEqual(ids, [("doi", ZENODO_CONCEPT_DOI)]
                         + [("doi", d) for d in ZENODO_VERSION_DOIS.values()])
        for ident in data["identifiers"]:
            self.assertTrue(ident.get("description"))
        for version, doi in ZENODO_VERSION_DOIS.items():
            desc = next(i["description"] for i in data["identifiers"] if i["value"] == doi)
            self.assertIn(f"Version {version}", desc)


def _package_version() -> str:
    m = re.search(r'(?m)^__version__ = "([^"]+)"$', _read("openveil/__init__.py"))
    return m.group(1)


class TestReleaseMetadata(unittest.TestCase):
    """Zenodo reads .zenodo.json in preference to CITATION.cff, so the two must agree."""

    def test_zenodo_json(self):
        import json
        data = json.loads(_read(".zenodo.json"))
        self.assertEqual(data["upload_type"], "software")
        self.assertEqual(data["version"], _package_version())
        self.assertEqual(data["license"], "apache-2.0")
        self.assertRegex(data["publication_date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual([c["name"] for c in data["creators"]], ["Yang, Haiyu", "Hostens, Miel"])
        for c in data["creators"]:
            family, given = c["name"].split(", ")
            self.assertEqual(c["orcid"], ORCIDS[(given, family)])
            self.assertIn("Cornell University", c["affiliation"])
        self.assertNotIn("doi", data)                  # Zenodo assigns the DOI at release
        self.assertNotIn("prereserve_doi", data)
        ids = {r["identifier"]: r for r in data["related_identifiers"]}
        self.assertEqual(ids["10.57967/hf/11689"]["scheme"], "doi")
        self.assertEqual(ids["arXiv:2611.21706"]["scheme"], "arxiv")
        self.assertEqual(ids["https://huggingface.co/" + WEIGHTS_REPO]["scheme"], "url")

    def test_zenodo_json_matches_citation_cff(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        import json
        cff = yaml.safe_load(_read("CITATION.cff"))
        zen = json.loads(_read(".zenodo.json"))
        self.assertEqual(zen["title"], cff["title"])
        self.assertEqual(zen["version"], str(cff["version"]))
        self.assertEqual(zen["publication_date"], str(cff["date-released"]))
        self.assertEqual(zen["keywords"], cff["keywords"])
        self.assertEqual([c["name"] for c in zen["creators"]],
                         [f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]])
        self.assertEqual([c["affiliation"] for c in zen["creators"]], [a["affiliation"] for a in cff["authors"]])

    def test_zenodo_json_is_not_ignored_by_git(self):
        gitignore = _read(".gitignore")
        self.assertIn("!/.zenodo.json", gitignore)

    def test_changelog_lists_the_current_version(self):
        text = _read("CHANGELOG.md")
        version = _package_version()
        self.assertRegex(text, rf"(?m)^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$")
        self.assertIn(f"Version {version}", _read("README.md"))
        # the newest entry is the current version, and it has a compare link
        self.assertEqual(re.findall(r"(?m)^## \[([^\]]+)\]", text)[0], version)
        self.assertRegex(text, rf"(?m)^\[{re.escape(version)}\]: https://github\.com/Sonam525/OpenVeil/compare/v\S+\.\.\.v{re.escape(version)}$")
        # every archived version DOI is named in the changelog entry of its version
        for v, doi in ZENODO_VERSION_DOIS.items():
            entry = text.split(f"## [{v}]", 1)[1].split("\n## [", 1)[0]
            self.assertIn(doi, entry)

    def test_versions_agree_across_files(self):
        version = _package_version()
        readme, page = _read("README.md"), _read("docs/index.html")
        self.assertIn(f"**Release:** {version} (", readme)
        self.assertIn(f"(Version {version}) [Computer software]", readme)
        self.assertIn(f"OpenVeil {version} ·", page)
        self.assertIn(f"Current release: {version} (", page)
        for rel, text in (("README.md", readme), ("docs/index.html", page)):
            with self.subTest(file=rel):
                entry = re.search(r"@software\{yang2026openveil,.*?\n\}", text, re.DOTALL).group(0)
                self.assertRegex(entry, rf"(?m)^\s*version\s*=\s*\{{{re.escape(version)}\}},$")
                self.assertRegex(entry, rf"(?m)^\s*doi\s*=\s*\{{{re.escape(ZENODO_CONCEPT_DOI)}\}},$")
                self.assertRegex(entry, r"(?m)^\s*publisher\s*=\s*\{Zenodo\},$")
                # no other version is named as the current one
                for other in re.findall(r"(?:OpenVeil|Version|Release:\*\*|release:) (\d+\.\d+\.\d+)", text):
                    self.assertEqual(other, version)

    def test_concept_doi_is_cited(self):
        doi = ZENODO_CONCEPT_DOI
        readme = _read("README.md")
        self.assertIn(f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)](https://doi.org/{doi})", readme)
        self.assertIn(f"https://doi.org/{doi}", _read("docs/index.html"))
        self.assertIn(f"Zenodo. https://doi.org/{doi}", readme)   # the APA reference
        for rel in ("README.md", "docs/index.html", "CITATION.cff"):
            with self.subTest(file=rel):
                self.assertNotRegex(_read(rel), r"(?i)DOI[^.]{0,60}\b(to be added|assigned by Zenodo at release)")

    def test_zenodo_dois_are_real(self):
        # every Zenodo DOI in the repository text is one Zenodo has assigned: no placeholders
        for rel in DOC_FILES + OTHER_FILES + PAGE_FILES:
            for doi in ZENODO_DOI_RE.findall(_read(rel)):
                doi = re.sub(r"\.svg$", "", doi).rstrip(".")     # badge URL, end of sentence
                with self.subTest(file=rel, doi=doi):
                    self.assertRegex(doi, r"^10\.5281/zenodo\.\d+$")
                    self.assertIn(doi, ZENODO_DOIS)


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
