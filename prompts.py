"""
prompts.py — templated minimal-pair prompt generation.

One base instruction, four genuinely-different paraphrases, every cue toggled
one token-cluster at a time. Everything else is held byte-identical so any
behavioral difference is attributable to the cue, not the wording.

A "cell" is a fully-specified combination of cue levels. expand_cell() turns a
cell + paraphrase id into the final prompt string.

Design notes / confounds deliberately avoided:
  * We ask for a SINGLE self-contained interface in every condition. This helps
    the auto-scorer compile the interface in isolation; it does NOT push toward
    closures vs. data-as-behavior, so it is not a confound for the hypothesis.
  * The structural situation (a C++ iterator; run user code per element) is
    IDENTICAL across the numeric and neutral domains. Only the flavor differs.
  * `recognition=offered` adds exactly one neutral clause naming closures as
    "one option among others" — used to separate "doesn't know" from "doesn't propose".
"""

from itertools import product
from dataclasses import dataclass, asdict
import hashlib


# ----------------------------------------------------------------------------
#  Per-language surface vocabulary (what the deliverable is called, how a
#  "function passed per element" reads natively). Kept minimal and parallel.
# ----------------------------------------------------------------------------
LANG = {
    "ocaml":  dict(name="OCaml",  artifact="the .mli interface file and the C stubs",
                   closure="a real OCaml function/closure"),
    "python": dict(name="Python", artifact="the Python-facing API and the binding glue",
                   closure="a real Python function/callable"),
    "julia":  dict(name="Julia",  artifact="the Julia-facing API and the ccall glue",
                   closure="a real Julia function"),
    "lua":    dict(name="Lua",    artifact="the Lua-facing API and the C stubs",
                   closure="a real Lua function"),
    "c":      dict(name="C",      artifact="the public C header and the wrapper",
                   closure="a real C callback (function pointer + void* userdata)"),
}

# Domain flavor. Structurally identical: C++ iterator over ELEMs, run user code per ELEM.
DOMAIN = {
    "numeric": dict(lib="BlockSci",
                    desc="a high-performance blockchain analytics engine",
                    elem="block", elems="blocks"),
    "neutral": dict(lib="libpcapng",
                    desc="a network protocol parser",
                    elem="frame", elems="frames"),
}

# Nuisance / placebo factor: an irrelevant token swap that should NOT move behavior.
PLACEBO = {"a": "Bitcoin", "b": "Litecoin"}  # only surfaces in numeric domain; no-op otherwise


def _perf_sentence(perf, d):
    if perf == "present":
        return (f" The collection has on the order of 800 million {d['elems']}, "
                f"and this path is performance-critical, so the per-{d['elem']} step is hot.")
    return ""


def _polars_sentence(polars):
    if polars == "present":
        return (" For inspiration, think about how libraries like Polars or pandas "
                "let users express per-row work.")
    return ""


def _recog_sentence(recognition, lng):
    if recognition == "offered":
        return f" (Note: passing {lng['closure']} to be invoked per element is one option among others.)"
    return ""


def _placebo_clause(domain, placebo):
    # Irrelevant to API shape. Only attaches in the numeric domain so the two
    # domains stay otherwise parallel; in neutral domain it contributes nothing.
    if domain == "numeric":
        return f" (the deployment indexes {PLACEBO[placebo]})"
    return ""


# ----------------------------------------------------------------------------
#  Four paraphrases. Same slots, different register / sentence order / framing.
#  Slots: {LNAME} {LIB} {DESC} {ELEM} {ELEMS} {PLACEBO} {PERF} {POLARS} {ART} {RECOG}
# ----------------------------------------------------------------------------
TEMPLATES = {
    "p0": (
        "I'm writing {LNAME} bindings to {LIB}{PLACEBO}, {DESC} that exposes a C++ "
        "iterator over {ELEMS}. I want {LNAME} users to be able to run a piece of their "
        "own code for each {ELEM} as the iterator walks the collection.{PERF}{POLARS} "
        "Please design the binding and give me {ART} as a single self-contained unit.{RECOG}"
    ),
    "p1": (
        "Task: design {LNAME} bindings for {LIB}{PLACEBO} — {DESC} whose C++ API hands you "
        "an iterator over {ELEMS}. The point is that {LNAME} users supply the body that "
        "runs per {ELEM}.{PERF}{POLARS} Produce {ART}; keep it self-contained.{RECOG}"
    ),
    "p2": (
        "I need help binding {LIB}{PLACEBO} ({DESC}) to {LNAME}. The C++ side gives an "
        "iterator that walks {ELEMS}; on the {LNAME} side a user wants their own logic to "
        "fire for every {ELEM}.{PERF}{POLARS} Give me {ART}, as one self-contained piece.{RECOG}"
    ),
    "p3": (
        "How would you build {LNAME} bindings over {LIB}{PLACEBO}? It's {DESC}; its C++ "
        "interface exposes an iterator across {ELEMS}, and {LNAME} callers need to attach "
        "their own per-{ELEM} behavior.{PERF}{POLARS} I'm after {ART}, self-contained.{RECOG}"
    ),
}


@dataclass(frozen=True)
class Cell:
    lang: str
    perf: str
    domain: str
    polars: str
    recognition: str
    placebo: str

    def id(self) -> str:
        s = "|".join(f"{k}={v}" for k, v in asdict(self).items())
        return hashlib.sha1(s.encode()).hexdigest()[:10]


def expand_cell(cell: Cell, paraphrase: str) -> str:
    lng = LANG[cell.lang]
    d = DOMAIN[cell.domain]
    return TEMPLATES[paraphrase].format(
        LNAME=lng["name"], ART=lng["artifact"],
        LIB=d["lib"], DESC=d["desc"], ELEM=d["elem"], ELEMS=d["elems"],
        PLACEBO=_placebo_clause(cell.domain, cell.placebo),
        PERF=_perf_sentence(cell.perf, d),
        POLARS=_polars_sentence(cell.polars),
        RECOG=_recog_sentence(cell.recognition, lng),
    )


# ----------------------------------------------------------------------------
#  PLANS — which subset of the full factorial to actually run.
#  The full grid (160 cells) is rarely what you want in one shot.
# ----------------------------------------------------------------------------
def _cells(lang, perf, domain, polars, recognition, placebo):
    return [Cell(*combo) for combo in product(lang, perf, domain, polars, recognition, placebo)]


def plan_cells(plan: str, grid: dict):
    """Return the list of Cells for a named plan."""
    if plan == "full":
        return _cells(grid["lang"], grid["perf"], grid["domain"],
                      grid["polars"], grid["recognition"], grid["placebo"])

    if plan == "core":
        # Pre-registered core. Holds the spine on OCaml and walks each cue, plus
        # a Python ceiling and a lang sweep at the "loudest" cue setting.
        cells = []
        # (1) Within-OCaml single-factor sweep around a clean baseline.
        #     baseline = perf absent, neutral, no polars, default recognition, placebo a
        base = dict(perf="absent", domain="neutral", polars="absent",
                    recognition="default", placebo="a")
        def ocaml(**over):
            kw = dict(base); kw.update(over); return Cell(lang="ocaml", **kw)
        cells += [
            ocaml(),                              # baseline
            ocaml(perf="present"),                # +perf
            ocaml(domain="numeric"),              # +numeric domain
            ocaml(polars="present"),              # +polars cue
            ocaml(recognition="offered"),         # recognition probe
            ocaml(placebo="b"),                   # placebo (should be ~no-op)
            ocaml(perf="present", domain="numeric", polars="present"),  # all-loud
        ]
        # (2) Positive control: Python at baseline and all-loud (expect data/kernel to dominate).
        cells += [
            Cell(lang="python", **base),
            Cell(lang="python", perf="present", domain="numeric",
                 polars="present", recognition="default", placebo="a"),
        ]
        # (3) Target-conditionality: hold the all-loud task, sweep language.
        for lg in ("julia", "lua", "c"):
            cells.append(Cell(lang=lg, perf="present", domain="numeric",
                              polars="present", recognition="default", placebo="a"))
        # de-dup while preserving order
        seen, out = set(), []
        for c in cells:
            if c.id() not in seen:
                seen.add(c.id()); out.append(c)
        return out

    if plan == "balanced_ocaml":
        # Fully crossed OCaml cues (2^5 = 32 cells) so EVERY main effect —
        # including the placebo null — is cleanly estimable in the GLM, with each
        # level appearing across many cells. Plus the Python ceiling and the
        # language sweep at the all-loud setting. Use THIS plan for odds ratios;
        # use `core` for a quick descriptive rate table.
        cells = _cells(["ocaml"], grid["perf"], grid["domain"],
                       grid["polars"], grid["recognition"], grid["placebo"])
        base = dict(perf="absent", domain="neutral", polars="absent",
                    recognition="default", placebo="a")
        cells.append(Cell(lang="python", **base))
        cells.append(Cell(lang="python", perf="present", domain="numeric",
                          polars="present", recognition="default", placebo="a"))
        for lg in ("julia", "lua", "c"):
            cells.append(Cell(lang=lg, perf="present", domain="numeric",
                              polars="present", recognition="default", placebo="a"))
        return cells

    raise ValueError(f"unknown plan: {plan!r}")


PLAN_NAMES = ("core", "balanced_ocaml", "full")
